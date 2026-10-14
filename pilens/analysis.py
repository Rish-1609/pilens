"""Step 5: the pre-registered tests (ANALYSIS_PLAN sections 7-8, decisions D-019 and D-022 to D-025).

Reads 04_Processed_Data/panel_raw_units.csv and panel_pi_groups.csv. Writes 05_Results/step5_RESULTS.md,
step5_coefficients.csv, step5_index_by_country.csv and step5_permutation_null.csv.
Nothing here is tuned after seeing results: every choice is fixed in the plan or the decision log.
"""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import config

warnings.filterwarnings("ignore")
MAIN = ["ln_pi1", "ln_pi2e", "pi3", "pi4", "ln_pi6"]          # plan 7.1
CLIMATE = ["pi3", "pi4"]


# ------------------------------------------------------------------ data
def load(project_root, pi3_source="cru"):
    p = config.paths(project_root)
    raw = pd.read_csv(p["processed"] / "panel_raw_units.csv")
    pis = pd.read_csv(p["processed"] / "panel_pi_groups.csv")
    df = pis.merge(raw[["iso3", "year", "HY", "Y", "A_L", "A_R", "R", "M_absent_set_zero", "G_co2_per_gdp"]],
                   on=["iso3", "year"], how="left")
    pop15 = df[df.year == 2015].set_index("iso3")["Np"]
    df["in_sample"] = df.iso3.isin(pop15[pop15 >= config.MIN_POP_2015].index)
    eps = df.loc[df.pi2_conflict_rate > 0, "pi2_conflict_rate"].min()          # plan 7.1: smallest positive Pi2
    df["ln_pi1"] = np.log(df.pi1_health_share)
    df["ln_pi2e"] = np.log(df.pi2_conflict_rate + eps)
    df["pi3"] = df.pi3_temp_anomaly if pi3_source == "cru" else df.pi3_faostat_check
    df["pi4"] = df.pi4_precip_anomaly
    df["ln_pi6"] = np.log(df.pi6_relative_income)
    df["ln_pi7"] = np.log(df.pi7_arable_share)
    df["ln_pi8"] = np.log(df.pi8_relative_density)
    df["pi5"] = df.pi5_lecz_share
    df["ln_Np"] = np.log(df.Np)
    return df, eps


def _design(d, cols, country_fe=True, year_fe=True):
    X = d[cols].astype(float).copy()
    if country_fe:
        X = X.join(pd.get_dummies(d.iso3, prefix="c", drop_first=True, dtype=float))
    if year_fe:
        X = X.join(pd.get_dummies(d.year, prefix="y", drop_first=True, dtype=float))
    return sm.add_constant(X, has_constant="add")


def _drop_all_zero(d, y="M"):
    """PPML with country effects: a country whose outcome is always 0 has no information (standard practice)."""
    keep = d.groupby("iso3")[y].transform("sum") > 0
    return d[keep]


def ppml(d, cols, y="M", offset=True, country_fe=True, year_fe=True, cluster=True):
    d = d.dropna(subset=cols + [y, "Np"])
    if country_fe:
        d = _drop_all_zero(d, y)
    X = _design(d, cols, country_fe, year_fe)
    off = d.ln_Np.values if offset else None
    mod = sm.GLM(d[y].values.astype(float), X, family=sm.families.Poisson(), offset=off)
    if cluster:
        codes = pd.factorize(d.iso3)[0]
        res = mod.fit(cov_type="cluster", cov_kwds={"groups": codes}, maxiter=200)
    else:
        res = mod.fit(maxiter=200)
    return res, d, X


# ------------------------------------------------------------------ H1
def h1_permutation(df, n_perm=config.N_PERMUTATIONS, seed=config.RANDOM_SEED):
    d = df[df.in_sample].dropna(subset=MAIN + ["M", "Np"])
    d = _drop_all_zero(d)
    def stat(dd):
        base = sm.GLM(dd.M.values, _design(dd, [], True, True), family=sm.families.Poisson(),
                      offset=dd.ln_Np.values).fit(maxiter=200)
        full = sm.GLM(dd.M.values, _design(dd, MAIN, True, True), family=sm.families.Poisson(),
                      offset=dd.ln_Np.values).fit(maxiter=200)
        return base.deviance - full.deviance
    obs = stat(d)
    # block permutation: each country's whole outcome series (by year) goes to another country (plan 7.2)
    wide = df[df.in_sample].pivot(index="iso3", columns="year", values="M")
    countries = wide.index.values
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(countries))
        shuffled = pd.DataFrame(wide.values[perm], index=pd.Index(list(countries), name="iso3"), columns=wide.columns)
        shuffled = shuffled.stack(future_stack=True).rename("M").reset_index()
        dd = df[df.in_sample].drop(columns="M").merge(shuffled, on=["iso3", "year"])
        dd = _drop_all_zero(dd.dropna(subset=MAIN + ["M", "Np"]))
        try:
            null.append(stat(dd))
        except Exception:
            continue
    null = np.array(null)
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    return obs, null, p


# ------------------------------------------------------------------ H2
def h2_homogeneity(df):
    d = df[df.in_sample].copy()
    d["ln_H"] = np.log(d.HY * d.Y)
    d["ln_Y"] = np.log(d.Y)
    d["ln_K1"] = np.log(d.K + 1)
    d["ln_AL"] = np.log(d.A_L)
    d["ln_AR"] = np.log(d.A_R)
    cols = ["ln_Np", "ln_H", "ln_Y", "ln_K1", "ln_AL", "ln_AR", "pi3", "pi4"]
    res, dd, X = ppml(d, cols, offset=False, country_fe=False, year_fe=True)
    R = np.zeros(len(res.params)); names = list(res.params.index)
    for v in ["ln_Np", "ln_K1", "ln_AL", "ln_AR"]:
        R[names.index(v)] = 1.0                     # D-023: gamma_Np + gamma_K + gamma_AL + gamma_AR = 1
    w = res.wald_test((R, np.array([1.0])), scalar=True)
    return res, float(np.squeeze(w.statistic)), float(w.pvalue), float(R @ res.params.values)


# ------------------------------------------------------------------ H3 / H4
def _forecast(df, cols, train_end, test_years):
    d = df[df.in_sample].dropna(subset=cols + ["M", "Np"])
    tr = d[d.year <= train_end]
    te = d[d.year.isin(test_years)]
    tr_fit = _drop_all_zero(tr)
    res, used, X = ppml(tr_fit, cols, cluster=False)
    b = res.params
    ce = {c.replace("c_", ""): v for c, v in b.items() if c.startswith("c_")}
    first_c = sorted(set(used.iso3))[0]
    ce[first_c] = 0.0
    ye = [v for c, v in b.items() if c.startswith("y_")] + [0.0]           # reference year effect = 0
    yavg = float(np.mean(ye))
    zero_c = set(tr.iso3) - set(tr_fit.iso3)
    te = te[te.iso3.isin(set(tr.iso3))].copy()
    lin = b["const"] + te.iso3.map(ce).fillna(np.nan) + yavg + te[cols].values @ b[cols].values
    te["pred_rate"] = np.where(te.iso3.isin(zero_c), 0.0, np.exp(lin))
    te["obs_rate"] = te.M / te.Np
    last = tr[tr.year == train_end].set_index("iso3")["pi0_displacement_rate"]
    mean = tr.groupby("iso3")["pi0_displacement_rate"].mean()
    te["persist"] = te.iso3.map(last)
    te["hist_mean"] = te.iso3.map(mean)
    pos = tr.loc[tr.pi0_displacement_rate > 0, "pi0_displacement_rate"]
    c = float(np.percentile(pos, 5))
    te = te.dropna(subset=["pred_rate", "persist", "hist_mean"])
    for k in ["pred_rate", "persist", "hist_mean"]:
        te[f"err_{k}"] = np.abs(np.log(te[k] + c) - np.log(te.obs_rate + c))
    per_c = te.groupby("iso3")[[f"err_{k}" for k in ["pred_rate", "persist", "hist_mean"]]].mean()
    return per_c, c


def h3(df, train_end, test_years):
    per_c, c = _forecast(df, MAIN, train_end, test_years)
    med = per_c.median()
    return {"model": med.err_pred_rate, "persistence": med.err_persist, "historical_mean": med.err_hist_mean,
            "n_countries": len(per_c), "c": c,
            "pass": bool(med.err_pred_rate < med.err_persist and med.err_pred_rate < med.err_hist_mean)}


def h4(df, train_end=config.TRAIN_END, test_years=range(config.TRAIN_END + 1, config.YEARS[1] + 1), n_boot=2000):
    full, _ = _forecast(df, MAIN, train_end, test_years)
    nocl, _ = _forecast(df, [v for v in MAIN if v not in CLIMATE], train_end, test_years)
    j = full.join(nocl, rsuffix="_nc").dropna()
    diff_obs = j.err_pred_rate.median() - j.err_pred_rate_nc.median()       # < 0 means climate helps
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = np.arange(len(j)); boots = []
    for _ in range(n_boot):
        s = j.iloc[rng.choice(idx, len(idx), replace=True)]
        boots.append(s.err_pred_rate.median() - s.err_pred_rate_nc.median())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"with_climate": j.err_pred_rate.median(), "without_climate": j.err_pred_rate_nc.median(),
            "difference": diff_obs, "ci_low": lo, "ci_high": hi, "pass": bool(diff_obs < 0 and hi < 0)}


# ------------------------------------------------------------------ between-country model and index (D-025)
BETWEEN = ["ln_pi1", "ln_pi2e", "pi3", "pi4", "pi5", "ln_pi6", "ln_pi7", "ln_pi8"]


def country_means(df):
    d = df[df.in_sample].dropna(subset=BETWEEN + ["M", "Np"])
    g = d.groupby(["iso3", "country", "region"])
    m = g[BETWEEN + ["M", "Np"]].mean().reset_index()
    m["ln_Np"] = np.log(m.Np)
    m["obs_rate"] = m.M / m.Np
    return m


def between(df):
    m = country_means(df)
    X = sm.add_constant(m[BETWEEN].join(pd.get_dummies(m.region, prefix="r", drop_first=True, dtype=float)))
    res = sm.GLM(m.M.values, X, family=sm.families.Poisson(), offset=m.ln_Np.values).fit(cov_type="HC1")
    m["index_rate"] = res.predict(X, offset=np.zeros(len(m)))
    return res, m


def index_transfer(df):
    m = country_means(df)
    out = []
    for r in sorted(m.region.unique()):
        tr, te = m[m.region != r], m[m.region == r]
        X = sm.add_constant(tr[BETWEEN])
        res = sm.GLM(tr.M.values, X, family=sm.families.Poisson(), offset=tr.ln_Np.values).fit()
        pred = res.predict(sm.add_constant(te[BETWEEN], has_constant="add"), offset=np.zeros(len(te)))
        rho, p = stats.spearmanr(pred, te.obs_rate) if len(te) > 3 else (np.nan, np.nan)
        out.append({"held_out_region": r.strip(), "countries": len(te), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ robustness (plan section 8)
def robustness(df):
    rows = []
    def add(name, res):
        for v in MAIN:
            if v in res.params:
                rows.append({"specification": name, "variable": v, "beta": res.params[v], "se": res.bse[v],
                             "p": res.pvalues[v]})
    s = df[df.in_sample]
    add("main", ppml(s, MAIN)[0])
    add("all countries", ppml(df, MAIN)[0])
    r = s.dropna(subset=["R"]).copy(); r["M"] = r.R.round()
    add("UNHCR outcome (refugees + asylum seekers)", ppml(r, MAIN)[0])
    t = s.copy(); t["block"] = ((t.year - 2008) // 3)
    t = t[t.year <= 2022].groupby(["iso3", "block"]).agg({**{v: "mean" for v in MAIN}, "M": "mean", "Np": "mean"}).reset_index()
    t["year"] = t.block; t["ln_Np"] = np.log(t.Np); t["M"] = t.M.round()
    add("three-year averages", ppml(t, MAIN)[0])
    q = s[s.M <= s.M.quantile(0.99)]
    add("excluding top 1% of country-years", ppml(q, MAIN)[0])
    a = s.copy(); a.loc[a.M_absent_set_zero == True, "M"] = np.nan
    add("absent IDMC records treated as missing", ppml(a, MAIN)[0])
    for reg in sorted(s.region.dropna().unique()):
        add(f"drop region: {reg.strip()}", ppml(s[s.region != reg], MAIN)[0])
    g = s.dropna(subset=["G_co2_per_gdp"]).copy(); g["ln_G"] = np.log(g.G_co2_per_gdp.where(g.G_co2_per_gdp > 0))
    resg = ppml(g, MAIN + ["ln_G"])[0]
    add("plus carbon intensity (old paper's variable)", resg)
    rows.append({"specification": "plus carbon intensity (old paper's variable)", "variable": "ln_G",
                 "beta": resg.params["ln_G"], "se": resg.bse["ln_G"], "p": resg.pvalues["ln_G"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ run everything
def run(project_root, n_perm=config.N_PERMUTATIONS):
    p = config.paths(project_root)
    df, eps = load(project_root)
    out = []
    res, used, _ = ppml(df[df.in_sample], MAIN)
    coef = pd.DataFrame({"beta": res.params[MAIN], "se": res.bse[MAIN], "p": res.pvalues[MAIN],
                         "ci_low": res.conf_int().loc[MAIN, 0], "ci_high": res.conf_int().loc[MAIN, 1]})
    print("Main model fitted:", len(used), "country-years,", used.iso3.nunique(), "countries")
    obs, null, p1 = h1_permutation(df, n_perm)
    print(f"H1 done: p = {p1:.4f}")
    h2res, w, p2, lhs = h2_homogeneity(df)
    print(f"H2 done: p = {p2:.4f}")
    f3 = h3(df, config.TRAIN_END, range(config.TRAIN_END + 1, config.YEARS[1] + 1))
    f3r = h3(df, 2015, range(2016, 2020))
    r4 = h4(df)
    print("H3, H4 done")
    # D-019: FAOSTAT temperature
    dff, _ = load(project_root, pi3_source="faostat")
    obs_f, null_f, p1f = h1_permutation(dff, n_perm)
    r4f = h4(dff)
    resf = ppml(dff[dff.in_sample], MAIN)[0]
    print("FAOSTAT robustness done")
    rob = robustness(df)
    bres, idx = between(df)
    tr = index_transfer(df)
    # decision gate (plan section 9)
    H1, H2, H3 = p1 < 0.05, p2 >= 0.05, f3["pass"]
    if not H1:
        gate = "H1 fails: methods paper, 'When do dimensionless indices carry signal?' -> Environmental Research Communications"
    elif not H3:
        gate = "H1 passes, H3 fails: screening framework with stated limits -> Environmental Modelling & Software"
    elif not H2:
        gate = "H1 and H3 pass, H2 fails: predictive framework without a 'law' claim -> Earth's Future"
    else:
        gate = "H1, H2 and H3 pass: tested dimensional relationship -> Earth's Future or Communications Earth & Environment"

    coef.to_csv(p["results"] / "step5_coefficients.csv")
    rob.to_csv(p["results"] / "step5_robustness.csv", index=False)
    pd.DataFrame({"null_deviance_reduction": null}).to_csv(p["results"] / "step5_permutation_null.csv", index=False)
    idx[["iso3", "country", "region", "obs_rate", "index_rate"] + BETWEEN].sort_values("index_rate", ascending=False) \
        .to_csv(p["results"] / "step5_index_by_country.csv", index=False)
    tr.to_csv(p["results"] / "step5_index_transfer.csv", index=False)
    bcoef = pd.DataFrame({"beta": bres.params[BETWEEN], "se": bres.bse[BETWEEN], "p": bres.pvalues[BETWEEN]})
    fmt = lambda x: f"{x:.3f}"
    L = ["# Step 5: pre-registered tests", "",
         f"Sample: {used.iso3.nunique()} countries, {len(used)} country-years (countries with no recorded displacement in any year drop out of the country-effects model). eps for Pi2 = {eps:.3g}.", "",
         "## Main model (plan 7.1): PPML, population offset, country and year effects, country-clustered SE", "",
         coef.round(4).to_markdown(), "",
         "## Hypotheses", "",
         "| Test | Result | Pass rule | Pass? |", "|---|---|---|---|",
         f"| H1 signal (block permutation, {len(null)} draws) | deviance reduction {obs:.0f}; null 95th pct {np.percentile(null,95):.0f}; p = {p1:.4f} | p < 0.05 | {'PASS' if H1 else 'FAIL'} |",
         f"| H2 dimensional homogeneity (D-023) | gamma_Np+gamma_K+gamma_AL+gamma_AR = {lhs:.3f}; Wald = {w:.2f}; p = {p2:.4f} | not rejected at 5% | {'PASS' if H2 else 'FAIL'} |",
         f"| H3 forecast 2020-2023 (median abs log error) | model {fmt(f3['model'])}; persistence {fmt(f3['persistence'])}; historical mean {fmt(f3['historical_mean'])} ({f3['n_countries']} countries) | beats both | {'PASS' if H3 else 'FAIL'} |",
         f"| H3 rolling origin 2016-2019 | model {fmt(f3r['model'])}; persistence {fmt(f3r['persistence'])}; historical mean {fmt(f3r['historical_mean'])} | (check) | {'beats both' if f3r['pass'] else 'does not beat both'} |",
         f"| H4 climate adds value | error with climate {fmt(r4['with_climate'])}, without {fmt(r4['without_climate'])}; difference {r4['difference']:.4f} [95% CI {r4['ci_low']:.4f}, {r4['ci_high']:.4f}] | falls, CI excludes 0 | {'PASS' if r4['pass'] else 'FAIL'} |",
         "", f"**Decision gate (plan 9): {gate}**", "",
         "## Robustness: FAOSTAT temperature instead of CRU (D-019)", "",
         f"- H1: p = {p1f:.4f} ({'PASS' if p1f < 0.05 else 'FAIL'})",
         f"- H4: difference {r4f['difference']:.4f} [95% CI {r4f['ci_low']:.4f}, {r4f['ci_high']:.4f}] ({'PASS' if r4f['pass'] else 'FAIL'})",
         f"- Pi3 coefficient: {resf.params['pi3']:.4f} (SE {resf.bse['pi3']:.4f}, p = {resf.pvalues['pi3']:.4f})", "",
         "## Robustness: other specifications (plan 8)", "",
         rob.pivot_table(index="specification", columns="variable", values="beta").round(3).to_markdown(), "",
         "Standard errors and p-values for every row are in step5_robustness.csv.", "",
         "## Between-country model and migration index (plan 7.5, D-025; exploratory)", "",
         "PPML on country means, population offset, region effects, robust SE.", "",
         bcoef.round(4).to_markdown(), "",
         "Index = fitted displacement rate exp(X beta), per country in step5_index_by_country.csv.", "",
         "Leave-one-region-out transfer (model refitted without the region and without region effects):", "",
         tr.round(3).to_markdown(index=False), "",
         "Rank correlation > 0 means the index orders the held-out region's countries in the right direction."]
    (p["results"] / "step5_RESULTS.md").write_text("\n".join(L))
    print("\n".join(L[:30]))
    return {"coef": coef, "h1": p1, "h2": p2, "h3": f3, "h4": r4, "gate": gate}
