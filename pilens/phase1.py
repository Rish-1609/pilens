"""Study 2, Phase 1 tests P1 to P4 and robustness (STUDY2_PHASE1_PLAN.md, D-026).

Reads 04_Processed_Data/exposure_shares.csv (from exposure.collect) and the Study 1 panel.
Writes 05_Results/phase1_RESULTS.md and phase1_coefficients.csv.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import analysis as A
from . import config

BASE = ["ln_pi1", "ln_pi2e", "ln_pi6"]
YEARS = (2008, 2018)
TRAIN_END, TEST = 2015, range(2016, 2019)


def load(project_root, F="F", C="C100", D="D15"):
    df, _ = A.load(project_root)
    ex = pd.read_csv(config.paths(project_root)["processed"] / "exposure_shares.csv")
    df = df[df.year.between(*YEARS)].merge(ex[["iso3", "year", f"share_{F}", f"share_{C}", f"share_{D}"]],
                                           on=["iso3", "year"], how="left")
    for name, col in (("F", f"share_{F}"), ("C", f"share_{C}")):
        pos = df.loc[df[col] > 0, col]
        eps = pos.min() if len(pos) else 1e-9
        df[f"ln_{name}"] = np.log(df[col] + eps)
    df["D"] = df[f"share_{D}"]
    return df


EXPO = ["ln_F", "ln_C", "D"]


def p1(df, n_perm=config.N_PERMUTATIONS, seed=config.RANDOM_SEED):
    cols = BASE + EXPO
    d = A._drop_all_zero(df[df.in_sample].dropna(subset=cols + ["M", "Np"]))
    def stat(dd):
        fit = lambda c: sm.GLM(dd.M.values, A._design(dd, c), family=sm.families.Poisson(),
                               offset=dd.ln_Np.values).fit(maxiter=200)
        return fit(BASE).deviance - fit(cols).deviance
    obs = stat(d)
    wide = df[df.in_sample].pivot(index="iso3", columns="year", values="M")
    countries = list(wide.index)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(countries))
        s = pd.DataFrame(wide.values[perm], index=pd.Index(countries, name="iso3"), columns=wide.columns)
        s = s.stack(future_stack=True).rename("M").reset_index()
        dd = df[df.in_sample].drop(columns="M").merge(s, on=["iso3", "year"])
        dd = A._drop_all_zero(dd.dropna(subset=cols + ["M", "Np"]))
        try:
            null.append(stat(dd))
        except Exception:
            continue
    null = np.array(null)
    return obs, null, (1 + (null >= obs).sum()) / (1 + len(null))


def p2(df):
    per_c, _ = A._forecast(df, BASE + EXPO, TRAIN_END, TEST)
    m = per_c.median()
    return {"model": m.err_pred_rate, "persistence": m.err_persist, "historical_mean": m.err_hist_mean,
            "n": len(per_c), "pass": bool(m.err_pred_rate < m.err_persist and m.err_pred_rate < m.err_hist_mean)}


def p3(df, n_boot=2000):
    full, _ = A._forecast(df, BASE + EXPO, TRAIN_END, TEST)
    base, _ = A._forecast(df, BASE, TRAIN_END, TEST)
    j = full.join(base, rsuffix="_b").dropna()
    diff = j.err_pred_rate.median() - j.err_pred_rate_b.median()
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx = np.arange(len(j))
    boots = [(lambda s: s.err_pred_rate.median() - s.err_pred_rate_b.median())(j.iloc[rng.choice(idx, len(idx))])
             for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"with": j.err_pred_rate.median(), "without": j.err_pred_rate_b.median(), "diff": diff,
            "lo": lo, "hi": hi, "pass": bool(diff < 0 and hi < 0)}


def run(project_root, n_perm=config.N_PERMUTATIONS):
    p = config.paths(project_root)
    df = load(project_root)
    res, used, _ = A.ppml(df[df.in_sample], BASE + EXPO)
    coef = pd.DataFrame({"beta": res.params[BASE + EXPO], "se": res.bse[BASE + EXPO], "p": res.pvalues[BASE + EXPO]})
    obs, null, pp1 = p1(df, n_perm)
    r2, r3 = p2(df), p3(df)
    P4 = bool(coef.loc["ln_F", "beta"] > 0 and coef.loc["ln_F", "p"] < 0.05 and
              coef.loc["ln_C", "beta"] > 0 and coef.loc["ln_C", "p"] < 0.05)
    rob = []
    for lab, kw in [("cyclone 50 km", {"C": "C50"}), ("cyclone 150 km", {"C": "C150"}),
                    ("drought SPEI <= -1.0", {"D": "D10"}), ("drought SPEI <= -2.0", {"D": "D20"})]:
        r = A.ppml(load(project_root, **kw).query("in_sample"), BASE + EXPO)[0]
        rob.append({"spec": lab, **{v: round(r.params[v], 3) for v in EXPO}})
    d = df[df.in_sample].copy()
    for v in ["ln_F", "ln_C"]:
        d[f"{v}_x_inc"] = d[v] * d.ln_pi6
    ri = A.ppml(d, BASE + EXPO + ["ln_F_x_inc", "ln_C_x_inc"])[0]
    gate = ("P1 fails: no signal from physical exposure either; rethink hazard side before Phase 2" if pp1 >= 0.05 else
            "P1 passes, P2 fails: Phase 2 builds a screening (ranking) index" if not r2["pass"] else
            "P1 and P2 pass: Phase 2 builds the regional index with physical exposure as the core hazard group")
    coef.to_csv(p["results"] / "phase1_coefficients.csv")
    L = ["# Study 2 Phase 1: physical exposure (pre-registered, D-026)", "",
         f"Sample: {used.iso3.nunique()} countries, {len(used)} country-years, 2008-2018.", "",
         "## Model: PPML, population offset, country and year effects, country-clustered SE", "",
         coef.round(4).to_markdown(), "",
         "| Test | Result | Pass rule | Pass? |", "|---|---|---|---|",
         f"| P1 signal (block permutation, {len(null)} draws) | deviance reduction {obs:.0f}; null 95th pct {np.percentile(null, 95):.0f}; p = {pp1:.4f} | p < 0.05 | {'PASS' if pp1 < 0.05 else 'FAIL'} |",
         f"| P2 forecast 2016-2018 | model {r2['model']:.3f}; persistence {r2['persistence']:.3f}; historical mean {r2['historical_mean']:.3f} ({r2['n']} countries) | beats both | {'PASS' if r2['pass'] else 'FAIL'} |",
         f"| P3 exposure adds value | error with {r3['with']:.3f}, without {r3['without']:.3f}; difference {r3['diff']:.4f} [95% CI {r3['lo']:.4f}, {r3['hi']:.4f}] | falls, CI excludes 0 | {'PASS' if r3['pass'] else 'FAIL'} |",
         f"| P4 direction | gF = {coef.loc['ln_F','beta']:.3f} (p {coef.loc['ln_F','p']:.3f}); gC = {coef.loc['ln_C','beta']:.3f} (p {coef.loc['ln_C','p']:.3f}) | both > 0, p < 0.05 | {'PASS' if P4 else 'FAIL'} |",
         "", f"**Decision gate: {gate}**", "", "## Robustness", "",
         pd.DataFrame(rob).to_markdown(index=False), "",
         f"Income interaction: ln_F x ln_pi6 = {ri.params['ln_F_x_inc']:.3f} (p {ri.pvalues['ln_F_x_inc']:.3f}); "
         f"ln_C x ln_pi6 = {ri.params['ln_C_x_inc']:.3f} (p {ri.pvalues['ln_C_x_inc']:.3f})", "",
         "Benchmarks against IDMC GDRM and INFORM are added once those files are downloaded."]
    (p["results"] / "phase1_RESULTS.md").write_text("\n".join(L))
    print("\n".join(L))
    return gate
