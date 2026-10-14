"""Paper 1 (methods paper, decision gate outcome of Step 5): every figure and every number quoted in the
manuscript that is not already in 05_Results/step5_*.csv is produced here.

Writes 06_Manuscript/figures/fig1..fig4 (PDF and PNG), 06_Manuscript/paper_numbers.json and supp_tables.tex.
Uses only the processed panel (04_Processed_Data) and the Step 5 outputs; nothing is typed by hand.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from . import analysis as A
from . import config

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300})
SEED = config.RANDOM_SEED


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- pitfall 1: shared terms
def shared_term_null(df, n_perm=1000):
    """The 2026 draft regressed pi1 = M/(D A) on pi2 = G/(D A E). Rebuild both from raw country means with the
    real outcome, then with the outcome shuffled across countries (no information left)."""
    raw = df[df.in_sample].groupby("iso3")[["M", "D_wdi", "A_R", "Y", "G_co2_per_gdp"]].mean()
    raw = raw[(raw > 0).all(axis=1)].dropna()
    lnM = np.log(raw.M.values)
    lnDA = np.log(raw.D_wdi * raw.A_R).values
    lnpi2 = np.log(raw.G_co2_per_gdp / (raw.D_wdi * raw.A_R * raw.Y)).values
    def r2(lm):
        return stats.linregress(lnpi2, lm - lnDA).rvalue ** 2
    def slope(lm):
        return stats.linregress(lnpi2, lm - lnDA).slope
    rng = np.random.default_rng(SEED)
    null_r2, null_b = [], []
    for _ in range(n_perm):
        s = rng.permutation(lnM)
        null_r2.append(r2(s)); null_b.append(slope(s))
    # the same question asked without a shared term: does ln G explain ln M once D, A, Y enter on their own?
    X = sm.add_constant(np.column_stack([np.log(raw.G_co2_per_gdp), np.log(raw.D_wdi), np.log(raw.A_R),
                                         np.log(raw.Y)]))
    sep = sm.OLS(lnM, X).fit(cov_type="HC1")
    return {"n_countries": int(len(raw)), "r2_observed": float(r2(lnM)), "slope_observed": float(slope(lnM)),
            "r2_null_median": float(np.median(null_r2)), "r2_null_p05": float(np.percentile(null_r2, 5)),
            "r2_null_p95": float(np.percentile(null_r2, 95)), "slope_null_median": float(np.median(null_b)),
            "p_perm_r2": float((1 + np.sum(np.array(null_r2) >= r2(lnM))) / (1 + n_perm)),
            "separate_terms_lnG_beta": float(sep.params[1]), "separate_terms_lnG_p": float(sep.pvalues[1])}, \
        np.array(null_r2), lnpi2, lnM - lnDA


# ---------------------------------------------------------------- pitfall 2: min-max scaling
def minmax_bias(b=1.5, sds=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0), n=150, sd_noise=0.3, reps=1000):
    """y = x^b * noise, x log-normal with spread sd (SD of ln x). Fit ln y on ln x on raw values, then after
    min-max scaling both to 0-10 (rows that scale to exactly 0 are dropped, since ln 0 is undefined).
    The bias depends on spread: min-max subtracts the minimum, which matters most when values are bunched."""
    rng = np.random.default_rng(SEED)
    rows = []
    for sd in sds:
        raw_b, mm_b = [], []
        for _ in range(reps):
            x = np.exp(rng.normal(0, sd, n))
            y = x ** b * np.exp(rng.normal(0, sd_noise * sd, n))
            raw_b.append(stats.linregress(np.log(x), np.log(y)).slope)
            xs = 10 * (x - x.min()) / (x.max() - x.min())
            ys = 10 * (y - y.min()) / (y.max() - y.min())
            k = (xs > 0) & (ys > 0)
            mm_b.append(stats.linregress(np.log(xs[k]), np.log(ys[k])).slope)
        rows.append({"true": b, "sd_lnx": sd, "raw_median": float(np.median(raw_b)),
                     "minmax_median": float(np.median(mm_b)), "minmax_p05": float(np.percentile(mm_b, 5)),
                     "minmax_p95": float(np.percentile(mm_b, 95))})
    return pd.DataFrame(rows)


def spreads(df):
    """Cross-country spread (SD of ln country mean) of raw variables, to place real data on panel c."""
    m = df[df.in_sample].groupby("iso3")[["Y", "D_wdi", "Np", "A_L"]].mean()
    return {k: float(np.log(m[k]).std()) for k in m.columns}


def fig1(out, null_r2, lnpi2, lnpi1, st, mm):
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1, 1, 1]})
    a = ax[0]
    a.scatter(lnpi2, lnpi1, s=9, color=BLUE, edgecolor="white", linewidth=0.4)
    a.set_xlabel(r"ln $\pi_2$ = ln $G/(DAE)$"); a.set_ylabel(r"ln $\pi_1$ = ln $M/(DA)$")
    a.set_title("a  Ratios sharing D and A", loc="left", fontsize=8, fontweight="bold")
    a.text(0.03, 0.97, f"$R^2$ = {st['r2_observed']:.2f}", transform=a.transAxes, va="top", color=INK)
    a = ax[1]
    a.hist(null_r2, bins=30, color=GREY, edgecolor="white", linewidth=0.4)
    a.axvline(st["r2_observed"], color=ORANGE, lw=1.5)
    a.set_xlim(0, max(0.5, null_r2.max() * 1.05))
    a.text(st["r2_observed"], a.get_ylim()[1] * 0.97, " real\n outcome", color=INK, va="top", fontsize=7)
    a.set_xlabel("$R^2$ with outcome shuffled across countries"); a.set_ylabel("permutations")
    a.set_title("b  Outcome shuffled", loc="left", fontsize=8, fontweight="bold")
    a = ax[2]
    a.axhline(mm.true.iloc[0], color=GREY, lw=0.8, ls="--")
    a.fill_between(mm.sd_lnx, mm.minmax_p05, mm.minmax_p95, color=ORANGE, alpha=0.18, lw=0)
    a.plot(mm.sd_lnx, mm.minmax_median, "o-", color=ORANGE, ms=3.5, lw=1.2, label="after 0-10 min-max scaling")
    a.plot(mm.sd_lnx, mm.raw_median, "o-", color=BLUE, ms=3.5, lw=1.2, label="raw units")
    a.set_xscale("log"); a.set_xticks([0.1, 0.2, 0.5, 1, 2]); a.set_xticklabels(["0.1", "0.2", "0.5", "1", "2"])
    a.set_xlabel("spread of x across countries (SD of ln x)"); a.set_ylabel("fitted exponent (true = 1.5)")
    a.set_ylim(0, 1.8)
    a.legend(loc="lower right", fontsize=7, handletextpad=0.3)
    a.set_title("c  Min-max scaling", loc="left", fontsize=8, fontweight="bold")
    fig.tight_layout(w_pad=1.5)
    _save(fig, out, "fig1_pitfalls")


# ---------------------------------------------------------------- application figures
def fig2(out, res_dir):
    null = pd.read_csv(res_dir / "step5_permutation_null.csv").iloc[:, -1].values
    df, _ = A.load(res_dir.parent)
    d = A._drop_all_zero(df[df.in_sample].dropna(subset=A.MAIN + ["M", "Np"]))
    fit = lambda c: sm.GLM(d.M.values, A._design(d, c), family=sm.families.Poisson(), offset=d.ln_Np.values).fit(maxiter=200)
    obs = fit([]).deviance - fit(A.MAIN).deviance
    p = (1 + (null >= obs).sum()) / (1 + len(null))
    fig, a = plt.subplots(figsize=(3.4, 2.4))
    bins = np.logspace(np.log10(null.min()), np.log10(null.max()), 35)
    a.hist(null, bins=bins, color=GREY, edgecolor="white", linewidth=0.4)
    a.axvline(obs, color=ORANGE, lw=1.5)
    a.set_xscale("log")
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xticks([2e6, 5e6, 1e7, 2e7, 5e7]); a.set_xticklabels(["2", "5", "10", "20", "50"])
    a.text(obs, a.get_ylim()[1] * 0.95, f" observed\n p = {p:.2f}", color=INK, va="top", fontsize=7)
    a.set_xlabel("deviance reduction from adding the five groups (millions)")
    a.set_ylabel("block permutations")
    fig.tight_layout()
    _save(fig, out, "fig2_permutation")
    return {"h1_obs": float(obs), "h1_null_p95": float(np.percentile(null, 95)), "h1_null_median": float(np.median(null)),
            "h1_p": float(p), "h1_n_perm": int(len(null)), "h1_share_null_above_obs": float((null >= obs).mean())}


def fig3(out, df):
    per_c, c = A._forecast(df, A.MAIN, config.TRAIN_END, range(config.TRAIN_END + 1, config.YEARS[1] + 1))
    cols = [("err_persist", "persistence, 2019 value", BLUE), ("err_pred_rate", "five-group model", ORANGE),
            ("err_hist_mean", "country mean, 2008-2019", AQUA)]
    fig, a = plt.subplots(figsize=(3.4, 2.5))
    for k, lab, col in cols:
        v = np.sort(per_c[k].values)
        a.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=col, lw=1.5,
               label=f"{lab} (median {np.median(v):.2f})")
    a.axhline(0.5, color=GREY, lw=0.6, ls=":")
    a.set_xlabel("mean absolute log error per country, 2020-2023"); a.set_ylabel("share of countries")
    a.legend(loc="lower right", fontsize=6.5)
    fig.tight_layout()
    _save(fig, out, "fig3_forecast")
    win = {k: float((per_c.err_pred_rate < per_c[k]).mean()) for k in ["err_persist", "err_hist_mean"]}
    return {"h3_n": int(len(per_c)), "h3_c": float(c), "h3_model": float(per_c.err_pred_rate.median()),
            "h3_persist": float(per_c.err_persist.median()), "h3_hist": float(per_c.err_hist_mean.median()),
            "h3_share_model_beats_persist": win["err_persist"], "h3_share_model_beats_hist": win["err_hist_mean"]}


def fig4(out, df):
    res, m = A.between(df)
    tr = A.index_transfer(df)
    fig, a = plt.subplots(figsize=(3.4, 2.6))
    pos = m.obs_rate > 0
    a.scatter(m.index_rate[pos], m.obs_rate[pos], s=9, color=BLUE, edgecolor="white", linewidth=0.4)
    lo, hi = 1e-6, 0.2
    a.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel("index: fitted displacement rate per year"); a.set_ylabel("observed mean rate 2008-2023")
    rho, p = stats.spearmanr(m.index_rate, m.obs_rate)
    a.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.2f} (in sample)", transform=a.transAxes, va="top", fontsize=7)
    fig.tight_layout()
    _save(fig, out, "fig4_index")
    return {"between_n": int(len(m)), "between_n_zero_obs": int((~pos).sum()), "between_rho_in_sample": float(rho),
            "between_pseudo_r2": float(1 - res.deviance / res.null_deviance)}


def h2_interval(df):
    d = df[df.in_sample].copy()
    d["ln_H"] = np.log(d.HY * d.Y); d["ln_Y"] = np.log(d.Y); d["ln_K1"] = np.log(d.K + 1)
    d["ln_AL"] = np.log(d.A_L); d["ln_AR"] = np.log(d.A_R)
    cols = ["ln_Np", "ln_H", "ln_Y", "ln_K1", "ln_AL", "ln_AR", "pi3", "pi4"]
    res, used, _ = A.ppml(d, cols, offset=False, country_fe=False, year_fe=True)
    names = list(res.params.index)
    R = np.zeros(len(names))
    for v in ["ln_Np", "ln_K1", "ln_AL", "ln_AR"]:
        R[names.index(v)] = 1.0
    s = float(R @ res.params.values); se = float(np.sqrt(R @ res.cov_params().values @ R))
    return {"h2_sum": s, "h2_se": se, "h2_ci_low": s - 1.96 * se, "h2_ci_high": s + 1.96 * se,
            "h2_n": int(len(used)), "h2_countries": int(used.iso3.nunique()),
            "h2_gamma": {v: float(res.params[v]) for v in ["ln_Np", "ln_K1", "ln_AL", "ln_AR", "ln_H", "ln_Y"]}}


def sample_facts(df):
    s = df[df.in_sample]
    main = A.ppml(s, A.MAIN)[1]
    return {"countries_in_sample": int(s.iso3.nunique()), "country_years_in_sample": int(len(s)),
            "main_countries": int(main.iso3.nunique()), "main_obs": int(len(main)),
            "share_zero_M": float((s.M == 0).mean()), "total_M_millions": float(s.M.sum() / 1e6),
            "share_M_top1pct": float(s.M[s.M >= s.M.quantile(0.99)].sum() / s.M.sum())}


def run(project_root):
    root = Path(project_root)
    out = root / "06_Manuscript" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    df, eps = A.load(root)
    raw = pd.read_csv(config.paths(root)["processed"] / "panel_raw_units.csv", usecols=["iso3", "year", "D_wdi"])
    dfr = df.merge(raw, on=["iso3", "year"], how="left")
    st, null_r2, x, y = shared_term_null(dfr)
    mm = minmax_bias()
    fig1(out, null_r2, x, y, st, mm)
    nums = {"shared_term": st, "minmax": mm.to_dict(orient="records"), "spreads": spreads(dfr), "eps_pi2": float(eps)}
    nums.update(fig2(out, root / "05_Results"))
    nums.update(fig3(out, df))
    nums.update(fig4(out, df))
    nums.update(h2_interval(df))
    nums.update(sample_facts(df))
    (root / "06_Manuscript" / "paper_numbers.json").write_text(json.dumps(nums, indent=2))
    supp_tables(root)
    print(json.dumps(nums, indent=2))
    return nums


NAMES = {"ln_pi1": r"$\ln\Pi_1$ health share", "ln_pi2e": r"$\ln(\Pi_2+\varepsilon)$ conflict rate",
         "pi3": r"$\Pi_3$ temperature anomaly", "pi4": r"$\Pi_4$ precipitation anomaly",
         "ln_pi6": r"$\ln\Pi_6$ relative income", "ln_G": r"$\ln G$ carbon intensity"}


def _n(x, k=3):
    return f"${x:.{k}f}$"


def _p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def supp_tables(project_root):
    """Supplementary tables S1 (main model) and S2 (robustness), written straight from the Step 5 CSVs."""
    root = Path(project_root)
    res = root / "05_Results"
    c = pd.read_csv(res / "step5_coefficients.csv", index_col=0)
    L = [r"\begin{table}[h]", r"\centering\small",
         r"\caption{Main model, equation (1): PPML with population offset, country and year effects, standard errors "
         r"clustered by country (149 countries, 2,366 country-years).}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Group & $\beta$ & SE & $p$ & 95\% CI \\", r"\midrule"]
    for v, r in c.iterrows():
        L.append(f"{NAMES[v]} & {_n(r.beta)} & {_n(r.se)} & {_p(r.p)} & {_n(r.ci_low, 2)} to {_n(r.ci_high, 2)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    rb = pd.read_csv(res / "step5_robustness.csv")
    rb = rb[rb.specification != "main"]
    L += [r"\begin{small}", r"\setlength{\LTcapwidth}{\textwidth}",
          r"\begin{longtable}{@{}p{0.42\textwidth}p{0.27\textwidth}rrr@{}}",
          r"\caption{Pre-registered robustness specifications. Coefficients with $p < 0.05$ are marked with an asterisk.}\\",
          r"\toprule", r"Specification & Group & $\beta$ & SE & $p$ \\", r"\midrule", r"\endhead"]
    for spec, g in rb.groupby("specification", sort=False):
        first = True
        for _, r in g.iterrows():
            star = "$^{*}$" if r.p < 0.05 else ""
            name = spec.replace("&", r"\&").replace("%", r"\%") if first else ""
            L.append(f"{name} & {NAMES[r.variable]} & {_n(r.beta)}{star} & {_n(r.se)} & {_p(r.p)} \\\\")
            first = False
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{longtable}", r"\end{small}"]
    (root / "06_Manuscript" / "supp_tables.tex").write_text("\n".join(L))
