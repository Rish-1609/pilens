"""Paper 0: main-text figures that combine panels, the updated methodology flowchart (March Figure 2, same layout),
in-sample fit plots (March SI Figs S26-S27) and every LaTeX table, written from the outputs of paper0.run().
Nothing in the tables is typed by hand."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import paper0 as P

NICE = {"pi1 group": r"$\pi_1$ group", "pi2 group": r"$\pi_2$ group", "pi5 group": r"$\pi_5$ group"}
VLAB = {"M": "M (persons)", "D": "D (persons/km$^2$)", "E": "E (\\$/person/yr)", "P": "P (deaths/yr)",
        "S": "S (\\$/person/yr)", "G": "G (kg CO$_2$/\\$)", "W": "W (mm/yr)", "F": "F ($^\\circ$C)",
        "O": "O (share)", "A": "A (km$^2$)"}


def n(x, k=2):
    if pd.isna(x):
        return "--"
    return f"${x:.{k}f}$"


def sci(x):
    if pd.isna(x):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -2 <= e <= 4:
        return f"{x:,.2f}" if abs(x) < 1000 else f"{x:,.0f}"
    return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"


def gname(g):
    return NICE.get(g, g)


# ------------------------------------------------------------------ figures
def methodology_figure(out):
    """March Figure 2, same boxes and colours, with the rebuilt steps."""
    boxes = [
        ("Data collection\n(M, P, E, D, S, G, W, F, O, A) from logged public sources\n196 countries, 2008-2023, real units", "#a8d8ff"),
        ("Data preprocessing\n3-year moving averages (14 windows)\nSix Pi groups in real units; reference-scaled where needed", "#a0e6f0"),
        ("K-means clustering (k = 7)", "#e6e7ea"),
        ("Case 1: pi-based features (pi1 ... pi6, one at a time)\nCase 2: traditional variables (var-N, min-max scaled 0-10)", "#ffe89a"),
        ("Temporal validation against documented events\n2008-2010, 2013-2015, 2018-2020, 2021-2023", "#ffc4c4"),
        ("Case A: pi1, pi2, pi5 groups | Case B: African coastal, islands,\nhigh/low GDP, high inflow/outflow (fixed country lists)", "#d4c8ff"),
        ("Statistical analysis\nPearson correlation heatmaps; self-similarity plots\nTemporal stability; data-collapse test; geographic coherence", "#ead9d6"),
        ("Power laws and forecasting\nLeast squares on logs and PPML count model\nFit 2014-2016, predict 2018-2020\nPersistence baseline; permutation nulls", "#ffd9a8"),
        ("Framework assessment\nWhat the dimensionless groups do and do not capture", "#b8f0bf"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 8.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    h, gap = 0.086, 0.024
    ys = [1 - 0.02 - i * (h + gap) - h for i in range(len(boxes))]
    for (txt, col), y in zip(boxes, ys):
        ax.add_patch(FancyBboxPatch((0.06, y), 0.88, h, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc=col, ec="#222222", lw=0.8))
        ax.text(0.5, y + h / 2, txt, ha="center", va="center", fontsize=7.2, family="DejaVu Sans Mono")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        ax.annotate("", xy=(0.5, y1 + h), xytext=(0.5, y0), arrowprops=dict(arrowstyle="->", lw=0.9, color="#222222"))
    P._save(fig, out, "fig2_methodology")


def fig3_main(cl, out):
    wd = P.world()
    fig, axes = plt.subplots(3, 1, figsize=(5.6, 8.4))
    for a, (feat, title) in zip(axes, [("pi1", r"(a) $\pi_1$ clustering, 2018-2020"),
                                       ("pi5", r"(b) $\pi_5$ clustering, 2018-2020"),
                                       ("varN", "(c) var-N clustering, 2018-2020")]):
        g = cl[(cl.feature == feat) & (cl.window == "2018-2020")]
        mm = wd.merge(g, on="iso3", how="left")
        mm[mm.cluster.isna()].plot(ax=a, color="#ecebe7", edgecolor="white", linewidth=0.2)
        pal = P.CAT if feat == "varN" else P.SEQ
        for c in range(P.K):
            sub = mm[mm.cluster == c]
            if len(sub):
                sub.plot(ax=a, color=pal[c], edgecolor="white", linewidth=0.2)
        a.set_axis_off(); a.set_title(title, fontsize=8, loc="left")
        if feat == "varN":
            hs = [plt.Rectangle((0, 0), 1, 1, color=P.CAT[c]) for c in range(P.K)]
            a.legend(hs, [f"{c + 1}" for c in range(P.K)], title="cluster", ncol=7, fontsize=6, title_fontsize=6,
                     loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False, handlelength=1)
        else:
            hs = [plt.Rectangle((0, 0), 1, 1, color=P.SEQ[c]) for c in range(P.K)]
            a.legend(hs, ["lowest", "", "", "", "", "", "highest"], ncol=7, fontsize=6, loc="lower center",
                     bbox_to_anchor=(0.5, -0.12), frameon=False, handlelength=1, columnspacing=0.6)
    fig.tight_layout()
    P._save(fig, out, "fig3_maps_2018")


def fit_figure(m, giso, case, out, fname, window=(2018, 2020)):
    keys = [k for k in giso if k[0] == case]
    fig, axes = plt.subplots(1, len(keys), figsize=(2.3 * len(keys), 2.3))
    for a, key in zip(np.atleast_1d(axes), keys):
        s = P.group_rows(m, giso[key])
        f = P.fit_group(s[s.window == P.wl(window)])
        if f is None:
            a.set_axis_off(); continue
        d = f["data"]
        pred = f["ols"].fittedvalues
        a.scatter(d.ln_pi1, pred, s=9, color=P.BLUE, linewidth=0)
        lo, hi = min(d.ln_pi1.min(), pred.min()) - 0.5, max(d.ln_pi1.max(), pred.max()) + 0.5
        a.plot([lo, hi], [lo, hi], color=P.GREY, lw=0.8, ls="--")
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_title(f"{key[1]} (R$^2$ = {f['r2']:.2f})", fontsize=7, loc="left")
        a.set_xlabel(r"observed ln $\pi_1$", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel(r"fitted ln $\pi_1$", fontsize=7)
    fig.tight_layout()
    P._save(fig, out, fname)


# ------------------------------------------------------------------ tables
def table_sources():
    rows = [
        ("M", "Migration", "Refugees and asylum seekers originating from the country plus those hosted by it (stock, mid-year)", "persons", "UNHCR Refugee Data Finder API"),
        ("D", "Demographic pressure", "Population density", "persons/km$^2$", "World Bank WDI EN.POP.DNST"),
        ("E", "Economic capacity", "GDP per capita, PPP, constant 2021 \\$", "\\$/person/yr", "WDI NY.GDP.PCAP.PP.KD"),
        ("P", "Conflict intensity", "Deaths from organized violence (best estimate)", "deaths/yr", "UCDP v26.1"),
        ("S", "Social resources", "Current health expenditure per person (share of GDP $\\times$ E)", "\\$/person/yr", "WDI SH.XPD.CHEX.GD.ZS"),
        ("G", "Carbon intensity", "Territorial CO$_2$ emissions per unit of GDP (PPP)", "kg CO$_2$/\\$", "OWID CO$_2$ and WDI GDP"),
        ("W", "Precipitation", "Annual precipitation", "mm/yr", "CRU TS (World Bank CCKP)"),
        ("F", "Temperature rise", "Temperature change against 1951-1980", "$^\\circ$C", "FAOSTAT; CRU TS where FAOSTAT is missing"),
        ("O", "Coastal exposure", "Share of population below 5 m elevation (2015)", "share", "CIESIN LECZ v3 via WDI"),
        ("A", "Land availability", "Arable land", "km$^2$", "WDI AG.LND.ARBL.HA (FAO)"),
    ]
    L = [r"\begin{table}[htbp]", r"\caption{Data sources and parametrization. Every file is logged with its address, download date and SHA-256 hash.}\label{tabS1}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llp{5.2cm}lp{3.2cm}}", r"\toprule",
         r"Symbol & Variable & Definition & Unit & Source \\", r"\midrule"]
    for r in rows:
        L.append(" & ".join(r) + r" \\")
    L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_descriptive(res):
    d = pd.read_csv(res / "tableS2_descriptive.csv", index_col=0)
    L = [r"\begin{table}[htbp]", r"\caption{Descriptive statistics of the 3-year window means, all countries and windows (2008-2010 to 2021-2023), in real units.}\label{tabS2}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrrr}", r"\toprule",
         r"Variable & N & Mean & Median & SD & Min & Max \\", r"\midrule"]
    for v, r in d.iterrows():
        lab = VLAB.get(v, f"$\\pi_{v[-1]}$" if v.startswith("pi") else v)
        L.append(f"{lab} & {int(r['count'])} & {sci(r['mean'])} & {sci(r['50%'])} & {sci(r['std'])} & {sci(r['min'])} & {sci(r['max'])} \\\\")
    L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_groups(res, giso, label0=4):
    d = pd.read_csv(res / "tablesS4_S12_group_statistics.csv")
    out, i = [], label0
    for (case, g), iso in giso.items():
        s = d[(d.case == case) & (d.group == g)]
        names = ", ".join(P.GROUPS[(case, g)])
        L = [r"\begin{table}[htbp]",
             f"\\caption{{Case {case}: descriptive statistics for the {gname(g)} (window means 2008-2023, real units). Countries: {names}.}}\\label{{tabS{i}}}",
             r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}lrrrrr}", r"\toprule",
             r"Variable & Mean & Median & SD & Min & Max \\", r"\midrule"]
        for _, r in s.iterrows():
            L.append(f"{VLAB[r.variable]} & {sci(r['mean'])} & {sci(r['median'])} & {sci(r['sd'])} & {sci(r['min'])} & {sci(r['max'])} \\\\")
        L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
        out.append("\n".join(L))
        i += 1
    return out


def table_powerlaw(res, case, label, caption, method="ols", windows=None):
    d = pd.read_csv(res / "table1_S13_S14_power_laws.csv")
    d = d[d.case.isin(list(case))]
    if windows:
        d = d[d.window.isin(windows)]
    pre = "ols_" if method == "ols" else "ppml_"
    head = r"Period & Group & n & " + (r"$\ln C$ & " if method == "ols" else "") + \
        r"$\ln\pi_2$ & $\ln\pi_3$ & $\ln\pi_4$ & $\ln\pi_5$ & $\ln\pi_6$ & " + \
        (r"$R^2$ & null 95\% & $p$ \\" if method == "ols" else r"pseudo-$R^2$ \\")
    ncol = 13 if method == "ols" else 9
    L = [r"\begin{table}[htbp]", f"\\caption{{{caption}}}\\label{{{label}}}", r"\footnotesize",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}ll" + "r" * (ncol - 2) + "}", r"\toprule", head, r"\midrule"]
    for g, gg in d.groupby("group", sort=False):
        for _, r in gg.iterrows():
            cells = [r.window.replace("-", "--"), gname(g), str(int(r.n))]
            if method == "ols":
                cells.append(n(r.ols_lnC))
            cells += [n(r[f"{pre}ln_pi{k}"]) for k in range(2, 7)]
            if method == "ols":
                cells += [n(r.r2_ols), n(r.r2_null95), n(r.p_perm, 3)]
            else:
                cells += [n(r.pseudo_r2_ppml)]
            L.append(" & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\botrule"
    L += [r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_forecast(res):
    d = pd.read_csv(res / "forecast_2014_16_to_2018_20.csv")
    L = [r"\begin{table}[htbp]", r"\caption{Forecast test: equations fitted on 2014-2016 predict $\pi_1$ in 2018-2020. Error = $|\hat\pi_1-\pi_1|/\pi_1$. Persistence carries each country's 2014-2016 value forward.}\label{tabS16}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrr}", r"\toprule",
         r"Case & Group & n & Median error, power law & Median error, persistence & Share where power law is better \\", r"\midrule"]
    for _, r in d.iterrows():
        L.append(f"{r.case} & {gname(r.group)} & {int(r.n)} & {r.median_err_model * 100:.0f}\\% & {r.median_err_persist * 100:.0f}\\% & {r.share_model_better * 100:.0f}\\% \\\\")
    L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_collapse(res):
    d = pd.read_csv(res / "self_similarity_collapse_test.csv")
    L = [r"\begin{table}[htbp]", r"\caption{Data-collapse test of self-similarity: one equation for the four shown windows against a separate equation per window (F-test, least squares on logs).}\label{tabS18}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrr}", r"\toprule",
         r"Case & Group & $F$ & df & $p$ & Same exponents rejected at 5\%? \\", r"\midrule"]
    for _, r in d.iterrows():
        L.append(f"{r.case} & {gname(r.group)} & {r.F:.2f} & {int(r.df1)}, {int(r.df2)} & {r.p:.3f} & {'no' if r.p >= 0.05 else 'yes'} \\\\")
    L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_coherence(res):
    d = pd.read_csv(res / "cluster_coherence.csv")
    w = d[d.window.isin([P.wl(x) for x in P.SHOW])].pivot(index="window", columns="feature", values="ari_region")
    cols = ["pi1", "pi2", "pi3", "pi4", "pi5", "pi6", "varN"]
    L = [r"\begin{table}[htbp]", r"\caption{Geographic coherence of each clustering: adjusted Rand index between the seven clusters and the seven World Bank regions (0 = no relation, 1 = clusters identical to regions). Bottom row: mean over all 14 windows.}\label{tabS19}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}l" + "r" * len(cols) + "}", r"\toprule",
         "Window & " + " & ".join([f"$\\pi_{c[-1]}$" for c in cols[:-1]] + ["var-N"]) + r" \\", r"\midrule"]
    for win, r in w.iterrows():
        L.append(win.replace("-", "--") + " & " + " & ".join(f"{r[c]:.3f}" for c in cols) + r" \\")
    mean = d.groupby("feature").ari_region.mean()
    L += [r"\midrule", "All windows & " + " & ".join(f"{mean[c]:.3f}" for c in cols) + r" \\",
          r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_idmc(res):
    d = pd.read_csv(res / "power_laws_idmc.csv")
    f = pd.read_csv(res / "forecast_idmc.csv")
    L = [r"\begin{table}[htbp]", r"\caption{Second outcome: IDMC disaster displacement in place of UNHCR refugees and asylum seekers in $\pi_1$. Least squares fit $R^2$ by window (with the 95th percentile of the shared-term null) and the 2014-2016 to 2018-2020 forecast.}\label{tabS20}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}llrrrrr}", r"\toprule",
         r"Case & Group & $R^2$ 2008--10 & $R^2$ 2013--15 & $R^2$ 2018--20 & Median error, power law & Median error, persistence \\", r"\midrule"]
    for (c, g), gg in d.groupby(["case", "group"], sort=False):
        r2 = {r.window: f"{r.r2_ols:.2f} ({r.r2_null95:.2f})" for _, r in gg.iterrows()}
        ff = f[(f.case == c) & (f.group == g)]
        e1 = f"{ff.median_err_model.iloc[0] * 100:.0f}\\%" if len(ff) else "--"
        e2 = f"{ff.median_err_persist.iloc[0] * 100:.0f}\\%" if len(ff) else "--"
        L.append(f"{c} & {gname(g)} & {r2.get('2008-2010', '--')} & {r2.get('2013-2015', '--')} & {r2.get('2018-2020', '--')} & {e1} & {e2} \\\\")
    L += [r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def table_pooled(res):
    d = json.loads((res / "paper0_numbers.json").read_text())["pooled_permutation"]
    L = [r"\begin{table}[htbp]", r"\caption{Pooled test across all countries: share of deviance explained by the five groups in a count model with window effects, against the reference in which each country's $\pi_1$ series is given to another country.}\label{tabS17}",
         r"\footnotesize", r"\begin{tabular*}{\textwidth}{@{\extracolsep\fill}rrrrrr}", r"\toprule",
         r"Countries & Country-windows & Observed share & Reference median & Reference 95th percentile & $p$ \\", r"\midrule",
         f"{d['countries']} & {d['rows']} & {d['obs']:.3f} & {d['null_median']:.3f} & {d['null_p95']:.3f} & {d['p']:.3f} \\\\",
         r"\botrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(L)


def run(project_root):
    root = Path(project_root)
    res = root / "05_Results" / "paper0"
    fig = root / "06_Manuscript_Paper0" / "figures"
    tex = root / "06_Manuscript_Paper0" / "tables"
    tex.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(res / "paper0_windows_pi.csv")
    cl = pd.read_csv(res / "clusters.csv")
    giso = P.group_iso()
    methodology_figure(fig)
    fig3_main(cl, fig)
    fit_figure(m, giso, "A", fig, "fit_caseA")
    fit_figure(m, giso, "B", fig, "fit_caseB")
    (tex / "tabS1.tex").write_text(table_sources())
    (tex / "tabS2.tex").write_text(table_descriptive(res))
    (tex / "tabS4_S12.tex").write_text("\n\n".join(table_groups(res, giso)))
    show3 = ["2008-2010", "2013-2015", "2018-2020"]
    (tex / "tab1.tex").write_text(table_powerlaw(
        res, "A", "tab1", r"Power law coefficients for Case A (Equation~\ref{eq2}), least squares on logs, with $R^2$, the 95th percentile of $R^2$ when $M$ is shuffled among the group's countries (shared-term null) and the permutation $p$-value.",
        windows=show3))
    (tex / "tabS13.tex").write_text(table_powerlaw(res, "A", "tabS13", r"Power law coefficients for Case A, all four windows (least squares on logs; null and $p$ as in Table~1)."))
    (tex / "tabS14.tex").write_text(table_powerlaw(res, "B", "tabS14", r"Power law coefficients for Case B, all four windows (least squares on logs; null and $p$ as in Table~1)."))
    (tex / "tabS15.tex").write_text(table_powerlaw(res, "AB", "tabS15", r"Count model (PPML of $M$ with offset $\ln(DA)$, robust standard errors): exponents for Cases A and B.", method="ppml"))
    (tex / "tabS16.tex").write_text(table_forecast(res))
    (tex / "tabS17.tex").write_text(table_pooled(res))
    (tex / "tabS18.tex").write_text(table_collapse(res))
    (tex / "tabS19.tex").write_text(table_coherence(res))
    (tex / "tabS20.tex").write_text(table_idmc(res))
    print("tables and extra figures written")
