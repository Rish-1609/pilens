"""Paper 0: the March 2026 manuscript ("A Non-Dimensional Framework for Climate Migration Dynamics and Early-Stage
Policy Scoping") rebuilt on the cleaned, sourced data. Same six Pi groups, same clustering, same Case A and Case B
country groups (the March country lists), same correlation, self-similarity, power-law and forecast analyses.

What changes, and why (each change is listed in the manuscript):
- Variables come from the logged raw sources in real units (no 0-10 min-max scaling before the Pi groups).
- Expert scores are replaced by measured quantities: P = conflict deaths per year (UCDP), S = health spending per
  person per year (WDI), O (sea-level rise) by the share of people below 5 m (LECZ).
- The six formulas are kept as written. Where a formula is not dimensionless with measured units (pi2, pi3, pi4,
  pi6) it is divided by its cross-country median in the same window, which makes it dimensionless and does not
  change its exponents.
- M = refugees and asylum seekers from the country (outflow) + hosted by it (inflow), UNHCR, as in March but gross
  instead of net (a net figure can be negative and has no logarithm). IDMC disaster displacement is a second outcome.
- Power laws are fitted both as in March (least squares on logs) and with a count model (PPML, offset ln(D A)).
- Added checks: permutation nulls for the fits, persistence baseline for the forecasts, a data-collapse test for
  self-similarity, and a geographic-coherence score for the clustering comparison.

run(project_root) writes 05_Results/paper0/* and 06_Manuscript_Paper0/figures/*, and paper0_numbers.json.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from . import config

warnings.filterwarnings("ignore")
SEED = config.RANDOM_SEED
WINDOWS = [(y, y + 2) for y in range(2008, 2022)]                     # 3-year moving averages, 2008-10 ... 2021-23
SHOW = [(2008, 2010), (2013, 2015), (2018, 2020), (2021, 2023)]      # March's three windows + the new one
FIT_W, PRED_W = (2014, 2016), (2018, 2020)                            # March forecast test
K = 7                                                                 # March: k-means with 7 clusters
PIS = ["pi1", "pi2", "pi3", "pi4", "pi5", "pi6"]
LABEL = {"pi1": "migration pressure", "pi2": "economic climate burden", "pi3": "social resource accessibility",
         "pi4": "conflict-climate interaction", "pi5": "coastal vulnerability", "pi6": "climate stress ratio"}
VARN = ["M", "D", "E", "P", "S", "G", "W", "F", "O", "A"]

# The March country lists (SI Tables S4 to S12), exactly as printed.
GROUPS = {
    ("A", "pi1 group"): ["Armenia", "Azerbaijan", "Benin", "Cambodia", "Costa Rica", "Cyprus", "Ecuador", "Egypt",
                         "Honduras", "Kuwait", "Myanmar", "Nicaragua", "Pakistan", "Senegal", "Serbia", "Slovenia",
                         "Tajikistan", "Togo"],
    ("A", "pi2 group"): ["Afghanistan", "Argentina", "Bangladesh", "Brazil", "Burkina Faso", "Burundi", "Cambodia",
                         "Cameroon", "Central African Republic", "China", "Colombia", "Côte d'Ivoire", "DR Congo",
                         "Haiti", "India", "Kenya", "Sierra Leone", "Sri Lanka", "Uganda", "Zambia"],
    ("A", "pi5 group"): ["Benin", "Cameroon", "El Salvador", "Gabon", "Ghana", "Guinea-Bissau", "Guyana", "Ireland",
                         "Kenya", "Liberia", "North Korea", "Portugal", "Senegal", "Sierra Leone", "Somalia",
                         "South Korea", "Sri Lanka", "Suriname", "Togo", "Yemen"],
    ("B", "African coastal"): ["Algeria", "Angola", "Benin", "Cameroon", "Côte d'Ivoire", "DR Congo", "Egypt", "Gabon",
                               "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Liberia", "Libya", "Madagascar",
                               "Mauritania", "Morocco", "Mozambique", "Namibia", "Nigeria"],
    ("B", "High GDP"): ["Austria", "Belgium", "Canada", "Denmark", "Finland", "Germany", "Iceland", "Ireland", "Kuwait",
                        "Luxembourg", "Malta", "Netherlands", "Norway", "Qatar", "Saudi Arabia", "Singapore", "Sweden",
                        "Switzerland", "United Arab Emirates", "United States"],
    ("B", "High migration inflow"): ["Austria", "Belgium", "Canada", "France", "Germany", "Italy", "Japan",
                                     "Luxembourg", "Netherlands", "Norway", "Qatar", "Saudi Arabia", "Singapore",
                                     "South Korea", "Spain", "Sweden", "Switzerland", "United Arab Emirates",
                                     "United Kingdom", "United States"],
    ("B", "High migration outflow"): ["Afghanistan", "Bangladesh", "DR Congo", "El Salvador", "Eritrea", "Guatemala",
                                      "Haiti", "Honduras", "India", "Iraq", "Lebanon", "Mexico", "Nigeria", "Pakistan",
                                      "Somalia", "Sri Lanka", "Sudan", "Syria", "Venezuela", "Yemen"],
    ("B", "Islands"): ["Cuba", "Cyprus", "Dominican Republic", "Grenada", "Iceland", "Ireland", "Japan", "Madagascar",
                       "Maldives", "Mauritius", "Seychelles", "Singapore", "Sri Lanka", "Bahamas",
                       "Trinidad and Tobago"],
    ("B", "Low GDP"): ["Afghanistan", "Burkina Faso", "Burundi", "Central African Republic", "Chad", "DR Congo",
                       "Eritrea", "Guinea-Bissau", "Haiti", "Liberia", "Madagascar", "Malawi", "Mali", "Mozambique",
                       "Niger", "Sierra Leone", "Somalia", "South Sudan", "Togo", "Yemen"],
}

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]            # 7 ordered clusters
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]          # 7 unordered clusters
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "savefig.dpi": 300})


def wl(w):
    return f"{w[0]}-{w[1]}"


# ------------------------------------------------------------------ data
def group_iso():
    import country_converter as coco
    out = {}
    for key, names in GROUPS.items():
        iso = coco.convert([n.replace("DR Congo", "Democratic Republic of the Congo") for n in names], to="ISO3",
                           not_found=None)
        bad = [n for n, i in zip(names, iso) if i is None]
        if bad:
            raise ValueError(f"unmatched names in {key}: {bad}")
        out[key] = list(iso)
    return out


def yearly(project_root):
    p = config.paths(project_root)
    d = pd.read_csv(p["processed"] / "panel_raw_units.csv")
    inflow = p["raw"] / "readable_csv" / "unhcr_refugees_by_asylum.csv"
    if inflow.exists():
        h = pd.read_csv(inflow)
        d = d.merge(h, on=["iso3", "year"], how="left")
        has_in = True
    else:
        d["hosted"] = np.nan
        has_in = False
    y = pd.DataFrame({"iso3": d.iso3, "country": d.country, "region": d.region.str.strip(), "year": d.year,
                      "Np": d.Np})
    y["M_out"], y["M_in"] = d.R, d.hosted
    y["M"] = d[["R", "hosted"]].sum(axis=1, min_count=1) if has_in else d.R          # gross: outflow + inflow
    y["M2"] = d.M                                                                    # IDMC disaster displacement
    y["D"], y["A"], y["E"] = d.D_wdi, d.A_R, d.Y
    y["S"] = d.HY * d.Y                                                              # health spending per person
    y["P"] = d.K                                                                     # conflict deaths per year
    # G: CO2 per unit of GDP, computed for every country as OWID territorial CO2 / WDI GDP (PPP, constant 2021 $),
    # because OWID's own co2_per_gdp is missing for many small states (Bahamas, Grenada, Maldives, Somalia ...).
    owid = pd.read_csv(p["raw"] / "owid" / "owid-co2-data.csv", usecols=["iso_code", "year", "co2"]) \
        .rename(columns={"iso_code": "iso3"})
    y = y.merge(owid, on=["iso3", "year"], how="left")
    y["G"] = y.co2 * 1e9 / (d.Y.values * d.Np.values)                              # kg CO2 per $
    y["G_owid"] = d.G_co2_per_gdp.values
    # F: temperature change vs 1951-1980 (the definition behind the IMF series used in March): FAOSTAT, and where
    # FAOSTAT has no value (e.g. Yemen, Singapore, Burundi) the same quantity from CRU TS.
    js = json.loads(next((p["raw"] / "cckp_tas").glob("*.json")).read_text())["data"]
    cru = pd.DataFrame([(i, int(k[:4]), v) for i, ser in js.items() if isinstance(ser, dict)
                        for k, v in ser.items() if v is not None], columns=["iso3", "year", "Tc"])
    base = cru[cru.year.between(1951, 1980)].groupby("iso3").Tc.mean().rename("Tc5180")
    cru = cru.merge(base, on="iso3")
    cru["dT_cru"] = cru.Tc - cru.Tc5180
    y = y.merge(cru[["iso3", "year", "dT_cru"]], on=["iso3", "year"], how="left")
    y["F_source"] = np.where(d.dT.notna().values, "FAOSTAT", np.where(y.dT_cru.notna(), "CRU", None))
    y["F"] = np.where(d.dT.notna().values, d.dT.values, y.dT_cru)
    y["W"], y["O"] = d.Pr.values, d.LECZ.values
    return y, has_in


def windows(y):
    rows = []
    cols = ["M", "M_out", "M_in", "M2", "D", "A", "E", "S", "P", "G", "W", "F", "O", "Np"]
    for w in WINDOWS:
        s = y[y.year.between(*w)]
        g = s.groupby(["iso3", "country", "region"])[cols].mean(numeric_only=True)   # mean of available years
        n = s.groupby(["iso3", "country", "region"])[cols].count()
        g = g.where(n >= 2)                                                          # need 2 of the 3 years
        g = g.reset_index()
        g["window"] = wl(w)
        g["w0"] = w[0]
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def pi_groups(m):
    """The six March formulas. Groups that are not dimensionless in measured units are divided by their median
    across countries in the same window."""
    m = m.copy()
    m["pi1"] = m.M / (m.D * m.A)
    m["pi2_raw"] = m.G / (m.D * m.A * m.E)
    m["pi3_raw"] = m.S * np.sqrt(m.A) / (m.W * m.E)
    m["pi4_raw"] = m.P * m.W * m.E / np.sqrt(m.A)
    m["pi5"] = m.O
    m["pi6_raw"] = m.F.where(m.F > 0) / m.W
    for k in ["pi2", "pi3", "pi4", "pi6"]:
        med = m.groupby("window")[f"{k}_raw"].transform(lambda s: s[s > 0].median())
        m[k] = m[f"{k}_raw"] / med
    m["pi1_idmc"] = m.M2 / (m.D * m.A)
    # logs; groups that can be 0 (pi4: no conflict deaths, pi5: landlocked) get the smallest positive value added
    eps = {}
    for k in PIS + ["pi1_idmc"]:
        pos = m.loc[m[k] > 0, k]
        eps[k] = float(pos.min()) if k in ("pi4", "pi5", "pi1_idmc") else 0.0
        m[f"ln_{k}"] = np.log(m[k] + eps[k]).where(m[k] + eps[k] > 0)
    return m, eps


# ------------------------------------------------------------------ clustering (March section 3.1, SI S6 and S8)
def kmeans_labels(X):
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X)
    return km.labels_, km.cluster_centers_


def clustering(m):
    out = []
    for w in WINDOWS:
        s = m[m.window == wl(w)]
        for k in PIS:
            d = s.dropna(subset=[f"ln_{k}"])
            lab, cen = kmeans_labels(d[[f"ln_{k}"]].values)
            rank = np.argsort(np.argsort(cen[:, 0]))                                  # 0 = lowest, 6 = highest
            out.append(pd.DataFrame({"iso3": d.iso3, "window": wl(w), "feature": k, "cluster": rank[lab]}))
        d = s.dropna(subset=VARN)
        X = d[VARN].values
        X = 10 * (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))               # var-N: 0-10 as in March
        lab, _ = kmeans_labels(X)
        out.append(pd.DataFrame({"iso3": d.iso3, "window": wl(w), "feature": "varN", "cluster": lab}))
    return pd.concat(out, ignore_index=True)


def coherence(cl, m):
    """Geographic coherence of a clustering = adjusted Rand index between clusters and World Bank regions."""
    reg = m.drop_duplicates("iso3").set_index("iso3").region
    rows = []
    for (w, f), g in cl.groupby(["window", "feature"]):
        r = reg.reindex(g.iso3).values
        ok = pd.notna(r)
        rows.append({"window": w, "feature": f, "ari_region": adjusted_rand_score(r[ok], g.cluster.values[ok]),
                     "n": int(ok.sum())})
    return pd.DataFrame(rows)


def world():
    import geopandas as gpd
    try:
        w = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        w["iso3"] = w.iso_a3
    except Exception:                                                                # geopandas >= 1.0 (Colab)
        w = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
        w["iso3"] = np.where(w.ISO_A3 == "-99", w.ADM0_A3, w.ISO_A3)
    return w[w.iso3 != "ATA"]


def map_figure(cl, feature, out, name, title):
    wd = world()
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.3))
    for a, w in zip(axes.ravel(), SHOW):
        g = cl[(cl.feature == feature) & (cl.window == wl(w))]
        mm = wd.merge(g, on="iso3", how="left")
        mm[mm.cluster.isna()].plot(ax=a, color="#ecebe7", edgecolor="white", linewidth=0.2)
        pal = CAT if feature == "varN" else SEQ
        for c in range(K):
            sub = mm[mm.cluster == c]
            if len(sub):
                sub.plot(ax=a, color=pal[c], edgecolor="white", linewidth=0.2)
        a.set_axis_off()
        a.set_title(wl(w), fontsize=8, loc="left")
    if feature == "varN":
        handles = [plt.Rectangle((0, 0), 1, 1, color=CAT[c]) for c in range(K)]
        labels = [f"cluster {c + 1}" for c in range(K)]
    else:
        handles = [plt.Rectangle((0, 0), 1, 1, color=SEQ[c]) for c in range(K)]
        labels = ["lowest"] + [""] * (K - 2) + ["highest"]
    fig.legend(handles, labels, loc="lower center", ncol=K, fontsize=7, handlelength=1.2, columnspacing=0.8)
    fig.suptitle(title, fontsize=9, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    _save(fig, out, name)


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ groups: statistics, correlations, scatter
def group_rows(m, iso):
    return m[m.iso3.isin(iso)]


def describe(m, giso):
    rows = []
    for (case, name), iso in giso.items():
        s = group_rows(m, iso)
        for v in VARN:
            x = s[v].dropna()
            rows.append({"case": case, "group": name, "variable": v, "mean": x.mean(), "median": x.median(),
                         "sd": x.std(), "min": x.min(), "max": x.max(), "n_countries": s.iso3.nunique()})
    return pd.DataFrame(rows)


def correlations(m, giso):
    rows = []
    for (case, name), iso in giso.items():
        for w in SHOW:
            s = group_rows(m, iso)
            s = s[s.window == wl(w)]
            for i, a in enumerate(PIS):
                for b in PIS[i + 1:]:
                    d = s[[f"ln_{a}", f"ln_{b}"]].dropna()
                    if len(d) >= 5 and d.std().min() > 0:
                        r, p = stats.pearsonr(d.iloc[:, 0], d.iloc[:, 1])
                    else:
                        r, p = np.nan, np.nan
                    rows.append({"case": case, "group": name, "window": wl(w), "a": a, "b": b, "r": r, "p": p,
                                 "n": len(d)})
    return pd.DataFrame(rows)


def heatmap_figure(corr, case, name, out, fname):
    fig, axes = plt.subplots(1, len(SHOW), figsize=(7.4, 2.2))
    for a, w in zip(axes, SHOW):
        c = corr[(corr.case == case) & (corr.group == name) & (corr.window == wl(w))]
        M = np.eye(6)
        P = np.zeros((6, 6))
        for _, r in c.iterrows():
            i, j = PIS.index(r.a), PIS.index(r.b)
            M[i, j] = M[j, i] = r.r
            P[i, j] = P[j, i] = r.p
        im = a.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(6):
            for j in range(6):
                if i != j and np.isfinite(M[i, j]):
                    star = "*" if P[i, j] < 0.05 else ""
                    a.text(j, i, f"{M[i, j]:.2f}{star}", ha="center", va="center", fontsize=4.6,
                           color="white" if abs(M[i, j]) > 0.6 else INK)
        a.set_xticks(range(6)); a.set_yticks(range(6))
        a.set_xticklabels([f"π{k}" for k in range(1, 7)], fontsize=6)
        a.set_yticklabels([f"π{k}" for k in range(1, 7)], fontsize=6)
        a.set_title(wl(w), fontsize=7, loc="left")
        for s in a.spines.values():
            s.set_visible(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Pearson r of ln values")
    fig.suptitle(f"{name}: correlations of ln π (* p < 0.05)", fontsize=8, x=0.02, ha="left")
    _save(fig, out, fname)


def scatter_figure(m, iso, name, out, fname, outcome="pi1"):
    s = group_rows(m, iso)
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.8), sharey=True)
    cmap = plt.get_cmap("viridis")
    for a, k in zip(axes, PIS[1:]):
        for i, w in enumerate(WINDOWS):
            d = s[s.window == wl(w)]
            a.scatter(d[f"ln_{k}"], d[f"ln_{outcome}"], s=4, color=cmap(i / (len(WINDOWS) - 1)), linewidth=0)
        a.set_xlabel(f"ln π{k[-1]}", fontsize=7)
    axes[0].set_ylabel("ln π1", fontsize=7)
    sm_ = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(2008, 2021))
    fig.colorbar(sm_, ax=axes, shrink=0.8, label="window start year")
    fig.suptitle(f"{name}: ln π1 against the other groups, 14 windows", fontsize=8, x=0.02, ha="left")
    _save(fig, out, fname)


# ------------------------------------------------------------------ power laws (March eq. 2, Table 1, S13, S14)
X_COLS = [f"ln_{k}" for k in PIS[1:]]


def fit_group(s, outcome="pi1"):
    """March method: least squares of ln pi1 on ln pi2..pi6. Count model: PPML of M with offset ln(D A)."""
    y_col, count = (f"ln_{outcome}", "M") if outcome == "pi1" else ("ln_pi1_idmc", "M2")
    d = s.dropna(subset=X_COLS + [y_col, count])
    d = d[d[count] > 0] if outcome == "pi1" else d
    if len(d) < len(X_COLS) + 3:
        return None
    X = sm.add_constant(d[X_COLS], has_constant="add")
    ols = sm.OLS(d[y_col], X).fit()
    try:
        ppml = sm.GLM(d[count], X, family=sm.families.Poisson(), offset=np.log(d.D * d.A)).fit(cov_type="HC1")
        pr2 = 1 - ppml.deviance / ppml.null_deviance
    except Exception:
        ppml, pr2 = None, np.nan
    return {"n": len(d), "ols": ols, "ppml": ppml, "r2": ols.rsquared, "pseudo_r2": pr2, "data": d}


def perm_r2(d, count, eps=0.0, n_perm=1000):
    """Null for a group fit: the count (M) is shuffled across the countries of the group while each country keeps
    its own D and A, so pi1 = M/(D A) keeps the terms it shares with pi2, pi3 and pi4. A fit that does not beat
    this null is explained by the shared terms alone (Pearson 1897)."""
    rng = np.random.default_rng(SEED)
    X = sm.add_constant(d[X_COLS], has_constant="add").values
    da = (d.D * d.A).values
    yfun = lambda mm: np.log(mm / da + eps)
    obs = sm.OLS(yfun(d[count].values), X).fit().rsquared
    null = np.array([sm.OLS(yfun(rng.permutation(d[count].values)), X).fit().rsquared for _ in range(n_perm)])
    return obs, float(np.percentile(null, 95)), float((1 + (null >= obs).sum()) / (1 + n_perm))


def power_laws(m, giso, outcome="pi1", eps_idmc=0.0):
    rows = []
    y_col = "ln_pi1" if outcome == "pi1" else "ln_pi1_idmc"
    for (case, name), iso in giso.items():
        for w in SHOW:
            s = group_rows(m, iso)
            f = fit_group(s[s.window == wl(w)], outcome)
            if f is None:
                continue
            obs, n95, pp = perm_r2(f["data"], "M" if outcome == "pi1" else "M2", 0.0 if outcome == "pi1" else eps_idmc)
            row = {"case": case, "group": name, "window": wl(w), "n": f["n"], "r2_ols": f["r2"],
                   "r2_null95": n95, "p_perm": pp, "pseudo_r2_ppml": f["pseudo_r2"]}
            for k in X_COLS:
                row[f"ols_{k}"] = f["ols"].params[k]
                if f["ppml"] is not None:
                    row[f"ppml_{k}"] = f["ppml"].params[k]
                    row[f"ppml_se_{k}"] = f["ppml"].bse[k]
            row["ols_lnC"] = f["ols"].params["const"]
            rows.append(row)
    return pd.DataFrame(rows)


def collapse_test(m, giso):
    """Self-similarity as data collapse: are the exponents the same in every window? F-test of one pooled
    equation against separate equations per window (OLS, as in March)."""
    rows = []
    for (case, name), iso in giso.items():
        s = group_rows(m, iso).dropna(subset=X_COLS + ["ln_pi1"])
        s = s[s.w0.isin([w[0] for w in SHOW])]
        if s.window.nunique() < 2 or len(s) < 30:
            continue
        pooled = sm.OLS(s.ln_pi1, sm.add_constant(s[X_COLS], has_constant="add")).fit()
        rss_sep, df_sep = 0.0, 0
        for _, g in s.groupby("window"):
            f = sm.OLS(g.ln_pi1, sm.add_constant(g[X_COLS], has_constant="add")).fit()
            rss_sep += f.ssr
            df_sep += f.df_resid
        q = pooled.df_resid - df_sep
        F = ((pooled.ssr - rss_sep) / q) / (rss_sep / df_sep) if q > 0 and df_sep > 0 else np.nan
        p = 1 - stats.f.cdf(F, q, df_sep) if np.isfinite(F) else np.nan
        rows.append({"case": case, "group": name, "F": F, "df1": q, "df2": df_sep, "p": p,
                     "collapse_not_rejected": bool(p >= 0.05) if np.isfinite(p) else None})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ forecast (March: fit 2014-16, predict 2018-20)
def forecasts(m, giso, outcome="pi1"):
    rows, pts = [], []
    y_col = "ln_pi1" if outcome == "pi1" else "ln_pi1_idmc"
    for (case, name), iso in giso.items():
        s = group_rows(m, iso)
        f = fit_group(s[s.window == wl(FIT_W)], outcome)
        if f is None:
            continue
        te = s[s.window == wl(PRED_W)].dropna(subset=X_COLS + [y_col])
        tr = s[s.window == wl(FIT_W)].set_index("iso3")[y_col]
        te = te[te.iso3.isin(tr.index)]
        pred = f["ols"].predict(sm.add_constant(te[X_COLS], has_constant="add"))
        obs = np.exp(te[y_col]); pr = np.exp(pred); pers = np.exp(te.iso3.map(tr))
        ok = np.isfinite(pers.values) & np.isfinite(pr.values)
        te, obs, pr, pers = te[ok], obs[ok], pr[ok], pers[ok]
        err_m = np.abs(pr - obs) / obs
        err_p = np.abs(pers - obs) / obs
        rows.append({"case": case, "group": name, "n": len(te), "median_err_model": float(np.median(err_m)),
                     "median_err_persist": float(np.median(err_p)), "mean_err_model": float(np.mean(err_m)),
                     "mean_err_persist": float(np.mean(err_p)),
                     "share_model_better": float((err_m < err_p).mean())})
        pts.append(pd.DataFrame({"case": case, "group": name, "iso3": te.iso3, "obs": obs, "pred": pr,
                                 "persist": pers.values}))
    return pd.DataFrame(rows), pd.concat(pts) if pts else pd.DataFrame()


def forecast_figure(pts, case, out, fname):
    groups = pts[pts.case == case].group.unique()
    fig, axes = plt.subplots(1, len(groups), figsize=(2.5 * len(groups), 2.4))
    axes = np.atleast_1d(axes)
    for a, g in zip(axes, groups):
        d = pts[(pts.case == case) & (pts.group == g)]
        lo = min(d.obs.min(), d.pred.min(), d.persist.min()) * 0.5
        hi = max(d.obs.max(), d.pred.max(), d.persist.max()) * 2
        a.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--")
        a.scatter(d.obs, d.persist, s=10, color=BLUE, label="persistence", linewidth=0)
        a.scatter(d.obs, d.pred, s=10, color=ORANGE, label="power law", linewidth=0)
        a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel("observed π1, 2018-2020", fontsize=7); a.set_title(g, fontsize=8, loc="left")
    axes[0].set_ylabel("predicted π1", fontsize=7)
    axes[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    _save(fig, out, fname)


# ------------------------------------------------------------------ pooled test (SI S12)
def pooled_permutation(m, n_perm=1000):
    """All countries, the four SHOW windows: PPML of M on ln pi2..pi6 with offset ln(D A) and window effects.
    Statistic: share of the remaining deviance explained by adding the five groups. Null: each country's whole series of pi1 rates is
    given to another country, which keeps its own D and A (so shared terms cannot create the signal)."""
    s = m[m.w0.isin([w[0] for w in SHOW])].dropna(subset=X_COLS + ["M", "D", "A"])
    s = s[(s.D * s.A) > 0]
    def stat(d):
        W = pd.get_dummies(d.window, drop_first=True, dtype=float)
        base = sm.add_constant(W, has_constant="add")
        full = sm.add_constant(pd.concat([d[X_COLS], W], axis=1), has_constant="add")
        off = np.log(d.D * d.A).values
        f0 = sm.GLM(d.M.values, base, family=sm.families.Poisson(), offset=off).fit(maxiter=200)
        f1 = sm.GLM(d.M.values, full, family=sm.families.Poisson(), offset=off).fit(maxiter=200)
        return (f0.deviance - f1.deviance) / f0.deviance        # share of deviance the five groups explain
    obs = stat(s)
    s = s.assign(rate=s.M / (s.D * s.A))
    wide = s.pivot(index="iso3", columns="window", values="rate")
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n_perm):
        perm = rng.permutation(len(wide))
        sh = pd.DataFrame(wide.values[perm], index=wide.index, columns=wide.columns).stack().rename("rate_p").reset_index()
        d = s.merge(sh, on=["iso3", "window"]).dropna(subset=["rate_p"])
        d["M"] = d.rate_p * d.D * d.A                       # each country keeps its own D and A
        try:
            null.append(stat(d))
        except Exception:
            pass
    null = np.array(null)
    return {"obs": float(obs), "null_median": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "p": float((1 + (null >= obs).sum()) / (1 + len(null))), "n_perm": int(len(null)),
            "countries": int(s.iso3.nunique()), "rows": int(len(s))}


# ------------------------------------------------------------------ named examples (March discussion and SI S8)
EXAMPLES = {"pi1": ["CHN", "SYR", "LBN", "TUR", "JOR", "DEU", "CAN", "AFG", "VEN", "UKR"],
            "pi5": ["BGD", "VNM", "MDV", "NLD", "EGY", "GUY", "SUR", "CHN"],
            "pi6": ["RUS", "CAN", "NOR", "FIN", "SWE", "ISL", "SAU", "EGY"],
            "pi4": ["SYR", "AFG", "YEM", "IRQ", "SOM", "NGA", "UKR", "MLI"],
            "pi2": ["CHN", "IND", "RUS", "SAU", "POL", "ZAF"]}


def examples(cl):
    rows = []
    for k, isos in EXAMPLES.items():
        for iso in isos:
            for w in SHOW:
                c = cl[(cl.feature == k) & (cl.window == wl(w)) & (cl.iso3 == iso)].cluster
                rows.append({"feature": k, "iso3": iso, "window": wl(w),
                             "cluster_rank_1to7": int(c.iloc[0]) + 1 if len(c) else None})
    return pd.DataFrame(rows).pivot_table(index=["feature", "iso3"], columns="window", values="cluster_rank_1to7")


# ------------------------------------------------------------------ run
def run(project_root):
    root = Path(project_root)
    res = root / "05_Results" / "paper0"
    fig = root / "06_Manuscript_Paper0" / "figures"
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    y, has_in = yearly(root)
    giso = group_iso()
    m, eps = pi_groups(windows(y))
    m.to_csv(res / "paper0_windows_pi.csv", index=False)
    nums = {"has_inflow": has_in, "eps": eps, "countries_with_pi1": int(m.dropna(subset=["pi1"]).iso3.nunique()),
            "windows": len(WINDOWS)}
    # descriptive statistics (Table S2 equivalent, raw units)
    m[VARN + [k for k in PIS]].describe().T.to_csv(res / "tableS2_descriptive.csv")
    describe(m, giso).to_csv(res / "tablesS4_S12_group_statistics.csv", index=False)
    nums["group_sizes"] = {f"{c}:{g}": int(m[m.iso3.isin(i)].iso3.nunique()) for (c, g), i in giso.items()}
    # clustering + maps
    cl = clustering(m)
    cl.to_csv(res / "clusters.csv", index=False)
    coh = coherence(cl, m)
    coh.to_csv(res / "cluster_coherence.csv", index=False)
    nums["coherence_mean_by_feature"] = coh.groupby("feature").ari_region.mean().round(3).to_dict()
    for feat, title in [("varN", "var-N clustering (10 raw variables, 0-10 scaled, k = 7)")] + \
                       [(k, f"π{k[-1]} ({LABEL[k]}) clustering, k = 7, darker = higher") for k in PIS]:
        map_figure(cl, feat, fig, f"map_{feat}", title)
    examples(cl).to_csv(res / "named_examples_cluster_ranks.csv")
    ex_iso = sorted({i for v in EXAMPLES.values() for i in v})
    m[m.iso3.isin(ex_iso) & m.window.isin([wl(w) for w in SHOW])][["iso3", "window", "M", "M_out", "M_in"] + PIS] \
        .to_csv(res / "named_examples_pi_values.csv", index=False)
    # correlations, scatter
    corr = correlations(m, giso)
    corr.to_csv(res / "correlations.csv", index=False)
    # March "average Delta": mean absolute change of each correlation between consecutive shown windows
    cw = corr.pivot_table(index=["case", "group", "a", "b"], columns="window", values="r")
    cols3 = [wl(w) for w in SHOW[:3]]
    delta = cw[cols3].diff(axis=1).abs().iloc[:, 1:].mean(axis=1).groupby(level=["case", "group"]).mean()
    signflip = (np.sign(cw[cols3]).nunique(axis=1) > 1).groupby(level=["case", "group"]).sum()
    stab = pd.DataFrame({"avg_delta_2008_2020": delta, "pairs_with_sign_change_2008_2020": signflip}).reset_index()
    stab.to_csv(res / "correlation_stability.csv", index=False)
    nums["correlation_stability"] = stab.round(3).to_dict(orient="records")
    for (c, g), iso in giso.items():
        tag = g.replace(" ", "_")
        heatmap_figure(corr, c, g, fig, f"heat_{c}_{tag}")
        scatter_figure(m, iso, g, fig, f"scatter_{c}_{tag}")
    # power laws, permutation, collapse
    pl = power_laws(m, giso)
    pl.to_csv(res / "table1_S13_S14_power_laws.csv", index=False)
    pl2 = power_laws(m, giso, outcome="idmc", eps_idmc=eps["pi1_idmc"])
    pl2.to_csv(res / "power_laws_idmc.csv", index=False)
    col = collapse_test(m, giso)
    col.to_csv(res / "self_similarity_collapse_test.csv", index=False)
    nums["power_law"] = {"r2_ols_range": [float(pl.r2_ols.min()), float(pl.r2_ols.max())],
                         "r2_ols_median": float(pl.r2_ols.median()),
                         "share_fits_beating_null95": float((pl.r2_ols > pl.r2_null95).mean()),
                         "n_fits": int(len(pl))}
    nums["collapse_not_rejected_groups"] = col[col.collapse_not_rejected == True].group.tolist()
    # forecasts
    fc, pts = forecasts(m, giso)
    fc.to_csv(res / "forecast_2014_16_to_2018_20.csv", index=False)
    pts.to_csv(res / "forecast_points.csv", index=False)
    for c in ("A", "B"):
        forecast_figure(pts, c, fig, f"forecast_case{c}")
    fc2, _ = forecasts(m, giso, outcome="idmc")
    fc2.to_csv(res / "forecast_idmc.csv", index=False)
    nums["forecast"] = fc[["group", "median_err_model", "median_err_persist", "share_model_better"]] \
        .round(3).to_dict(orient="records")
    # pooled permutation test
    nums["pooled_permutation"] = pooled_permutation(m)
    (res / "paper0_numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps(nums, indent=2, default=str))
    return nums
