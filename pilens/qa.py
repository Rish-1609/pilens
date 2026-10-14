"""Step 3: automated quality checks on the panel. Nothing is changed here; problems are reported.

Output: 05_Results/step3_QA_REPORT.md and step3_qa_flags.csv.
BLOCKING problems must be fixed (or explained in DECISION_LOG.csv) before Step 4.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CORE = ["M", "Np", "A_L", "A_R", "Y", "HY", "K", "dT", "Pr", "LECZ"]
# plausible physical ranges, in the panel's units
RANGES = {
    "Np": (1e4, 2e9), "A_L": (1, 2e7), "A_R": (0, 2e6), "Y": (200, 2e5), "HY": (0.005, 0.25),
    "dT": (-6, 6), "Pr": (0, 6000), "LECZ": (0, 1), "M": (0, 1e8), "K": (0, 1e6),
}


def run(project_root):
    p = config.paths(project_root)
    df = pd.read_csv(p["processed"] / "panel_raw_units.csv")
    T = "Tc" if "Tc" in df else "dT"          # temperature series used for Pi3
    core = [T if v == "dT" else v for v in CORE]
    flags = []

    def flag(level, check, iso3, year, var, value, detail):
        flags.append(dict(level=level, check=check, iso3=iso3, year=year, variable=var, value=value, detail=detail))

    # 1. structure
    dup = df.duplicated(["iso3", "year"]).sum()
    if dup:
        flag("BLOCKING", "duplicate country-year rows", "", "", "", dup, "each country-year must appear once")
    # 2. sample
    pop15 = df[df.year == 2015].set_index("iso3")["Np"]
    sample = set(pop15[pop15 >= config.MIN_POP_2015].index)
    df["in_sample"] = df.iso3.isin(sample)
    s = df[df.in_sample]
    # 3. coverage
    cov = s.groupby("year")[core].apply(lambda g: g.notna().mean()).round(2)
    for v in core:
        share = s[v].notna().mean()
        if share < 0.8:
            flag("WARNING", "coverage below 80%", "", "", v, round(share, 2), "check source or restrict years")
    # 4. ranges
    for v, (lo, hi) in RANGES.items():
        bad = s[(s[v] < lo) | (s[v] > hi)]
        for _, r in bad.iterrows():
            flag("BLOCKING", "outside physical range", r.iso3, r.year, v, r[v], f"allowed {lo}-{hi}")
    # 5. internal consistency
    for _, r in s[(s.M > 0.5 * s.Np)].iterrows():
        flag("BLOCKING", "displacement above 50% of population", r.iso3, r.year, "M", r.M, f"population {r.Np:.0f}")
    if "D_wdi" in s:                           # density computed from Np/A_L must match WDI's published density
        rel = ((s.Np / s.A_L) - s.D_wdi).abs() / s.D_wdi
        for _, r in s[rel > 0.05].iterrows():
            flag("WARNING", "computed density differs from WDI density by >5%", r.iso3, r.year, "Np/A_L",
                 r.Np / r.A_L, f"WDI {r.D_wdi:.1f}")
    for _, r in s[(s.A_R > s.A_L)].iterrows():
        flag("BLOCKING", "arable land exceeds land area", r.iso3, r.year, "A_R", r.A_R, f"A_L {r.A_L:.0f}")
    # 6. jumps: year-on-year change of more than 10x in slow variables
    for v in ["Np", "A_L", "A_R", "Y", "HY"]:
        ordered = s.sort_values(["iso3", "year"])
        ratio = (ordered[v] / ordered.groupby("iso3")[v].shift(1)).abs()
        idx = ratio[(ratio > 10) | (ratio < 0.1)].index
        for i in idx:
            flag("BLOCKING", "10x jump in a slow-moving variable", s.at[i, "iso3"], s.at[i, "year"], v, s.at[i, v], "")
    # 7. cross-source check: WDI population vs OWID population (catches unit errors like x100)
    owid_f = p["raw"] / "owid" / "owid-co2-data.csv"
    if owid_f.exists():
        o = pd.read_csv(owid_f, usecols=["iso_code", "year", "population"]).rename(columns={"iso_code": "iso3"})
        m = s.merge(o, on=["iso3", "year"], how="inner")
        m["rel"] = (m.Np - m.population).abs() / m.population
        for _, r in m[m.rel > 0.05].iterrows():
            flag("WARNING", "population differs from OWID by >5%", r.iso3, r.year, "Np", r.Np,
                 f"OWID {r.population:.0f} ({100*r.rel:.0f}%)")
    # 8. outcome outliers (reported, not removed)
    rate = (s.M / s.Np)
    top = s.assign(rate=rate).nlargest(15, "rate")
    for _, r in top.iterrows():
        flag("INFO", "largest displacement rates (verify against IDMC event records)", r.iso3, r.year, "M", r.M,
             f"rate {r.rate:.3f}")
    # 9. climate baseline completeness
    if f"{T}_nbase" in s:
        for c in s[s[f"{T}_nbase"] < 25].iso3.unique():
            flag("WARNING", "temperature baseline has fewer than 25 of 30 years", c, "", T, "", "")
    for v in ("Pr", "Tc"):                    # CRU series that are climatology fill: anomaly set missing (D-014)
        if f"{v}_climatology_fill" in s:
            for c in s[s[f"{v}_climatology_fill"] == True].iso3.unique():
                flag("WARNING", "CRU series is climatology fill; anomaly set missing", c, "", v, "", "")
    if T == "Tc" and "dT_nbase" in s:          # cross-source check: CRU vs FAOSTAT temperature anomalies agree
        for c, g in s[s.dT_nbase >= 25].groupby("iso3"):
            g = g.dropna(subset=["Tc", "dT"])
            if len(g) >= 10:
                r = np.corrcoef(g.Tc, g.dT)[0, 1]
                if r < 0.7:
                    flag("WARNING", "CRU and FAOSTAT temperature disagree (r < 0.7)", c, "", "Tc", round(r, 2), "")
    if "Pr_nbase" in s:
        short = s[s.Pr_nbase < 25].iso3.unique()
        for c in short:
            flag("WARNING", "precipitation baseline has fewer than 25 of 30 years", c, "", "Pr", "", "")

    fl = pd.DataFrame(flags)
    fl.to_csv(p["results"] / "step3_qa_flags.csv", index=False)
    n_block = int((fl.level == "BLOCKING").sum()) if len(fl) else 0
    status = "PASS" if n_block == 0 else f"BLOCKED ({n_block} blocking problems)"
    lines = [
        "# Step 3 quality report", "",
        f"**Status: {status}**", "",
        f"- Countries in file: {df.iso3.nunique()}; in sample (population >= {config.MIN_POP_2015:,} in 2015): {len(sample)}",
        f"- Years: {df.year.min()}-{df.year.max()}",
        f"- IDMC source: {df.idmc_source.iloc[0] if 'idmc_source' in df else 'n/a'}",
        "",
        "## Flags by type", "",
        (fl.groupby(["level", "check"]).size().rename("count").reset_index().to_markdown(index=False) if len(fl) else "No flags."),
        "", "## Coverage (share of sample countries with data)", "", cov.to_markdown(), "",
        "## What to do", "",
        "1. Open step3_qa_flags.csv. For every BLOCKING row, check the raw file and the source website.",
        "2. A real error in the source: record it in DECISION_LOG.csv and add a correction rule in pilens/build.py (never edit raw files).",
        "3. A true value (for example a catastrophic flood): record it in DECISION_LOG.csv as verified.",
        "4. Re-run Steps 2 and 3 until the status is PASS or every remaining flag is logged.",
    ]
    (p["results"] / "step3_QA_REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:8]))
    return fl
