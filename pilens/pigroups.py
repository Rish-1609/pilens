"""Step 4: derive the dimensionless groups from the dimension matrix, verify them, and compute them.

The groups are not typed in by hand and trusted. The null space of the dimension matrix gives the
number of independent groups (Buckingham: n - rank). Each planned group is then checked to have
zero exponent in every base dimension, and the planned set is checked to span the null space.
"""
import numpy as np
import pandas as pd
import sympy as sp

from . import config

BASE = ["N", "C", "L", "T", "Theta"]
# exponents of (N, C, L, T, Theta) for every dimensional variable in the plan
DIMS = {
    "M":       (1, 0, 0, -1, 0),   # persons per year
    "Np":      (1, 0, 0, 0, 0),
    "tau":     (0, 0, 0, 1, 0),    # 1-year reporting period
    "H":       (-1, 1, 0, -1, 0),  # $ per person per year
    "Y":       (-1, 1, 0, -1, 0),
    "Y_ref":   (-1, 1, 0, -1, 0),
    "K":       (1, 0, 0, -1, 0),   # deaths per year
    "A_L":     (0, 0, 2, 0, 0),
    "A_R":     (0, 0, 2, 0, 0),
    "D_ref":   (1, 0, -2, 0, 0),   # persons per km2
    "T_anom":  (0, 0, 0, 0, 1),    # T - mean(T, 1991-2020); T = CRU TS mean temperature (FAOSTAT dT if absent)
    "T_sd":    (0, 0, 0, 0, 1),    # SD of T, 1991-2020
    "Pr_anom": (0, 0, 1, -1, 0),   # Pr - Pr_mean
    "Pr_sd":   (0, 0, 1, -1, 0),
}
# planned groups as exponent maps (ANALYSIS_PLAN section 5); LECZ is already a share
PLANNED = {
    "pi0_displacement_rate": {"M": 1, "tau": 1, "Np": -1},
    "pi1_health_share":      {"H": 1, "Y": -1},
    "pi2_conflict_rate":     {"K": 1, "tau": 1, "Np": -1},
    "pi3_temp_anomaly":      {"T_anom": 1, "T_sd": -1},
    "pi4_precip_anomaly":    {"Pr_anom": 1, "Pr_sd": -1},
    "pi6_relative_income":   {"Y": 1, "Y_ref": -1},
    "pi7_arable_share":      {"A_R": 1, "A_L": -1},
    "pi8_relative_density":  {"Np": 1, "A_L": -1, "D_ref": -1},
}
# admissible but deliberately excluded (ANALYSIS_PLAN section 5): no physical mechanism
EXCLUDED = {
    "rainfall_depth_vs_country_size": {"Pr_sd": 1, "tau": 1, "A_L": -0.5},
}


def derive_and_verify():
    names = list(DIMS)
    D = sp.Matrix([[DIMS[v][i] for v in names] for i in range(len(BASE))])
    rank = D.rank()
    n_groups = len(names) - rank
    lines = [f"Variables: {len(names)}; rank of dimension matrix: {rank}; independent groups (n - rank): {n_groups}"]
    vecs = []
    for g, ex in PLANNED.items():
        v = sp.Matrix([ex.get(n, 0) for n in names])
        res = D * v
        ok = all(x == 0 for x in res)
        lines.append(f"  {g}: residual exponents {list(res)} -> {'dimensionless' if ok else 'NOT DIMENSIONLESS'}")
        if not ok:
            raise AssertionError(f"{g} is not dimensionless")
        vecs.append(v)
    span_rank = sp.Matrix.hstack(*vecs).rank()
    lines.append(f"Planned groups are independent: rank {span_rank} of {len(vecs)}")
    for g, ex in EXCLUDED.items():
        v = sp.Matrix([sp.Rational(ex.get(n, 0)).limit_denominator() for n in names])
        assert all(x == 0 for x in D * v), f"excluded group {g} is not dimensionless"
        vecs.append(v)
        lines.append(f"  excluded by design: {g}")
    full = sp.Matrix.hstack(*vecs).rank()
    lines.append(f"Planned + excluded groups span the full null space ({full} of {n_groups}): {full == n_groups}")
    if full != n_groups:
        lines.append("  NOTE: some admissible groups are not in the plan; listing a basis of the null space:")
        for b in D.nullspace():
            lines.append("   " + " * ".join(f"{n}^{b[i]}" for i, n in enumerate(names) if b[i] != 0))
    return lines


def compute(project_root):
    p = config.paths(project_root)
    df = pd.read_csv(p["processed"] / "panel_raw_units.csv")
    tau = 1.0
    y_ref = df.groupby("year")["Y"].transform("median")
    # Pi8 uses WDI's published density: WDI population excludes breakaway regions (GEO, MDA) while WDI land
    # area includes them, so Np/A_L understates density there by 15-20% (D-015). Same dimension, N L^-2.
    dens = df["D_wdi"].where(df["D_wdi"].notna(), df["Np"] / df["A_L"])
    d_ref = dens.groupby(df["year"]).transform("median")
    out = df[["iso3", "country", "region", "income", "year", "Np", "M", "K"]].copy()
    out["pi0_displacement_rate"] = df.M * tau / df.Np
    out["pi1_health_share"] = df.HY          # WDI publishes H/Y directly (same price base): no currency mismatch
    out["pi2_conflict_rate"] = df.K * tau / df.Np
    if "Tc" in df:                           # primary: CRU TS via CCKP, same source and baseline as Pi4 (D-013)
        pi3 = (df.Tc - df.Tc_mean) / df.Tc_sd
        pi3 = pi3.where(~df.Tc_climatology_fill.fillna(False).astype(bool))
        out["pi3_source"] = "CRU TS (CCKP)"
    else:                                    # fallback: FAOSTAT, only where the baseline has >= 25 of 30 years
        pi3 = ((df.dT - df.dT_mean) / df.dT_sd).where(df.dT_nbase >= 25)
        out["pi3_source"] = "FAOSTAT"
    out["pi3_temp_anomaly"] = pi3
    out["pi3_faostat_check"] = ((df.dT - df.dT_mean) / df.dT_sd).where(df.dT_nbase >= 25)
    out["pi4_precip_anomaly"] = ((df.Pr - df.Pr_mean) / df.Pr_sd).where(~df.Pr_climatology_fill.fillna(False).astype(bool))
    out["pi5_lecz_share"] = df.LECZ
    out["pi6_relative_income"] = df.Y / y_ref
    out["pi7_arable_share"] = df.A_R / df.A_L
    out["pi8_relative_density"] = dens / d_ref
    lines = derive_and_verify()
    desc = out.filter(regex=r"^pi\d_").select_dtypes("number").describe().T.round(4)
    path = p["processed"] / "panel_pi_groups.csv"
    out.to_csv(path, index=False)
    report = ["# Step 4: dimensionless groups", "", "## Derivation check", ""] + [f"    {l}" for l in lines] + \
             ["", "## Summary of computed groups", "", desc.to_markdown(), "", f"Saved: {path.name}"]
    (p["results"] / "step4_PI_REPORT.md").write_text("\n".join(report))
    print("\n".join(lines))
    return out
