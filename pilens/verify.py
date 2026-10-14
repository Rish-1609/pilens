"""Step 1 check: open every raw file and confirm it is the right dataset before anything is built.

For each source it checks the identity (indicator code, version, expected columns), coverage
(years, number of countries) and basic sanity. Writes 02_Raw_Data/SOURCE_CHECK.md.
A source passes only if every check passes. Step 2 must not run while any source fails.
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

from . import build, config

Y0, Y1 = config.YEARS


def _row(source, check, ok, detail):
    return {"source": source, "check": check, "result": "PASS" if ok else "FAIL", "detail": detail}


def check_wdi(raw):
    rows = []
    try:
        cty = build.countries(raw)
        rows.append(_row("WDI", "economy list, aggregates removed", 190 <= len(cty) <= 230, f"{len(cty)} economies"))
        iso = set(cty.iso3)
    except Exception as e:
        return [_row("WDI", "economy list", False, str(e))]
    for sym, (code, unit) in config.WDI.items():
        try:
            jf = raw / "wdi" / f"{code}.json"
            if jf.exists():
                js = json.loads(jf.read_text())
                ids = {(r.get("indicator") or {}).get("id") for r in js[1]}
                rows.append(_row(f"WDI {code}", "indicator code matches", ids == {code}, f"found {sorted(ids)}"))
            s = build.wdi_series(raw, sym, code)
            s = s[s.iso3.isin(iso)]
            if sym == "LECZ":
                n = s[(s.year == 2015) & s[sym].notna()].iso3.nunique()
                rows.append(_row(f"WDI {code}", "2015 values for at least 150 economies", n >= 150, f"{n} economies"))
            else:
                per_year = s[s.year.between(Y0, Y1) & s[sym].notna()].groupby("year").iso3.nunique()
                worst = int(per_year.min()) if len(per_year) else 0
                need = 150 if sym in ("Np", "A_L", "D_wdi") else 120
                rows.append(_row(f"WDI {code}", f"at least {need} economies in every year {Y0}-{Y1}",
                                 worst >= need and len(per_year) == Y1 - Y0 + 1,
                                 f"min {worst} economies; years with data {len(per_year)}"))
        except Exception as e:
            rows.append(_row(f"WDI {code}", "readable", False, str(e)))
    return rows


def check_ucdp(raw):
    try:
        z = next((raw / "ucdp").glob("*.zip"))
        with zipfile.ZipFile(z) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            df = pd.read_csv(zf.open(name))
        c = config.UCDP_COLUMNS
        rows = [_row("UCDP", "codebook columns present", all(v in df.columns for v in c.values()),
                     ", ".join(v for v in c.values() if v not in df.columns) or "all present")]
        ver = sorted(df[c["version"]].astype(str).unique()) if c["version"] in df else []
        rows.append(_row("UCDP", "version 26.1", any(v.startswith("26.1") for v in ver), f"version {ver}"))
        yrs = df[c["year"]]
        rows.append(_row("UCDP", f"covers {Y0}-{Y1}", yrs.min() <= Y0 and yrs.max() >= Y1, f"{yrs.min()}-{yrs.max()}"))
        n = df[df[c["year"]] == 2015][c["gw"]].nunique()
        rows.append(_row("UCDP", "includes zero-violence country-years (>=150 countries in 2015)", n >= 150, f"{n} countries in 2015"))
        return rows
    except Exception as e:
        return [_row("UCDP", "readable", False, str(e))]


def check_fao(raw):
    try:
        t = build.fao_temperature(raw)
        t = t.dropna(subset=["iso3"])
        cover = t[t.year.between(Y0, Y1) & t.dT.notna()].groupby("year").iso3.nunique()
        return [
            _row("FAOSTAT ET", "temperature change present with 1991-2020 baseline", t.dT.notna().any() and (t.dT_nbase >= 25).mean() > 0.9, ""),
            _row("FAOSTAT ET", f"at least 150 countries every year {Y0}-{Y1}",
                 len(cover) == Y1 - Y0 + 1 and cover.min() >= 150, f"min {int(cover.min()) if len(cover) else 0}"),
            _row("FAOSTAT ET", "values in plausible range (-6 to 6 C)", t.dT.dropna().between(-6, 6).all(),
                 f"range {t.dT.min():.2f} to {t.dT.max():.2f}"),
        ]
    except Exception as e:
        return [_row("FAOSTAT ET", "readable", False, str(e))]


def check_cckp(raw):
    try:
        p = build.cckp_precip(raw)
        b0, b1 = config.PRECIP_BASELINE
        n = p.iso3.nunique()
        return [
            _row("CCKP precipitation", "at least 150 countries", n >= 150, f"{n}"),
            _row("CCKP precipitation", f"covers baseline {b0}-{b1} and {Y0}-{Y1}",
                 p.year.min() <= b0 and p.year.max() >= Y1, f"{p.year.min()}-{p.year.max()}"),
            _row("CCKP precipitation", "annual totals in plausible range (0-6000 mm)", p.Pr.dropna().between(0, 6000).all(),
                 f"median {p.Pr.median():.0f} mm"),
        ]
    except Exception as e:
        return [_row("CCKP precipitation", "readable", False, str(e))]


def check_cckp_temp(raw):
    try:
        t = build.cckp_temperature(raw)
        if t is None:
            return [_row("CCKP temperature", "file present", False,
                         "not downloaded; Pi3 would fall back to FAOSTAT and lose countries with short baselines")]
        b0, b1 = config.CLIMATE_BASELINE
        return [
            _row("CCKP temperature", "at least 150 countries", t.iso3.nunique() >= 150, f"{t.iso3.nunique()}"),
            _row("CCKP temperature", f"covers baseline {b0}-{b1} and {Y0}-{Y1}",
                 t.year.min() <= b0 and t.year.max() >= Y1, f"{t.year.min()}-{t.year.max()}"),
            _row("CCKP temperature", "annual means in plausible range (-30 to 35 C)", t.Tc.dropna().between(-30, 35).all(),
                 f"range {t.Tc.min():.1f} to {t.Tc.max():.1f}"),
        ]
    except Exception as e:
        return [_row("CCKP temperature", "readable", False, str(e))]


def check_idmc(raw):
    try:
        m, src = build.idmc(raw)
        cover = m[m.year.between(Y0, Y1)].groupby("year").iso3.nunique()
        return [
            _row("IDMC", "official IDMC file (not the OWID fallback)", "manual" in src, src),
            _row("IDMC", f"every year {Y0}-{Y1} present", len(cover) == Y1 - Y0 + 1, f"{len(cover)} years"),
            _row("IDMC", "at least 100 countries in file", m.iso3.nunique() >= 100, f"{m.iso3.nunique()} countries"),
            _row("IDMC", "no negative values", (m.M.dropna() >= 0).all(), ""),
        ]
    except Exception as e:
        return [_row("IDMC", "readable", False, str(e))]


def check_unhcr(raw):
    try:
        r = build.unhcr(raw)
        yrs = sorted(r.year.unique())
        return [_row("UNHCR", f"years {Y0}-{Y1}", yrs[0] <= Y0 and yrs[-1] >= Y1, f"{yrs[0]}-{yrs[-1]}"),
                _row("UNHCR", "at least 150 origins", r.iso3.nunique() >= 150, f"{r.iso3.nunique()}")]
    except Exception as e:
        return [_row("UNHCR (secondary)", "readable", False, str(e))]


def run(project_root):
    p = config.paths(project_root)
    raw = p["raw"]
    rows = []
    for fn in (check_wdi, check_ucdp, check_fao, check_cckp, check_cckp_temp, check_idmc, check_unhcr):
        rows += fn(raw)
    df = pd.DataFrame(rows)
    core_fail = df[(df.result == "FAIL") & ~df.source.str.startswith("UNHCR")]
    status = "ALL CORE SOURCES PASS: Step 2 may run" if core_fail.empty else f"{len(core_fail)} CORE CHECKS FAIL: fix before Step 2"
    text = ["# Source check", "", f"**{status}**", "", df.to_markdown(index=False), "",
            "UNHCR is the secondary outcome; its failure does not block Step 2."]
    (raw / "SOURCE_CHECK.md").write_text("\n".join(text))
    print(status)
    return df
