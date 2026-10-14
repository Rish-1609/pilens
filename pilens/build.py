"""Step 2: turn the untouched raw files into one country-year panel in raw units.

Every column keeps its published unit, except three explicit conversions (A_R hectares -> km2,
LECZ % -> share, HY % -> share). Column choices that a human must confirm are printed and saved to
04_Processed_Data/BUILD_LOG.txt.
"""
import io
import json
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logging.getLogger("country_converter").setLevel(logging.ERROR)
LOG = []


def note(msg):
    LOG.append(msg)
    print(msg)


def _coco(values, src, override=None):
    """Convert codes to ISO3. Anything that is not a valid 3-letter code comes back as NaN.
    (country_converter with not_found=None returns the INPUT unchanged, so unmatched codes must be blanked here.)"""
    import country_converter as coco
    vals = pd.Series([str(v) for v in values], dtype=object)
    uniq = sorted(vals.unique())                       # convert each distinct code once (much faster)
    conv = coco.convert(uniq, src=src, to="ISO3", not_found=None)
    conv = conv if isinstance(conv, list) else [conv]
    lookup = {u: (override or {}).get(u, c) for u, c in zip(uniq, conv)}
    lookup = {u: (c if isinstance(c, str) and re.fullmatch(r"[A-Z]{3}", c) else np.nan) for u, c in lookup.items()}
    return vals.map(lookup).values


# ------------------------------------------------------------------ WDI
def countries(raw):
    jf = raw / "wdi" / "wdi_countries.json"
    if not jf.exists():                    # manual fallback: World Bank bulk CSV metadata file
        meta = next((raw / "wdi").glob("Metadata_Country_*.csv"))
        m = pd.read_csv(meta)
        m = m[m["Region"].notna()]         # aggregates have no region in this file
        df = pd.DataFrame({"iso3": m["Country Code"], "country": m.get("TableName", m["Country Code"]),
                           "region": m["Region"], "income": m["IncomeGroup"]})
        note(f"WDI economies from manual metadata CSV (aggregates removed): {len(df)}")
        return df
    js = json.loads(jf.read_text())
    rows = [c for c in js[1] if c.get("region", {}).get("value") != "Aggregates"]
    df = pd.DataFrame({"iso3": [c["id"] for c in rows], "country": [c["name"] for c in rows],
                       "region": [c["region"]["value"] for c in rows],
                       "income": [c["incomeLevel"]["value"] for c in rows]})
    note(f"WDI economies (aggregates removed): {len(df)}")
    return df


def wdi_series(raw, sym, code):
    jf = raw / "wdi" / f"{code}.json"
    if not jf.exists():                    # manual fallback: World Bank bulk CSV (API_<code>_DS2_...csv)
        f = next(p for p in (raw / "wdi").glob(f"API_{code}_*.csv") if not p.name.startswith("Metadata"))
        w = pd.read_csv(f, skiprows=4)
        years = [c for c in w.columns if str(c).isdigit()]
        df = w.melt(id_vars=["Country Code"], value_vars=years, var_name="year", value_name=sym)
        df = df.rename(columns={"Country Code": "iso3"})
        df["year"] = df["year"].astype(int)
        df[sym] = pd.to_numeric(df[sym], errors="coerce")
        note(f"WDI {code} read from manual CSV {f.name}")
        return df
    js = json.loads(jf.read_text())
    df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), sym: r["value"]} for r in js[1]
                       if r.get("countryiso3code")])
    df[sym] = pd.to_numeric(df[sym], errors="coerce")
    return df


# ------------------------------------------------------------------ UCDP
def ucdp(raw):
    z = next((raw / "ucdp").glob("*.zip"))
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        df = pd.read_csv(zf.open(name))
    c = config.UCDP_COLUMNS
    missing = [v for v in (c["gw"], c["year"], c["deaths"]) if v not in df.columns]
    if missing:
        raise ValueError(f"UCDP: expected codebook columns {missing} not found. Columns: {list(df.columns)}")
    note(f"UCDP columns (codebook v26.1): country {c['gw']}, year {c['year']}, deaths {c['deaths']} (all violence types, best estimate)")
    out = pd.DataFrame({"gw": df[c["gw"]], "year": df[c["year"]].astype(int),
                        "K": pd.to_numeric(df[c["deaths"]], errors="coerce")})
    out["iso3"] = _coco(out["gw"].astype(int).astype(str), "GWcode", config.GW_TO_ISO3_OVERRIDE)
    note(f"UCDP: Gleditsch-Ward overrides applied {config.GW_TO_ISO3_OVERRIDE}  [CONFIRM]")
    miss = df.loc[out.iso3.isna(), ["country", c["gw"]]].drop_duplicates() if "country" in df else out.loc[out.iso3.isna(), ["gw"]]
    if len(miss):
        note(f"UCDP: codes with no ISO3 match (dropped): {miss.values.tolist()}")
    return out.dropna(subset=["iso3"]).groupby(["iso3", "year"], as_index=False)["K"].sum()


# ------------------------------------------------------------------ FAOSTAT temperature
def fao_temperature(raw):
    z = next((raw / "fao_temperature").glob("*.zip"))
    with zipfile.ZipFile(z) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv") and "all_data" in n.lower())
        df = pd.read_csv(zf.open(name), encoding="latin-1")
    df = df[df["Months"].astype(str).str.contains("Meteorological year", case=False)]
    m49col = next((c for c in df.columns if "M49" in c), None)
    if m49col:
        codes = df[m49col].astype(str).str.replace("'", "").str.lstrip("0")
        df["iso3"] = _coco(codes.replace("", "0"), "UNcode")
        note(f"FAOSTAT country codes mapped from '{m49col}'")
    else:
        df["iso3"] = _coco(df["Area Code"].astype(str), "FAOcode")
        note("FAOSTAT country codes mapped from FAO 'Area Code'")
    el = df["Element"].str.lower()
    t = df[el.str.contains("temperature change")][["iso3", "Year", "Value"]].rename(columns={"Year": "year", "Value": "dT"})
    s = df[el.str.contains("standard deviation")][["iso3", "Year", "Value"]].rename(columns={"Year": "year", "Value": "sigma_T"})
    out = t.merge(s, on=["iso3", "year"], how="outer").dropna(subset=["iso3"])
    note(f"FAOSTAT: elements found {sorted(df['Element'].unique())}; unit {sorted(df['Unit'].astype(str).unique())}")
    out = out.groupby(["iso3", "year"], as_index=False).mean(numeric_only=True)
    b0, b1 = config.CLIMATE_BASELINE
    base = out[out.year.between(b0, b1)].groupby("iso3")["dT"].agg(dT_mean="mean", dT_sd="std", dT_nbase="count").reset_index()
    note(f"Temperature: FAOSTAT change is relative to 1951-1980; anomaly re-based to {b0}-{b1} mean and SD per country")
    return out.merge(base, on="iso3", how="left")


# ------------------------------------------------------------------ CCKP precipitation
def _cckp_series(raw, folder, col):
    """Read a CCKP CRU country series (JSON from the API, or a manual CSV/XLSX export) -> iso3, year, col,
    plus the 1991-2020 baseline mean/SD/count and a flag for climatology-filled series."""
    d = raw / folder
    js_files = list(d.glob("*.json")) if d.exists() else []
    if js_files:
        data = json.loads(js_files[0].read_text())["data"]
        rows = [(iso, int(str(k)[:4]), v) for iso, series in data.items() if isinstance(series, dict)
                for k, v in series.items() if v is not None]
        df = pd.DataFrame(rows, columns=["iso3", "year", col])
    else:
        f = next(p for p in d.iterdir() if p.suffix.lower() in (".csv", ".xlsx"))
        raw_df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
        iso = next(c for c in raw_df.columns if re.search(r"code|iso", str(c), re.I))
        long = raw_df.melt(id_vars=[iso], var_name="period", value_name=col)
        long["year"] = pd.to_numeric(long["period"].astype(str).str[:4], errors="coerce")
        df = long.dropna(subset=["year"]).rename(columns={iso: "iso3"})[["iso3", "year", col]]
        df["year"] = df["year"].astype(int)
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df.iso3.astype(str).str.fullmatch(r"[A-Z]{3}")]
    b0, b1 = config.CLIMATE_BASELINE
    base = df[df.year.between(b0, b1)].groupby("iso3")[col].agg(**{f"{col}_mean": "mean", f"{col}_sd": "std",
                                                                     f"{col}_nbase": "count"}).reset_index()
    y0, y1 = b0, config.YEARS[1]
    def _longest_run(v):                         # longest stretch of identical values in consecutive years
        v = v.dropna().round(2)
        return int(v.groupby((v != v.shift()).cumsum()).size().max()) if len(v) else 0
    win = df[df.year.between(y0, y1)].sort_values(["iso3", "year"])
    run = win.groupby("iso3")[col].agg(_longest_run)
    filled = sorted(run[run >= config.CLIMATOLOGY_FILL_MIN_RUN].index)
    base[f"{col}_climatology_fill"] = base.iso3.isin(filled)
    note(f"CCKP {col}: {df.iso3.nunique()} countries, years {df.year.min()}-{df.year.max()}; baseline {b0}-{b1}; "
         f"climatology-filled series (anomaly set missing, D-014): {filled}  [CONFIRM]")
    return df.merge(base, on="iso3", how="left")


def cckp_precip(raw):
    return _cckp_series(raw, "cckp_precip", "Pr")


def cckp_temperature(raw):
    """CRU TS annual mean temperature (deg C). Primary source for Pi3 (D-013); None if not downloaded."""
    d = raw / "cckp_tas"
    if not d.exists() or not any(p.suffix.lower() in (".json", ".csv", ".xlsx") for p in d.iterdir()):
        note("CCKP temperature not found: Pi3 falls back to FAOSTAT (see D-013)")
        return None
    return _cckp_series(raw, "cckp_tas", "Tc")


# ------------------------------------------------------------------ IDMC
def idmc(raw):
    folder = raw / "idmc_disasters"
    files = [p for p in folder.iterdir() if p.suffix.lower() in (".xlsx", ".xls", ".csv")]
    manual = [p for p in files if not p.name.startswith("OWID_")]
    f = (manual or files)[0]
    source = "IDMC manual download" if manual else "IDMC via OWID (FALLBACK)"
    df = pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)
    cols = list(df.columns)
    iso = next((c for c in cols if re.fullmatch(r"(iso3?|iso_?code|code)", str(c).strip(), re.I)), None)
    year = next((c for c in cols if re.fullmatch(r"year", str(c).strip(), re.I)), None)
    if config.IDMC_DISPLACEMENT_COLUMN in cols:      # set explicitly after inspecting the file (D-010)
        disp = [config.IDMC_DISPLACEMENT_COLUMN]
    else:
        disp = [c for c in cols if re.search(r"disaster", str(c), re.I) and re.search(r"displace", str(c), re.I)
                and not re.search(r"stock|idps?\b|total number", str(c), re.I)]
    if not (iso and year and len(disp) == 1):
        raise ValueError(f"IDMC: cannot identify columns unambiguously (iso={iso}, year={year}, displacement={disp}). "
                         f"Columns are: {cols}. Set them by hand in this function.")
    note(f"IDMC source: {source} [{f.name}]; columns -> iso3: {iso}; year: {year}; M: {disp[0]}  [CONFIRM]")
    out = df[[iso, year, disp[0]]].rename(columns={iso: "iso3", year: "year", disp[0]: "M"})
    out["M"] = pd.to_numeric(out["M"], errors="coerce")
    out = out[out.iso3.astype(str).str.fullmatch(r"[A-Z]{3}")]
    out["year"] = out["year"].astype(int)
    return out.groupby(["iso3", "year"], as_index=False)["M"].sum(min_count=1), source


# ------------------------------------------------------------------ UNHCR
def unhcr(raw):
    items = json.loads((raw / "unhcr" / "unhcr_population_by_origin.json").read_text())
    df = pd.DataFrame(items)
    iso = next(c for c in df.columns if re.fullmatch(r"coo_iso", c))
    for c in ("refugees", "asylum_seekers"):
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    out = df.groupby([iso, "year"], as_index=False)[["refugees", "asylum_seekers"]].sum(min_count=1)
    out["R"] = out[["refugees", "asylum_seekers"]].sum(axis=1, min_count=1)
    return out.rename(columns={iso: "iso3"})[["iso3", "year", "R"]]


# ------------------------------------------------------------------ assemble
def build_panel(project_root):
    LOG.clear()
    p = config.paths(project_root)
    raw = p["raw"]
    y0, y1 = config.YEARS
    base = countries(raw)
    panel = base.merge(pd.DataFrame({"year": range(y0, y1 + 1)}), how="cross")
    for sym, (code, unit) in config.WDI.items():
        s = wdi_series(raw, sym, code)
        if sym == "LECZ":                      # time-invariant: use 2015, the latest published value
            s = s[s.year == 2015][["iso3", sym]]
            panel = panel.merge(s, on="iso3", how="left")
        else:
            panel = panel.merge(s, on=["iso3", "year"], how="left")
        note(f"WDI {code} -> {sym} ({unit})")
    panel["A_R"] = panel["A_R"] / 100.0        # hectares -> km2
    panel["HY"] = panel["HY"] / 100.0          # % of GDP -> share
    panel["LECZ"] = panel["LECZ"] / 100.0      # % -> share
    note("Conversions: A_R hectares/100 -> km2; LECZ %/100 -> share; HY %/100 -> share")
    owid = raw / "owid" / "owid-co2-data.csv"
    if owid.exists():                          # robustness only: the old paper's carbon-intensity variable
        o = pd.read_csv(owid, usecols=["iso_code", "year", "co2_per_gdp"]).rename(columns={"iso_code": "iso3", "co2_per_gdp": "G_co2_per_gdp"})
        panel = panel.merge(o, on=["iso3", "year"], how="left")
        note("OWID co2_per_gdp (kg CO2 per international-$) added as G_co2_per_gdp, robustness only")
    k = ucdp(raw)
    panel = panel.merge(k, on=["iso3", "year"], how="left")
    # UCDP lists every independent state every year, with 0 for no recorded violence. A country that is
    # absent altogether (e.g. PSE, whose violence UCDP records under Israel) is NOT zero: leave it missing.
    in_ucdp = set(k.iso3)
    panel.loc[panel.iso3.isin(in_ucdp), "K"] = panel.loc[panel.iso3.isin(in_ucdp), "K"].fillna(0.0)
    absent = sorted(set(panel.iso3) - in_ucdp)
    note(f"UCDP: economies not in UCDP at all, K left missing (D-011): {absent}  [CONFIRM]")
    panel = panel.merge(fao_temperature(raw), on=["iso3", "year"], how="left")
    panel = panel.merge(cckp_precip(raw), on=["iso3", "year"], how="left")
    tc = cckp_temperature(raw)
    if tc is not None:
        panel = panel.merge(tc, on=["iso3", "year"], how="left")
    for (iso, var), y in config.SERIES_BREAKS.items():
        mask = (panel.iso3 == iso) & (panel.year < y)
        panel.loc[mask, var] = np.nan
        note(f"Series break (D-012): {iso} {var} set missing before {y} ({int(mask.sum())} cells)  [CONFIRM]")
    m, src = idmc(raw)
    in_idmc = set(m.iso3)
    panel = panel.merge(m, on=["iso3", "year"], how="left")
    covered = panel.iso3.isin(in_idmc)
    panel["M_absent_set_zero"] = covered & panel["M"].isna()
    panel.loc[panel["M_absent_set_zero"], "M"] = 0.0
    for iso, y in sorted(config.OUTCOME_SET_MISSING):
        mask = (panel.iso3 == iso) & (panel.year == y)
        panel.loc[mask, "M"] = np.nan
        note(f"Outcome set missing (D-020): {iso} {y} ({int(mask.sum())} cell)  [CONFIRM]")
    note(f"IDMC: {len(in_idmc)} countries in file; absent country-years for covered countries set to 0 "
         f"({int(panel['M_absent_set_zero'].sum())} cells, flagged in column M_absent_set_zero)")
    try:
        panel = panel.merge(unhcr(raw), on=["iso3", "year"], how="left")
    except Exception as e:
        note(f"UNHCR secondary outcome not built: {e}")
    panel["idmc_source"] = src
    out = p["processed"] / "panel_raw_units.csv"
    panel.to_csv(out, index=False)
    (p["processed"] / "BUILD_LOG.txt").write_text("\n".join(LOG))
    note(f"Saved {out} with {len(panel)} rows and {panel.iso3.nunique()} countries")
    return panel
