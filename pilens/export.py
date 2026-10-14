"""Step 1c: write readable copies of every raw source as CSV, plus one Excel workbook.

The original downloads in 02_Raw_Data stay untouched (they are what MANIFEST.csv hashes).
This only reshapes them into tables: one row per country (or event) and year, values exactly
as published, no unit conversion. Output: 02_Raw_Data/readable_csv/*.csv and PiLens_raw_data.xlsx.
"""
import json
import zipfile
from pathlib import Path

import pandas as pd

from . import config


def _wdi(raw):
    """All WDI indicators in one wide table: iso3, country, year, one column per indicator code."""
    names = {c["id"]: c["name"] for c in json.loads((raw / "wdi" / "wdi_countries.json").read_text())[1]}
    out = None
    for sym, (code, unit) in config.WDI.items():
        f = raw / "wdi" / f"{code}.json"
        if not f.exists():
            continue
        d = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), f"{code} ({unit})": r["value"]}
                          for r in json.loads(f.read_text())[1] if r.get("countryiso3code")])
        out = d if out is None else out.merge(d, on=["iso3", "year"], how="outer")
    out.insert(1, "country", out.iso3.map(names))
    return out.sort_values(["iso3", "year"])


def _ucdp(raw):
    z = next((raw / "ucdp").glob("*.zip"))
    with zipfile.ZipFile(z) as zf:
        return pd.read_csv(zf.open(next(n for n in zf.namelist() if n.lower().endswith(".csv"))))


def _fao(raw):
    z = next((raw / "fao_temperature").glob("*.zip"))
    with zipfile.ZipFile(z) as zf:
        n = next(n for n in zf.namelist() if n.lower().endswith(".csv") and "all_data" in n.lower())
        df = pd.read_csv(zf.open(n), encoding="latin-1")
    return df


def _cckp(raw, folder, label):
    f = next((raw / folder).glob("*.json"), None)
    if f is None:
        return None
    data = json.loads(f.read_text())["data"]
    rows = [(iso, int(str(k)[:4]), v) for iso, s in data.items() if isinstance(s, dict) for k, v in s.items()]
    return pd.DataFrame(rows, columns=["iso3", "year", label]).sort_values(["iso3", "year"])


def _unhcr(raw):
    return pd.DataFrame(json.loads((raw / "unhcr" / "unhcr_population_by_origin.json").read_text()))


def _idmc(raw):
    f = next(p for p in (raw / "idmc_disasters").iterdir() if p.suffix.lower() in (".xlsx", ".xls", ".csv"))
    return pd.read_csv(f) if f.suffix.lower() == ".csv" else pd.read_excel(f)


def run(project_root):
    raw = config.paths(project_root)["raw"]
    out = raw / "readable_csv"
    out.mkdir(exist_ok=True)
    tables = {
        "wdi_world_bank": (_wdi, "World Bank WDI, one column per indicator, units as published"),
        "ucdp_conflict_deaths": (_ucdp, "UCDP Organized Violence country-year v26.1, all columns"),
        "faostat_temperature_change": (_fao, "FAOSTAT temperature change, all months and elements, deg C"),
        "cckp_precipitation_mm": (lambda r: _cckp(r, "cckp_precip", "precipitation_mm_per_year"), "CRU TS annual precipitation, mm"),
        "cckp_temperature_c": (lambda r: _cckp(r, "cckp_tas", "mean_temperature_c"), "CRU TS annual mean temperature, deg C"),
        "unhcr_refugees_by_origin": (_unhcr, "UNHCR population statistics by country of origin"),
        "idmc_disaster_displacement_events": (_idmc, "IDMC GIDD disaster events (copy of the manual download)"),
    }
    written, frames = [], {}
    for name, (fn, desc) in tables.items():
        try:
            df = fn(raw)
        except Exception as e:
            print(f"  {name}: not written ({e})")
            continue
        if df is None:
            print(f"  {name}: source file not downloaded yet")
            continue
        df.to_csv(out / f"{name}.csv", index=False)
        frames[name] = (df, desc)
        written.append((name, len(df), desc))
        print(f"  {name}.csv: {len(df):,} rows")
    # One workbook with a sheet per source. FAOSTAT is large, so the workbook keeps its annual rows only;
    # the full table is in the CSV.
    xl = out / "PiLens_raw_data.xlsx"
    with pd.ExcelWriter(xl, engine="openpyxl") as w:
        pd.DataFrame(written, columns=["sheet", "rows", "description"]).to_excel(w, sheet_name="README", index=False)
        for name, (df, _) in frames.items():
            if name.startswith("faostat"):
                df = df[df["Months"].astype(str).str.contains("Meteorological year", case=False)]
            df.to_excel(w, sheet_name=name[:31], index=False)
    print(f"Saved {len(written)} CSV files and {xl.name} in {out}")
    return written
