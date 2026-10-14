"""Paper 0 (the March 2026 manuscript, rebuilt): extra raw data it needs beyond Study 1.

download_unhcr_asylum(): UNHCR refugees and asylum seekers by country of ASYLUM (inflow side of the March
paper's migration variable M, which combined inflow and outflow). Study 1 only downloaded the origin side.
Writes 02_Raw_Data/unhcr/unhcr_population_by_asylum.json (unchanged API rows), logs it in MANIFEST.csv,
and writes a readable copy 02_Raw_Data/readable_csv/unhcr_refugees_by_asylum.csv (iso3, year, hosted).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .download import _get, _log

URL = ("https://api.unhcr.org/population/v1/population/?yearFrom={y0}&yearTo={y1}&coa_all=true"
       "&cf_type=ISO&limit=1000&page={page}")


def download_unhcr_asylum(project_root):
    raw = config.paths(project_root)["raw"]
    out = raw / "unhcr"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = config.YEARS
    items, page, maxpages = [], 1, 1
    while page <= maxpages:
        payload = _get(URL.format(y0=y0, y1=y1, page=page)).json()
        items.extend(payload.get("items", []))
        maxpages = int(payload.get("maxPages", 1) or 1)
        page += 1
    if not items:
        raise RuntimeError("UNHCR returned no rows")
    f = out / "unhcr_population_by_asylum.json"
    f.write_text(json.dumps(items))
    _log(raw, "UNHCR population statistics (by country of asylum)", URL.format(y0=y0, y1=y1, page=1), f, "ok",
         f"{len(items)} rows")
    hosted = asylum_table(raw)
    rd = raw / "readable_csv"
    rd.mkdir(exist_ok=True)
    hosted.to_csv(rd / "unhcr_refugees_by_asylum.csv", index=False)
    print(f"UNHCR by asylum: {len(items)} rows -> {hosted.iso3.nunique()} countries, years "
          f"{hosted.year.min()}-{hosted.year.max()}; saved {rd / 'unhcr_refugees_by_asylum.csv'}")
    return hosted


def asylum_table(raw):
    df = pd.DataFrame(json.loads((Path(raw) / "unhcr" / "unhcr_population_by_asylum.json").read_text()))
    for c in ("refugees", "asylum_seekers"):
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce")
    g = df.groupby(["coa_iso", "year"], as_index=False)[["refugees", "asylum_seekers"]].sum(min_count=1)
    g["hosted"] = g[["refugees", "asylum_seekers"]].sum(axis=1, min_count=1)
    return g.rename(columns={"coa_iso": "iso3"})[["iso3", "year", "hosted"]]
