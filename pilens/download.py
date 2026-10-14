"""Step 1: download every source unchanged into 02_Raw_Data and log it in MANIFEST.csv.

Nothing here edits data. Each source gets its own folder. A failed download is logged with
the reason and the manual alternative, so a failure is never silent.
"""
import csv
import datetime as dt
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import requests

from . import config

HEADERS = {"User-Agent": "pilens-research-pipeline/1.0 (academic research)"}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _log(raw_dir, source, url, path, status, note=""):
    manifest = Path(raw_dir) / "MANIFEST.csv"
    new = not manifest.exists()
    with open(manifest, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp_utc", "source", "url", "file", "sha256", "bytes", "status", "note"])
        sha = _sha256(path) if path and Path(path).exists() else ""
        size = Path(path).stat().st_size if path and Path(path).exists() else ""
        w.writerow([dt.datetime.utcnow().isoformat(timespec="seconds"), source, url,
                    str(Path(path).name) if path else "", sha, size, status, note])


def _get(url, timeout=120, tries=3):
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = repr(e)
        time.sleep(2 * (i + 1))
    raise RuntimeError(last)


# ------------------------------------------------------------------ sources
def wdi(raw_dir):
    out = Path(raw_dir) / "wdi"
    out.mkdir(parents=True, exist_ok=True)
    url = config.URLS["wdi_countries"]
    try:
        r = _get(url)
        f = out / "wdi_countries.json"
        f.write_bytes(r.content)
        _log(raw_dir, "WDI country list", url, f, "ok")
    except Exception as e:
        _log(raw_dir, "WDI country list", url, None, "FAILED", str(e))
    for sym, (code, _) in config.WDI.items():
        url = config.URLS["wdi_indicator"].format(code=code)
        try:
            r = _get(url)
            payload = r.json()
            if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
                raise RuntimeError(f"unexpected response: {str(payload)[:200]}")
            if payload[0].get("pages", 1) > 1:
                raise RuntimeError("more than one page returned; raise per_page")
            f = out / f"{code}.json"
            f.write_bytes(r.content)
            _log(raw_dir, f"WDI {code} ({sym})", url, f, "ok", f"{len(payload[1])} rows")
        except Exception as e:
            _log(raw_dir, f"WDI {code} ({sym})", url, None, "FAILED",
                 f"{e}. Manual: download CSV for {code} from https://data.worldbank.org/indicator/{code}")


def ucdp(raw_dir):
    out = Path(raw_dir) / "ucdp"
    out.mkdir(parents=True, exist_ok=True)
    url = config.URLS["ucdp_cy"]
    try:
        r = _get(url)
        f = out / Path(url).name
        f.write_bytes(r.content)
        zipfile.ZipFile(f).testzip()
        _log(raw_dir, "UCDP organized violence country-year v26.1", url, f, "ok")
    except Exception as e:
        _log(raw_dir, "UCDP organized violence country-year v26.1", url, None, "FAILED",
             f"{e}. Manual: https://ucdp.uu.se/downloads/ -> 'UCDP Country-Year Dataset on Organized Violence' CSV")


def fao_temperature(raw_dir):
    out = Path(raw_dir) / "fao_temperature"
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in config.URLS["fao_temp"]:
        try:
            r = _get(url, timeout=300)
            f = out / "Environment_Temperature_change_E_All_Data_(Normalized).zip"
            f.write_bytes(r.content)
            zipfile.ZipFile(f).testzip()
            _log(raw_dir, "FAOSTAT Temperature change (ET)", url, f, "ok")
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
    _log(raw_dir, "FAOSTAT Temperature change (ET)", "; ".join(config.URLS["fao_temp"]), None, "FAILED",
         " | ".join(errors) + ". Manual: https://www.fao.org/faostat/en/#data/ET -> Bulk download -> All data normalized")


def _cckp(raw_dir, key, folder, fname, label, manual):
    out = Path(raw_dir) / folder
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in config.URLS[key]:
        try:
            r = _get(url, timeout=300)
            payload = r.json()
            if "data" not in payload:
                raise RuntimeError(f"no 'data' key; keys={list(payload)[:5]}")
            f = out / fname
            f.write_bytes(r.content)
            _log(raw_dir, label, url, f, "ok")
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
    _log(raw_dir, label, "; ".join(config.URLS[key]), None, "FAILED", " | ".join(errors) + ". Manual: " + manual)


def cckp_precip(raw_dir):
    _cckp(raw_dir, "cckp_precip", "cckp_precip", "cckp_cru_pr_annual_all_countries.json",
          "World Bank CCKP precipitation (CRU TS, annual)", config.CCKP_MANUAL)


def cckp_temperature(raw_dir):
    _cckp(raw_dir, "cckp_tas", "cckp_tas", "cckp_cru_tas_annual_all_countries.json",
          "World Bank CCKP mean temperature (CRU TS, annual)", config.CCKP_TAS_MANUAL)


def unhcr(raw_dir):
    out = Path(raw_dir) / "unhcr"
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = config.YEARS
    items, page, maxpages = [], 1, 1
    first_url = config.URLS["unhcr_population"].format(y0=y0, y1=y1, page=1)
    try:
        while page <= maxpages:
            url = config.URLS["unhcr_population"].format(y0=y0, y1=y1, page=page)
            payload = _get(url).json()
            items.extend(payload.get("items", []))
            maxpages = int(payload.get("maxPages", 1) or 1)
            page += 1
        if not items:
            raise RuntimeError("no items returned")
        f = out / "unhcr_population_by_origin.json"
        f.write_text(json.dumps(items))
        _log(raw_dir, "UNHCR population statistics (by origin)", first_url, f, "ok", f"{len(items)} rows")
    except Exception as e:
        _log(raw_dir, "UNHCR population statistics (by origin)", first_url, None, "FAILED",
             f"{e}. Manual: https://www.unhcr.org/refugee-statistics/download/ (by country of origin, {y0}-{y1})")


def owid_crosscheck(raw_dir):
    out = Path(raw_dir) / "owid"
    out.mkdir(parents=True, exist_ok=True)
    url = config.URLS["owid_co2"]
    try:
        f = out / "owid-co2-data.csv"
        f.write_bytes(_get(url, timeout=300).content)
        _log(raw_dir, "OWID CO2 dataset (population cross-check only)", url, f, "ok")
    except Exception as e:
        _log(raw_dir, "OWID CO2 dataset (population cross-check only)", url, None, "FAILED", str(e))


def idmc_check(raw_dir):
    """IDMC is downloaded by hand. If no file is present, try the OWID copy and flag it as secondary."""
    out = Path(raw_dir) / "idmc_disasters"
    out.mkdir(parents=True, exist_ok=True)
    files = [p for p in out.iterdir() if p.suffix.lower() in (".xlsx", ".xls", ".csv") and not p.name.startswith("OWID_")]
    if files:
        for f in files:
            _log(raw_dir, "IDMC GIDD disasters (manual download)", "manual", f, "ok")
        return "manual file found"
    url = config.URLS["owid_idmc_fallback"]
    try:
        f = out / "OWID_internal-displacement.csv"
        f.write_bytes(_get(url).content)
        _log(raw_dir, "IDMC via Our World in Data (FALLBACK, replace with manual IDMC file)", url, f, "ok-fallback",
             config.IDMC_MANUAL)
        return "fallback used"
    except Exception as e:
        _log(raw_dir, "IDMC GIDD disasters", "manual", None, "MISSING", f"{e}. {config.IDMC_MANUAL}")
        return "missing"


def run_all(raw_dir):
    for fn in (wdi, ucdp, fao_temperature, cckp_precip, cckp_temperature, unhcr, owid_crosscheck, idmc_check):
        print(f"-> {fn.__name__}")
        fn(raw_dir)
    import pandas as pd
    m = pd.read_csv(Path(raw_dir) / "MANIFEST.csv")
    last = m.groupby("source").tail(1)
    show = last[["source", "status", "note"]].copy()
    show["note"] = show["note"].fillna("").astype(str).str.slice(0, 160)
    print(show.to_string(index=False))
    return last
