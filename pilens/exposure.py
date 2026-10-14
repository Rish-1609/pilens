"""Study 2, Phase 1: population exposure shares computed in Google Earth Engine (STUDY2_PHASE1_PLAN.md, D-026).

For each country and year 2008-2018 it sums population (GHS-POP R2023A, interpolated between 5-year epochs)
inside: flood extents (Global Flood Database v1), 50/100/150 km of hurricane-strength cyclone track points
(IBTrACS v04r01, wind >= 64 kt), and drought cells (December SPEI-12 <= -1.0/-1.5/-2.0).
Shares = exposed population / total population of the same country polygon, so numerator and denominator
come from the same population grid.

Earth Engine runs the heavy work as export tasks (one per year and country batch, D-028) that write CSVs to My Drive/PiLens_EE_exports.
collect() then copies them into 02_Raw_Data/ee_exposure/, logs them in MANIFEST.csv and builds one table.
"""
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

YEARS = list(range(2008, 2019))
EXPORT_FOLDER = "PiLens_EE_exports"
IBTRACS_URL = ("https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/"
               "v04r01/access/csv/ibtracs.since1980.list.v04r01.csv")
COUNTRIES = "WM/geoLab/geoBoundaries/600/ADM0"          # property shapeGroup = ISO3
GHS_POP = "JRC/GHSL/P2023A/GHS_POP"
GFD = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
SPEI_CANDIDATES = ["CSIC/SPEI/2_11", "CSIC/SPEI/2_10", "CSIC/SPEI/2_9"]
RADII_KM = [50, 100, 150]                 # 100 km is the pre-registered value; 50 and 150 are robustness
SPEI_THRESHOLDS = [-1.0, -1.5, -2.0]      # -1.5 is the pre-registered value
SCALE_M = 100                             # GHS-POP native resolution
N_BATCHES = 4                             # D-028: each year is split into 4 country batches (one task each)
PREFIX = "pilensv2"                       # D-028 exports; older "pilens_exposure_*" files are ignored
# Largest countries first, then dealt round-robin into batches so no single task holds all of them.
BIG = ["RUS", "CAN", "USA", "CHN", "BRA", "AUS", "IND", "ARG", "KAZ", "DZA", "COD", "SAU", "MEX", "IDN",
       "SDN", "LBY", "IRN", "MNG", "PER", "TCD", "NER", "AGO", "MLI", "ZAF", "COL", "ETH", "BOL", "MRT", "EGY"]


def init(project_id):
    import ee
    try:
        ee.Initialize(project=project_id)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project_id)
    return ee


def check_sources(ee):
    """Confirm every Earth Engine dataset exists and has the expected bands, before any export starts."""
    out = {}
    adm0 = ee.FeatureCollection(COUNTRIES)
    out["countries"] = {"n": adm0.size().getInfo(), "fields": adm0.first().propertyNames().getInfo()}
    pop = ee.ImageCollection(GHS_POP)
    out["ghs_pop"] = {"bands": pop.first().bandNames().getInfo(),
                      "epochs": pop.aggregate_array("system:index").getInfo()}
    gfd = ee.ImageCollection(GFD)
    out["gfd"] = {"n_events": gfd.size().getInfo(), "bands": gfd.first().bandNames().getInfo(),
                  "events_2008_2018": gfd.filterDate("2008-01-01", "2019-01-01").size().getInfo()}
    for cid in SPEI_CANDIDATES:
        try:
            s = ee.ImageCollection(cid)
            n = s.filterDate("2018-12-01", "2019-01-01").size().getInfo()
            if n:
                out["spei"] = {"id": cid, "bands": s.first().bandNames().getInfo()}
                break
        except Exception:
            continue
    for k, v in out.items():
        print(k, v)
    assert "shapeGroup" in out["countries"]["fields"], "country boundaries lack shapeGroup (ISO3)"
    assert "population_count" in out["ghs_pop"]["bands"], "GHS-POP band population_count not found"
    assert "flooded" in out["gfd"]["bands"] and "jrc_perm_water" in out["gfd"]["bands"], "GFD bands not found"
    assert "spei" in out and "SPEI_12_month" in out["spei"]["bands"], "SPEI-12 not found in Earth Engine"
    return out


def pop_image(ee, year):
    """GHS-POP for a year, linear between the 5-year epochs that bracket it."""
    col = ee.ImageCollection(GHS_POP).select("population_count")
    a = (year // 5) * 5
    b = a + 5
    ia = col.filterDate(f"{a}-01-01", f"{a + 1}-01-01").first()
    ib = col.filterDate(f"{b}-01-01", f"{b + 1}-01-01").first()
    w = (year - a) / 5.0
    return ia.multiply(1 - w).add(ib.multiply(w)).unmask(0).rename("pop")


def flood_mask(ee, year):
    gfd = ee.ImageCollection(GFD)
    events = gfd.filterDate(f"{year}-01-01", f"{year + 1}-01-01")            # attributed to the start year
    flooded = events.select("flooded").max().unmask(0)
    # D-028: permanent water from the same year's events (the JRC layer is identical in every event image, and
    # flooded pixels only exist inside that year's footprints), instead of a max over all 913 events.
    perm = events.select("jrc_perm_water").max().unmask(0)
    return flooded.gt(0).And(perm.eq(0))


def cyclone_points(raw_dir):
    """Download IBTrACS (unchanged) and return hurricane-strength track points per year."""
    out = Path(raw_dir) / "ibtracs"
    out.mkdir(parents=True, exist_ok=True)
    f = out / Path(IBTRACS_URL).name
    if not f.exists():
        r = requests.get(IBTRACS_URL, timeout=600)
        r.raise_for_status()
        f.write_bytes(r.content)
        _manifest(raw_dir, "IBTrACS v04r01 since 1980", IBTRACS_URL, f)
    df = pd.read_csv(f, skiprows=[1], usecols=["SID", "ISO_TIME", "LAT", "LON", "USA_WIND", "WMO_WIND"],
                     low_memory=False)
    wind = pd.concat([pd.to_numeric(df.USA_WIND, errors="coerce"),
                      pd.to_numeric(df.WMO_WIND, errors="coerce")], axis=1).max(axis=1)
    df["year"] = pd.to_numeric(df.ISO_TIME.astype(str).str[:4], errors="coerce")
    df = df[(wind >= 64) & df.year.isin(YEARS)].copy()
    df["LON"] = ((pd.to_numeric(df.LON) + 180) % 360) - 180
    df["LAT"] = pd.to_numeric(df.LAT)
    df = df.round({"LAT": 2, "LON": 2}).drop_duplicates(["year", "LAT", "LON"])
    print(f"IBTrACS: {len(df)} hurricane-strength track points 2008-2018")
    return df[["year", "LAT", "LON"]]


def cyclone_mask(ee, pts_year, radius_km):
    """D-028: paint geodesic buffers (max error 1 km) instead of a per-pixel distance transform; same footprint."""
    r = radius_km * 1000
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(lo), float(la)]))
                               for la, lo in zip(pts_year.LAT, pts_year.LON)]).map(lambda f: f.buffer(r, 1000))
    return ee.Image(0).byte().paint(fc, 1)


def drought_mask(ee, spei_id, year, thr):
    img = ee.ImageCollection(spei_id).filterDate(f"{year}-12-01", f"{year + 1}-01-01").first() \
        .select("SPEI_12_month")
    return img.lte(thr).unmask(0)


def batches(iso):
    order = [c for c in BIG if c in iso] + sorted(c for c in iso if c not in BIG)
    return [order[k::N_BATCHES] for k in range(N_BATCHES)]


def cancel_unfinished(ee):
    """Cancel every queued or running pilens export (used before restarting with new code)."""
    n = 0
    for t in ee.batch.Task.list():
        st = t.status()
        if st["state"] in ("READY", "RUNNING") and st.get("description", "").startswith("pilens"):
            t.cancel()
            n += 1
    print(f"cancelled {n} unfinished pilens tasks")


def start_exports(ee, project_root, spei_id, years=YEARS):
    """One export task per year and country batch, written to My Drive/PiLens_EE_exports."""
    p = config.paths(project_root)
    pts = cyclone_points(p["raw"])
    iso = pd.read_csv(p["processed"] / "panel_raw_units.csv", usecols=["iso3"]).iso3.unique().tolist()
    groups = batches(iso)
    cols = ["shapeGroup", "year", "pop", "F"] + [f"C{r}" for r in RADII_KM] + \
           [f"D{int(abs(t) * 10)}" for t in SPEI_THRESHOLDS]
    tasks = []
    for y in years:
        # D-027: GHS-POP counts sit on a 100 m equal-area Mollweide grid. Summing in that grid fails for polygons
        # near its edge (dateline countries), so convert to people per m2 and multiply by each pixel's true area,
        # then sum in EPSG:4326. Totals stay exact whatever the output grid.
        pop = pop_image(ee, y).divide(1e4).multiply(ee.Image.pixelArea()).rename("pop")
        bands = [pop, pop.multiply(flood_mask(ee, y)).rename("F")]
        py = pts[pts.year == y]
        for r in RADII_KM:
            m = cyclone_mask(ee, py, r) if len(py) else ee.Image(0)
            bands.append(pop.multiply(m).rename(f"C{r}"))
        for t in SPEI_THRESHOLDS:
            bands.append(pop.multiply(drought_mask(ee, spei_id, y, t)).rename(f"D{int(abs(t) * 10)}"))
        img = ee.Image.cat(bands)
        for k, g in enumerate(groups):
            adm0 = ee.FeatureCollection(COUNTRIES).filter(ee.Filter.inList("shapeGroup", g))
            fc = img.reduceRegions(collection=adm0, reducer=ee.Reducer.sum(), crs="EPSG:4326", scale=SCALE_M,
                                   tileScale=16).map(lambda f: f.set("year", y))
            name = f"{PREFIX}_{y}_b{k}"
            task = ee.batch.Export.table.toDrive(collection=fc, description=name, folder=EXPORT_FOLDER,
                                                 fileNamePrefix=name, fileFormat="CSV", selectors=cols)
            task.start()
            tasks.append(task)
        print(f"started {len(groups)} export tasks for {y}")
    print(f"total {len(tasks)} tasks")
    return tasks


def wait(tasks, poll_s=60):
    while True:
        states = [t.status()["state"] for t in tasks]
        print(time.strftime("%H:%M:%S"), {s: states.count(s) for s in set(states)})
        if all(s in ("COMPLETED", "FAILED", "CANCELLED") for s in states):
            for t in tasks:
                st = t.status()
                if st["state"] != "COMPLETED":
                    print(st.get("description"), st["state"], st.get("error_message"))
            return states
        time.sleep(poll_s)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(raw_dir, source, url, path, note=""):
    import csv
    import datetime as dt
    m = Path(raw_dir) / "MANIFEST.csv"
    with open(m, "a", newline="") as fh:
        csv.writer(fh).writerow([dt.datetime.utcnow().isoformat(timespec="seconds"), source, url,
                                 Path(path).name, _sha(path), Path(path).stat().st_size, "ok", note])


def collect(project_root, drive_root="/content/drive/MyDrive"):
    """Copy the exported CSVs into 02_Raw_Data/ee_exposure, log them, and write the exposure-share table."""
    import shutil
    p = config.paths(project_root)
    # Earth Engine can create several folders with the same name when tasks start together, and Drive shows
    # them side by side. Look in every one and keep the newest file for each year.
    dirs = [d for d in Path(drive_root).iterdir() if d.is_dir() and d.name.startswith(EXPORT_FOLDER)]
    found = {}
    for d in dirs:
        for f in d.glob(f"{PREFIX}_*.csv"):
            _, y, b = f.stem.split("_")[:3]
            key = (int(y), int(b[1:]))
            if key not in found or f.stat().st_mtime > found[key].stat().st_mtime:
                found[key] = f
    dst = p["raw"] / "ee_exposure"
    dst.mkdir(exist_ok=True)
    missing = [(y, k) for y in YEARS for k in range(N_BATCHES) if (y, k) not in found]
    if missing:
        raise RuntimeError(f"{len(missing)} export files not there yet, e.g. {missing[:5]} (looked in {[str(d) for d in dirs]})")
    frames = []
    for key in sorted(found):
        f = found[key]
        g = dst / f.name
        shutil.copy2(f, g)
        _manifest(p["raw"], "Earth Engine exposure export (exposure.py, D-028)", "earthengine", g,
                  f"GHS-POP R2023A; GFD v1; IBTrACS v04r01 >=64 kt; SPEI-12; scale {SCALE_M} m")
        frames.append(pd.read_csv(g))
    d = pd.concat(frames).rename(columns={"shapeGroup": "iso3"})
    assert not d.duplicated(["iso3", "year"]).any(), "a country-year appears in two batches"
    iso = set(pd.read_csv(p["processed"] / "panel_raw_units.csv", usecols=["iso3"]).iso3)
    absent = sorted(iso - set(d.iso3))
    if absent:
        print(f"WARNING: {len(absent)} sample countries missing from the boundary file: {absent}")
    for c in [c for c in d.columns if c[0] in "FCD"]:
        d[f"share_{c}"] = d[c] / d["pop"]
    # sanity check: grid population vs World Bank population (should be close; shares do not depend on it)
    wdi = pd.read_csv(p["processed"] / "panel_raw_units.csv", usecols=["iso3", "year", "Np"])
    chk = d.merge(wdi, on=["iso3", "year"], how="inner")
    ratio = chk["pop"] / chk["Np"]
    print(f"GHS-POP / WDI population: median {ratio.median():.3f}; 5th-95th pct {ratio.quantile(.05):.3f}-{ratio.quantile(.95):.3f}")
    out = p["processed"] / "exposure_shares.csv"
    d.to_csv(out, index=False)
    print(f"Saved {out}: {d.iso3.nunique()} countries, years {d.year.min()}-{d.year.max()}")
    print(d.filter(like="share_").describe().T.round(4))
    return d
