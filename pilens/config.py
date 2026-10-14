"""Single place for every setting. Change values here only, and log the change in DECISION_LOG.csv."""
from pathlib import Path

YEARS = (2008, 2023)              # analysis window (ANALYSIS_PLAN section 3)
TRAIN_END = 2019                  # H3: fit through this year, test after it
MIN_POP_2015 = 500_000            # sample threshold (robustness: 0)
CLIMATE_BASELINE = (1991, 2020)   # WMO climate normal; baseline for temperature and precipitation anomalies
PRECIP_BASELINE = CLIMATE_BASELINE
N_PERMUTATIONS = 1000             # H1
RANDOM_SEED = 2026

# World Bank WDI indicators: our symbol -> (code, unit as published)
WDI = {
    "Np":   ("SP.POP.TOTL",        "persons"),
    "A_L":  ("AG.LND.TOTL.K2",     "km2"),
    "A_R":  ("AG.LND.ARBL.HA",     "hectares"),
    "Y":    ("NY.GDP.PCAP.PP.KD",  "constant 2021 international $ per person per year"),
    "HY":   ("SH.XPD.CHEX.GD.ZS",  "% of GDP (health spending and GDP in the same current prices)"),
    "D_wdi": ("EN.POP.DNST",       "persons per km2 (used for Pi8, D-015)"),
    "LECZ": ("EN.POP.EL5M.ZS",     "% of population below 5 m"),
}

URLS = {
    "wdi_indicator": "https://api.worldbank.org/v2/country/all/indicator/{code}?date=2000:2024&format=json&per_page=20000",
    "wdi_countries": "https://api.worldbank.org/v2/country?format=json&per_page=400",
    "ucdp_cy": "https://ucdp.uu.se/downloads/organizedviolencecy/organizedviolencecy-261-csv.zip",
    # verified 2026-10-13 in FAOSTAT's dataset index (bulks-faostat.fao.org/production/datasets_E.xml), updated 2026-04-01
    "fao_temp": [
        "https://bulks-faostat.fao.org/production/Environment_Temperature_change_E_All_Data_(Normalized).zip",
        "https://fenixservices.fao.org/faostat/static/bulkdownloads/Environment_Temperature_change_E_All_Data_(Normalized).zip",
    ],
    # First URL worked in Colab on 2026-10-13 (MANIFEST). CCKP does not document it, so CCKP_MANUAL stays as fallback.
    "cckp_precip": [
        "https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_pr_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean/all_countries?_format=json",
        "https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_pr_timeseries_annual_1901-2022_mean_historical_cru_ts4.07_mean/all_countries?_format=json",
    ],
    # Same CRU TS series and API pattern as precipitation, variable 'tas' (annual mean near-surface temperature, deg C)
    "cckp_tas": [
        "https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_tas_timeseries_annual_1901-2023_mean_historical_cru_ts4.08_mean/all_countries?_format=json",
        "https://cckpapi.worldbank.org/cckp/v1/cru-x0.5_timeseries_tas_timeseries_annual_1901-2022_mean_historical_cru_ts4.07_mean/all_countries?_format=json",
    ],
    "unhcr_population": "https://api.unhcr.org/population/v1/population/?yearFrom={y0}&yearTo={y1}&coo_all=true&cf_type=ISO&limit=1000&page={page}",
    "owid_co2": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
    "owid_idmc_fallback": "https://ourworldindata.org/grapher/internal-displacement.csv?v=1&csvType=full&useColumnShortNames=true",
}

CCKP_MANUAL = (
    "Open https://climateknowledgeportal.worldbank.org/download-data , tab 'Spatially Aggregated Data', choose Global, "
    "collection CRU (cru-x0.5), type time series, aggregation annual, scenario historical, variable precipitation; "
    "export CSV or XLSX and save it unchanged in 02_Raw_Data/cckp_precip/ ."
)
CCKP_TAS_MANUAL = CCKP_MANUAL.replace("variable precipitation", "variable mean temperature (tas)").replace("cckp_precip", "cckp_tas")
# A CRU country series where the same value appears in this many CONSECUTIVE years within 1991-2023 is
# station-free climatology fill: its anomaly is not real, so it is set to missing (D-014, amended D-018).
# Scattered repeats are coincidence (e.g. Libya temperature 22.91 C in 1998, 2004, 2012, 2014, 2023) and are kept.
CLIMATOLOGY_FILL_MIN_RUN = 4
# Definition breaks in a source series: values before the break year are set to missing (D-012).
SERIES_BREAKS = {("MNE", "A_R"): 2013}   # FAO arable land falls 1,730 -> 78 km2 in one year (reclassification)
# Outcome values that are physically implausible and are set to missing, raw file untouched (D-020).
# VUT 2015: one IDMC 'Countrywide' Cyclone Pam record of 600,000 displacements = 2.3x the population.
OUTCOME_SET_MISSING = {("VUT", 2015)}

UCDP_COLUMNS = {"gw": "country_id", "year": "year", "deaths": "cumulative_total_deaths_in_orgvio_best",
                "version": "Version"}   # as in OrganizedViolenceCYDataSet26_1.csv (checked on the downloaded file 2026-10-13).
# "cumulative_..." is the within-year total over state-based + non-state + one-sided violence (checked: equals their sum).
# Gleditsch-Ward codes that country_converter does not map, but which are current states (D-011)
GW_TO_ISO3_OVERRIDE = {"316": "CZE", "345": "SRB", "678": "YEM", "816": "VNM"}
IDMC_DISPLACEMENT_COLUMN = "Disaster Internal Displacements (Raw)"   # unrounded event figures (D-010)

# Manual download (IDMC has no open API without a client id)
IDMC_MANUAL = (
    "Open https://www.internal-displacement.org/database/displacement-data , choose 'Disasters', "
    "download the Excel/CSV of internal displacements (all countries, all years), and save it "
    "unchanged in 02_Raw_Data/idmc_disasters/ . Any file name is fine."
)


def paths(project_root):
    root = Path(project_root)
    p = {
        "root": root,
        "protocol": root / "01_Protocol",
        "raw": root / "02_Raw_Data",
        "code": root / "03_Code",
        "processed": root / "04_Processed_Data",
        "results": root / "05_Results",
    }
    for k in ("raw", "processed", "results"):
        p[k].mkdir(parents=True, exist_ok=True)
    return p
