# PiLens: dimensionless analysis of climate-related migration

Code and protocol for two manuscripts by Tanya Purwar, Rushabh Shah, Maissa Khatib, Neha Shakelly and Luciano Castillo:

- **Paper 0** (Climatic Change, in preparation): non-dimensional framework for climate migration, rebuilt on public data for 196 countries, 2008 to 2023.
- **Paper 1** (Environmental Research Communications, in preparation): pre-registered test of Buckingham Pi groups for migration, with shared-term and scaling checks.

## What is here
| Path | What it does |
|---|---|
| `pilens/download.py`, `pilens/paper0_data.py` | Download every input from the public source (UNHCR, World Bank WDI, UCDP, OWID, FAOSTAT, CRU TS, LECZ, IDMC). |
| `pilens/build.py`, `pilens/pigroups.py`, `pilens/verify.py`, `pilens/qa.py` | Build the country-year panel, form the Pi groups, check units and data quality. |
| `pilens/analysis.py`, `pilens/paper.py` | Paper 1: pre-registered tests (H1 to H3), figures and supplementary tables. |
| `pilens/paper0.py`, `pilens/paper0_tex.py` | Paper 0: clustering, correlations, power laws, reference tests, forecasts, figures and tables. |
| `pilens/exposure.py`, `pilens/phase1.py`, `pilens/export.py` | Study 2 (in progress): Earth Engine hazard exposure by country. |
| `pilens/config.py` | Paths and settings. |
| `protocol/` | Analysis plan, parameter specification and decision log (every change to the method, dated, with its reason). |

## How to reproduce
Python 3.10 or later. In Google Colab or locally:

```bash
pip install -r requirements.txt
```

```python
from pathlib import Path
from pilens import paper, paper0, paper0_tex
PROJECT = Path("path/to/project")   # data, results and manuscript folders are created here
paper.run(PROJECT)                  # Paper 1: numbers, figures, tables
paper0.run(PROJECT)                 # Paper 0: results and figures
paper0_tex.run(PROJECT)             # Paper 0: tables
```

Raw data are not stored here; the download functions fetch them from the original sources. Earth Engine steps (Study 2) need a registered Earth Engine project.

## Data sources
UNHCR Refugee Data Finder; World Bank World Development Indicators; UCDP organized violence; Our World in Data CO2; FAOSTAT temperature change; CRU TS 4; LECZ (CIESIN); IDMC Global Internal Displacement Database. Full citations are in the manuscripts.

## Citation
Use the Zenodo DOI shown on the badge in the Zenodo record for this repository, or `CITATION.cff`.

## License
MIT (see `LICENSE`).
