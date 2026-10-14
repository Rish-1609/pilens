# Analysis plan (protocol) v1.0

**Status: DRAFT. Lock this before Step 2 runs.** Locking means every author signs the decision log (01_Protocol/DECISION_LOG.csv, entry D-001).
After that, any change to this file needs a new decision-log entry that states the change and the reason, and the entry must be made **before** the analysis is re-run. This rule protects the paper against "we changed the method after seeing the results".

---

## 1. Research question
Do dimensionless combinations of climate, demographic, economic and conflict variables explain and predict **disaster displacement** across countries and over time, better than simple baselines?

## 2. Hypotheses (each one is tested once, with a pass rule fixed now)
| ID | Hypothesis | Test | Passes if |
|---|---|---|---|
| H1 | The Pi groups carry real signal about displacement, beyond country size and fixed country and year effects | Block permutation test (Section 7.2) | p < 0.05 |
| H2 | The data respect dimensional homogeneity | Wald test of the dimensional restrictions on the unconstrained model (Section 7.3) | Restrictions not rejected at 5% |
| H3 | The model predicts held-out years better than naive rules | Train 2008–2019, test 2020–2023 (Section 7.4) | Lower median absolute log error than **both** persistence and the country historical mean |
| H4 | The climate groups add predictive value | Out-of-sample error with vs without Π3 and Π4 | Error falls, and the bootstrap 95% CI of the difference excludes 0 |

What the results mean for the paper is set in Section 9.

## 3. Unit of analysis and sample
- Unit: country-year.
- Years: 2008–2023. The IDMC disaster series is consistent from 2008 onward.
- Countries: World Bank "economies" (aggregates excluded) with a population of at least 500,000 in 2015. Robustness check: all countries.
- Countries are identified by ISO-3 code only. Names are never used to join tables.

## 4. Variables (raw units only; nothing is rescaled to 0–10)
| Symbol | Definition | Unit | Dimension | Source (code) |
|---|---|---|---|---|
| M | New internal displacements from disasters, all hazards; unrounded event figures summed by country-year (D-010) | persons / year | N T⁻¹ | IDMC Global Internal Displacement Database (GIDD), disasters |
| R | Refugees + asylum seekers by country of origin (secondary outcome) | persons | N | UNHCR Refugee Data Finder API |
| Np | Population | persons | N | World Bank WDI `SP.POP.TOTL` |
| A_L | Land area | km² | L² | WDI `AG.LND.TOTL.K2` |
| A_R | Arable land | hectares → km² | L² | WDI `AG.LND.ARBL.HA` (FAO) |
| Y | GDP per capita, PPP, constant 2021 international $ | $ / person / year | C N⁻¹ T⁻¹ | WDI `NY.GDP.PCAP.PP.KD` |
| H/Y | Current health expenditure as a share of GDP. WDI publishes the ratio, so both terms are in the same current prices | share | 1 (C N⁻¹ T⁻¹ ÷ C N⁻¹ T⁻¹) | WDI `SH.XPD.CHEX.GD.ZS` |
| K | Deaths from organized violence (best estimate); missing, not 0, for economies UCDP does not list (D-011) | deaths / year | N T⁻¹ | UCDP Country-Year Organized Violence v26.1 |
| Tc | Annual mean near-surface temperature (D-013) | °C | Θ | World Bank Climate Change Knowledge Portal (CRU TS) |
| T̄, σ_T | 1991–2020 mean and SD of Tc, per country (WMO climate normal) | °C | Θ | Computed from Tc |
| ΔT | Temperature change vs 1951–1980 (cross-check of Π3 only) | °C | Θ | FAOSTAT Temperature change (ET) |
| Pr | Annual precipitation | mm / year | L T⁻¹ | World Bank Climate Change Knowledge Portal (CRU TS) |
| P̄, σ_P | 1991–2020 mean and SD of Pr, per country | mm / year | L T⁻¹ | Computed from Pr |
| LECZ | Share of population below 5 m elevation (2015) | % → share | 1 | WDI `EN.POP.EL5M.ZS` (CIESIN LECZ v3) |
| τ | Reporting period | 1 year | T | Definition |
| Y_ref, D_ref | World median of Y, and of density D, in that year | as above | as above | Computed |
| D | Published population density, used for Π8 (D-015) | persons / km² | N L⁻² | WDI `EN.POP.DNST` |
| G | CO₂ per unit GDP (robustness only: the old paper's carbon variable) | kg / international-$ | M C⁻¹ | OWID CO₂ dataset (GitHub) |

**Dropped from the 2026 draft, with reasons** (decision-log entries D-002 to D-005):
- Carbon intensity (G): no causal path to displacement, and its earlier "dominant driver" result came from a shared term.
- Sea-level-rise rate (O): close to uniform across countries. Exposure is captured by LECZ instead.
- The Fragile States Index composites: these are expert scores with no units, so they cannot carry dimensions.
- Min–max scaling.

## 5. Dimensional analysis
Base dimensions are N (people), C (currency), L (length), T (time) and Θ (temperature).
The groups are **derived by code** (Step 4) as the null space of the dimension matrix, then checked by substitution. Two anomalies are dimensionless by construction: (Tc − T̄)/σ_T and (Pr − P̄)/σ_P.

The expected core groups are:

| Group | Formula | Meaning |
|---|---|---|
| Π0 (outcome) | M·τ / Np | Share of population displaced per year |
| Π1 | H / Y | Health-spending share of GDP (adaptive capacity) |
| Π2 | K·τ / Np | Conflict death rate |
| Π3 | (Tc − T̄) / σ_T | Standardized temperature anomaly vs 1991–2020 |
| Π4 | (Pr − P̄) / σ_P | Standardized precipitation anomaly (drought < 0 < flood) |
| Π5 | LECZ | Coastal exposure (time-invariant) |
| Π6 | Y / Y_ref | Income relative to the world median |
| Π7 | A_R / A_L | Arable share of land |
| Π8 | D / D_ref | Density relative to the world median |

The dimension matrix has 14 variables and rank 5, so Buckingham gives 9 independent groups. The eight above (Π5 is already a share) cover 8 of them. The ninth admissible group, σ_P·τ/√A_L, compares yearly rainfall variability depth to the country's length scale. It is excluded because no mechanism links it to displacement. Step 4 verifies all of this in code.

## 6. Rules that prevent the earlier errors
1. No min–max scaling. Variables are normalized only by dividing by a reference value, which keeps power laws intact.
2. Never regress a ratio on another ratio that shares a variable without (a) a permutation null and (b) the shared variable entered as its own term. Here: the outcome is modelled as a **count with a population offset** (PPML), and ln Np also appears as a free regressor in the homogeneity test.
3. No subgroup is selected by its outcome values.
4. Every number in the paper is produced by code in 03_Code from 02_Raw_Data. Nothing is typed by hand.
5. Raw files are never edited. Every download is logged in `02_Raw_Data/MANIFEST.csv` with its URL, date, version and SHA-256 hash.
6. A climate anomaly is set to missing where the same CRU value appears in 4 or more consecutive years of 1991–2023 (station-free climatology fill, D-014, D-018). A definition break in a source series is handled by setting the earlier values to missing (D-012).
7. Missing is not zero, with one documented exception: a country-year in IDMC GIDD coverage with no recorded disaster event is set to 0 displacements (sensitivity check: treat it as missing).

## 7. Models and tests
### 7.1 Main model (Poisson pseudo-maximum likelihood with population offset)
E[M_it] = Np_it · exp(α_i + δ_t + β1 ln Π1 + β2 ln(Π2 + ε) + β3 Π3 + β4 Π4 + β6 ln Π6)

- α_i are country fixed effects, so this estimates **within-country** effects. Time-invariant groups (Π5, Π7) are identified only in the between-country model (7.5).
- ε is the smallest positive value of Π2.
- Standard errors are clustered by country.

### 7.2 H1: block permutation
Each country's whole outcome series is reassigned to another country, keeping its timing. Repeat 1,000 times.
- Statistic: the deviance reduction from adding the Π block to the fixed-effects-only model.
- p = the share of permutations whose reduction is at least the observed one.

### 7.3 H2: dimensional homogeneity
The unconstrained model regresses ln M on the logs of the raw variables (Np, H, Y, K+1, A_L, A_R, Y_ref, D_ref), with year effects.
Dimensional homogeneity implies D·γ = d_M, where D is the dimension matrix and d_M is the dimension of M. This is a Wald test with one degree of freedom per base dimension involved.

### 7.4 H3: forecasting
- Fit on 2008–2019 and predict the Π0 rate for 2020–2023.
- Metric: median over countries of |ln(pred + c) − ln(obs + c)|, with c = the 5th percentile of the positive Π0 values.
- Baselines: persistence (the 2019 value) and the country mean over 2008–2019.
- Rolling-origin check: train to 2015, test 2016–2019.

### 7.5 Between-country model
Country means of all groups. PPML with region fixed effects. Includes Π5 and Π7.

## 8. Robustness (reported whatever the result)
- All countries, with no population threshold.
- UNHCR outcome (R).
- Three-year averages.
- Excluding the top 1% of displacement events.
- Treating absent IDMC records as missing.
- Drop one region at a time.
- Π3 from FAOSTAT instead of CRU, for H1 and H4 (D-019): CRU and FAOSTAT disagree (r < 0.7) in 27 countries, mostly where weather stations are sparse.

## 9. Decision gate (from results to paper)
| H1 | H3 | H2 | Paper and target |
|---|---|---|---|
| fail | — | — | Methods paper: "When do dimensionless indices carry signal?" → Environmental Research Communications |
| pass | fail | — | Screening framework with stated limits → Environmental Modelling & Software |
| pass | pass | fail | Predictive framework without a "law" claim → Earth's Future |
| pass | pass | pass | Tested dimensional relationship → Earth's Future or Communications Earth & Environment |

## 10. What this plan deliberately does not do
- No clustering-based "validation" by selected examples.
- No interpretation of coefficient sign changes without confidence intervals.
- No claim of "self-similarity" unless a data-collapse test is added to this plan first.
