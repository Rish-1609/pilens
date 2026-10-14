# Parameter specification: old method vs rebuild

How each parameter was built in the March 2026 draft (SI Tables S1–S2 and Section S3), what was wrong with it, and how the rebuild computes it. The code that implements each line is named in the last column.

## 1. Input variables

| Concept | Old draft: source, unit, processing | Problem | Rebuild: source, unit | Code |
|---|---|---|---|---|
| Migration (outcome) | UNHCR refugee + asylum inflow and outflow, combined into a net figure; min–max 0–10 | Mostly conflict-driven; a net figure can be negative and breaks the log | **IDMC** new disaster displacements, persons/yr (unrounded 'Raw' event figures summed by country-year, D-010). UNHCR refugees + asylum seekers by origin kept as a secondary outcome | `build.idmc`, `build.unhcr` |
| Population density D | MacroTrends; min–max | A secondary source, and scaling destroys units | WDI published density `EN.POP.DNST`, persons/km². Np ÷ land area understates GEO and MDA by 15–20% because population excludes breakaway regions and land area includes them (D-015) | `pigroups.compute`, `qa` check 3 |
| GDP per capita E | Listed as "Fragile States Index" (the FSI does not publish GDP); min–max | Wrong source; units lost | WDI `NY.GDP.PCAP.PP.KD`, constant 2021 international $/person/yr | `config.WDI["Y"]` |
| Political / conflict P | FSI political indicator, a 0–10 expert score, assigned the dimensions N T⁻¹ C⁻¹ | An expert score has no units, so its "dimensions" were invented, and π4 was not dimensionless | **UCDP** deaths from organized violence (best estimate), deaths/yr. Column `cumulative_total_deaths_in_orgvio_best` is the within-year sum of state-based, non-state and one-sided deaths (checked on the file) | `build.ucdp` |
| Social resources S | FSI social indicator (0–10 score), assigned C T⁻¹ N⁻² L⁻² | Same problem; π3 was not dimensionless | WDI health spending as a share of GDP (both in the same current prices) | `config.WDI["HY"]` |
| Carbon intensity G | OWID CO₂ per $ GDP; min–max | No causal path to displacement; its large exponent came from a shared term | Same OWID series, **robustness only** | `build.build_panel` (G_co2_per_gdp) |
| Precipitation W | OWID average annual precipitation; min–max; quoted elsewhere in raw mm | Long-term average with no year-to-year variation; inconsistent units in the SI | **CCKP CRU TS** annual precipitation, mm/yr, as a standardized anomaly vs 1991–2020 | `build.cckp_precip` |
| Temperature F | IMF annual temperature change, °C, assigned the dimension L T⁻¹ ("habitable zone shift") | Temperature is not a length rate; the km-per-°C conversion was never stated | **CCKP CRU TS** annual mean temperature (°C), standardized vs 1991–2020 (D-013). FAOSTAT ΔT kept as a cross-check: it lacks baseline years for 12 countries, including YEM, SGP, SDN, SSD | `build.cckp_temperature` |
| Sea-level rise O | IMF annual sea-level rise, mm; π5 = O/√A | Close to uniform across countries, so π5 measured land size; undefined for landlocked countries | WDI share of population below 5 m elevation (CIESIN LECZ v3, 2015) | `config.WDI["LECZ"]` |
| Land A | FAO arable land; min–max | Units lost | WDI arable land (FAO), hectares → km²; WDI total land area, km². Montenegro 2008–2012 set missing: series break in 2013 (D-012) | `build.build_panel` |

## 2. Groups

| Group | Formula | Unit check | Notes |
|---|---|---|---|
| Π0 displacement rate | M·τ / Np | (N T⁻¹)(T)/N = 1 | τ = 1 year. IDMC counts movements, so one person displaced twice counts twice |
| Π1 health share | H / Y | same price base, = 1 | Uses WDI's published ratio. Dividing current-PPP spending by constant-2021 GDP would drift with inflation (D-007) |
| Π2 conflict rate | K·τ / Np | (N T⁻¹)(T)/N = 1 | UCDP codes a country-year with no recorded violence as 0; economies UCDP does not list stay missing (D-011) |
| Π3 temperature anomaly | (Tc − T̄) / σ_T | Θ/Θ = 1 | CRU TS, baseline 1991–2020 (D-008, D-013) |
| Π4 precipitation anomaly | (Pr − P̄) / σ_P | (L T⁻¹)/(L T⁻¹) = 1 | Same baseline. Negative = drier, positive = wetter |
| Π5 coastal exposure | LECZ share | already a share | Time-invariant, so it is identified only in the between-country model |
| Π6 relative income | Y / Y_ref | = 1 | Y_ref = world median in that year |
| Π7 arable share | A_R / A_L | L²/L² = 1 | |
| Π8 relative density | D / D_ref | (N L⁻²)/(N L⁻²) = 1 | D = WDI published density; D_ref = its world median in that year |

The code (`pigroups.derive_and_verify`) re-checks every row of this table against the dimension matrix each time it runs.

## 3. Logic checks the pipeline runs automatically

1. Indicator codes in each WDI file match the requested code (Step 1b).
2. UCDP file is version 26.1 and has the codebook columns (Step 1b).
3. Computed density (Np ÷ land area) is within 5% of WDI's published density (Step 3). Differences are definitional (GEO, MDA), which is why Π8 uses the published density.
4. WDI population is within 5% of OWID population (Step 3). This catches errors such as the 100× Italy value in the 2023 sheet.
5. No country-year has displacement above 50% of its population (Step 3). This catches errors such as the UK 2016 value.
6. No 10× year-on-year jump in a slowly changing variable (Step 3).
7. Each climate baseline has at least 25 of its 30 years (Step 3).
8. CRU series filled with climatology (the same value in 4+ consecutive years) get a missing anomaly (Step 2, D-014, D-018).
9. CRU and FAOSTAT temperature agree within each country (correlation ≥ 0.7, Step 3).
10. Country codes that do not convert to a valid ISO3 are blanked and listed, never kept as raw codes (Step 2).

## 4. Results of the first run on real data (2026-10-13)

- Coverage of the Π groups in the 170-country sample: 96–99% of country-years for every group, including 2023 health spending.
- Fixed after the run: UCDP column names (v26.1 file), country-code conversion (D-011), IDMC column choice (D-010), Montenegro series break (D-012), density (D-015), climatology fill (D-014).
- Added: CRU temperature (D-013). With it, Π3 covers 98.8% of sample country-years (FAOSTAT alone: 92.2%). CRU and FAOSTAT anomalies correlate 0.81; they disagree (r < 0.7) in 27 countries, mostly tropical Africa, so a FAOSTAT robustness check is pre-registered (D-019).
- Climatology-fill rule amended to 4+ consecutive identical years (D-018): flags SLB and CPV precipitation and SLB temperature in the sample.
- Step 3 passes with no blocking problems; Step 4 confirms all 8 groups are dimensionless and span the null space.

## 5. Datasets not used, and why

- **Fragile States Index**: the old draft's P, S and E came from it, but expert scores cannot carry physical dimensions. It could be added later as a robustness control, not as a Pi variable.
- **EM-DAT disaster events**: useful for hazard counts, but it requires a login and has licence limits on redistribution. Optional for robustness.
- **IMF Climate Change Indicators (sea level, temperature)**: replaced by LECZ and CRU TS (with FAOSTAT as a cross-check), which are more widely used and cover every country.
