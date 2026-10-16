# SMS Food-Shortage Early Warning — Market-Price Module & Engine Layer

Status as of **2026-10-16**. Covers everything built so far: price and IPC ingestion, the shared data
contract, the storage layer, the risk engine (features → fusion → SMS → back-test), the test suite, and
what other teams need to know to plug in their data sources and the SMS/dispatch layer.

---

## 1. Important things first

1. **One data contract connects everything:** the `signal_observation` row (§4). Any new data source
   (satellite, weather, conflict, announcements, crowd reports) only has to write rows in this format to the
   `signals` collection. Nothing downstream needs to change, apart from the two lookup tables in §9.1.
2. **Only one source family exists today: `market_prices`.** Fusion needs flags from **two different
   source families** to reach Level 2 (Warning). Until another family is ingested, the engine stops at
   Level 1 (Watch). That is why the real-data back-test shows no Warning lead time.
3. **Every signal needs history.** A value only becomes a feature when the same calendar month exists for at
   least **3 of the previous 5 years** of that same series. A new source with only recent data produces no
   features and therefore no flags.
4. **Point-in-time is built in.** Every `--as-of` run assumes a **1-month publication lag**
   (`period_end + 1 month ≤ as_of`), and every baseline only looks backwards. New sources must give a
   truthful `period_end`.
5. **`ipc_phase` is ground truth only.** It is never turned into a feature and never used by fusion. It is
   only used to score the back-test.
6. **Missing data is never filled with zero.** A missing value stays absent, and confidence drops instead.
   A `value = 0` in the store is always a genuinely observed 0 (e.g. a 0% change).
7. **All writes are idempotent upserts** on fixed keys (§5). Re-running any script never duplicates rows.
8. **Zimbabwe's data is thin for 2019–2023**, and its IPC data ends in 2020. Kenya is the country with
   usable data on both sides (prices and IPC through 2026).

---

## 2. Architecture and data flow

```
        EXTERNAL SOURCES (no API keys)                       OTHER TEAMS (to integrate)
  WFP HDX food prices (KE, ZW)   FEWS NET FDW (KE)          satellite EO, weather, conflict,
  WFP Global Market Monitor      IPC country data (HDX)     announcements, crowd reports
            |                          |                                 |
   scripts/ingest_prices.py    scripts/ingest_ipc.py          (write signal_observation rows)
            |                          |                                 |
            +-------------+------------+---------------------------------+
                          v
     signals  (TinyDB data/out/signal_observation.json  +  SQLite data/out/signal_observation.sqlite)
                          |
            scripts/engineer_features.py      z-score vs same-month baseline, direction, flags
                          v
     features
                          |
            scripts/fuse_risk.py              levels L0-L3, hysteresis, p_shortage, confidence
                          v
     risk_assessment                       (+ data/out/hysteresis_state.json)
                          |
            scripts/compose_sms.py            Claude / OpenAI / mock, validated <=155-char SMS
                          v
     alerts   ---> (future) dispatch_alerts.py  [SMS team]

     scripts/backtest.py      replays features + fusion weekly vs IPC ground truth -> data/out/backtest_*.json
     scripts/show_results.py  read-only console summary of everything above
```

Shared helpers: `scripts/db.py` handles all TinyDB access. `scripts/ingest_prices.py` also exports reusable
utilities (`norm`, `slug`, `fetch`, `ckan_resource`, `load_units`, `same_month_stats`, `pit_filter`,
`now_utc`, `COUNTRIES`).

---

## 3. Quick start

```bash
pip install requests pandas geopandas shapely scipy tinydb anthropic pytest

# 1. Ingest (writes raw/clean files, SQLite and the TinyDB `signals` collection)
python scripts/ingest_prices.py --country KE
python scripts/ingest_prices.py --country ZW
python scripts/ingest_ipc.py   --country KE
python scripts/ingest_ipc.py   --country ZW

# 2. Engine
python scripts/engineer_features.py --country KE
python scripts/fuse_risk.py         --country KE            # add --as-of YYYY-MM-DD for a past date
python scripts/compose_sms.py       --country KE --audience all
python scripts/backtest.py          --country KE --from 2022-01-01 --to 2026-06-30

# 3. Look at the results
python scripts/show_results.py [--country KE]

# Tests (synthetic data, no network, no LLM calls, ~30 s)
python -m pytest tests/ -v --tb=short
```

Same-day re-runs reuse the downloaded raw files; add `--refresh` to the ingest scripts to download again.

---

## 4. The data contract: `signal_observation`

Every source, existing or future, produces rows with exactly these fields:

| Field | Type | Meaning |
|---|---|---|
| `unit_id` | str | Admin unit, e.g. `KE-ADM1-turkana`, `ZW-ADM2-chiredzi`, `KE-ADM0-kenya` (§6.4) |
| `source_id` | str | Producer, e.g. `wfp_hdx_ke`, `fewsnet_ke`, `wfp_gmm`, `ipc_phase`. **Its prefix decides the source family** (§9.1) |
| `signal_type` | str | What is measured (catalogue below) |
| `value` | float | The measurement. Never NaN; never a zero standing in for "missing" |
| `unit_of_measure` | str | Free text, e.g. `pct_vs_5y_same_month_median_usd_retail` |
| `period_start` | `YYYY-MM-DD` | Start of the period. **Monthly signals use the 1st of the month** (required for same-month baselines) |
| `period_end` | `YYYY-MM-DD` | End of the period. Drives the point-in-time filter |
| `ingested_at` | ISO-8601 UTC | When the row was written |
| `confidence` | float 0–1 | Source-level quality (freshness, location match, coverage, history) |
| `raw_ref` | str | Provenance: `URL|last_modified=...|fetched_at=...` |

**Unique key:** `(unit_id, source_id, signal_type, period_start)`.

### 4.1 Signal catalogue (what exists today)

| signal_type | Producer | unit_of_measure | Meaning | Risk direction |
|---|---|---|---|---|
| `price_anom_real` | wfp_hdx_*, fewsnet_ke | `pct_vs_5y_same_month_median_usd_<retail\|wholesale>` | Median over food items of (USD price − 5-yr same-month median) / median × 100. **The price signal meant for fusion** | +1 (higher = worse) |
| `price_anom_nominal` | same | `..._local_...` | Same calculation in local currency (kept for reference; ZW: identical to real, since USD is the local currency) | +1 |
| `price_velocity_anom` | same | `abs_diff_3m_velocity_vs_baseline` | 3-month mean of monthly % change in USD price, minus its same-month baseline | +1 |
| `price_dispersion_anom` | same | `abs_diff_iqr_over_median_vs_baseline` | Cross-market IQR/median of USD price, minus its same-month baseline (needs ≥3 markets per item-month) | +1 |
| `market_coverage_drop` | same | `yoy_change_active_market_count` | Distinct markets reporting now minus 12 months earlier (negative = worse) | −1 |
| `data_freshness_share` | same | `share_of_active_series` | Share of the unit's food series (seen in the last 12 months) reporting this month | −1 |
| `basket_cost_change` | wfp_gmm | `pct_change_3m_basket_cost` | GMM `TotImpactQuarterlyChange`: 3-month % change in food-basket cost | +1 |
| `ipc_phase` | ipc_phase | `ipc_phase` | IPC area phase 1–5, current validity period. **Ground truth only** | 0 (never fused) |

Signal types already reserved in the engine for other teams (direction already defined, see §9.1):
`ndvi_z`, `evi_z`, `spi_1m`, `spi_3m`, `soil_moist_z` (−1); `fx_devaluation_3m`, `fuel_price_anom`,
`food_cpi_yoy`, `conflict_events_30d`, `announcement_severity`, `crowd_crop_status` (+1).

---

## 5. Storage

| Store | Path | Written by | Notes |
|---|---|---|---|
| TinyDB (main engine store) | `data/out/signal_observation.json` | all scripts via `scripts/db.py` | One JSON file, 4 collections |
| SQLite (mirror of signals) | `data/out/signal_observation.sqlite` | ingest scripts | Table `signal_observation`, PK = signal key. Same row count as TinyDB `signals` (15,794) |
| SQLite back-test snapshots | `data/out/signal_observation_asof_<date>.sqlite` | ingest scripts with `--as-of` | Kept separate so live data is never mixed with back-test data |
| Hysteresis state | `data/out/hysteresis_state.json` (live), `hysteresis_state_asof_<date>.json` (back-dated runs) | `fuse_risk.py` | Plain JSON for easy inspection |

### 5.1 TinyDB collections

| Collection | Key (idempotent upsert) | Written by | Read by |
|---|---|---|---|
| `signals` | `unit_id, source_id, signal_type, period_start` | ingest_prices, ingest_ipc (other teams' sources go here too) | engineer_features, backtest |
| `features` | `unit_id, signal_type, period_start` | engineer_features | fuse_risk |
| `risk_assessment` | `unit_id, run_at` | fuse_risk | compose_sms, show_results |
| `alerts` | `assessment_id, audience` | compose_sms | dispatch (future), show_results |

`db.py` API: `upsert_signals / query_signals(country, signal_type, source_id, period_end_le)`,
`upsert_features / query_features(country, period_start_le)`, `upsert_risk / query_risk(country, level_ge)`,
`upsert_alerts / query_alerts(assessment_id, audience)`, `count_by(table, field)`,
`load_hysteresis / save_hysteresis`.
- Upserts are done in bulk: one update pass plus one insert, because TinyDB rewrites the whole file on each
  write.
- Upserts keep TinyDB document IDs stable. The risk document ID is used as `alerts.assessment_id`.
- Query results carry the document ID as `_doc_id`.
- The country filter is a `unit_id` prefix match (`KE-`, `ZW-`).

### 5.2 Other files

```
data/raw/<CC>_<source>_<fetch_date>.csv (+ .meta.json with URL, Last-Modified, fetch time)   unchanged downloads
data/clean/<CC>_<source>_<YYYY-MM>_<YYYY-MM>.csv                                            cleaned market-level rows
data/geo/<ISO3>_<ADM>.geojson                                                               cached boundaries
data/out/signal_observation_prices_<CC>_<run_at>.csv / signal_observation_ipc_<CC>_<run_at>.csv  per-run exports
data/out/freshness_<run_at>.json / freshness_ipc_<CC>_<run_at>.json                         per-run source report + smoke tests
data/out/backtest_<CC>_<run_at>.json                                                        back-test reports
```

---

## 6. Ingestion: market prices (`scripts/ingest_prices.py`)

### 6.1 Sources

| source_id | Source | Countries | Freshness (2026-10-16) | Notes |
|---|---|---|---|---|
| `wfp_hdx_ke`, `wfp_hdx_zw` | WFP Food Prices on HDX (CKAN API) | KE, ZW | KE fresh (2026-09); **ZW stale (2026-05) → confidence × 0.5** | ZW: only USD rows kept (`usdprice`), ZWL rows dropped |
| `fewsnet_ke` | FEWS NET FDW API (`fdw.fews.net/api/marketpricefacts`) | KE | fresh (2026-08) | Nairobi only, 6 products (2 are fuel). No USD column, so USD uses the WFP-implied monthly rate |
| `wfp_gmm` | WFP Global Market Monitor (HDX) | KE, ZW | ZW to 2026-06; **KE only 2020-10 → 2021-05 (× 0.25)** | Pre-computed % changes; no outlier filter |
| — | FEWS NET ZW | — | skipped (HTTP 500) | Recorded as `skipped` in the freshness report; the FEWS fetch is wrapped so failures degrade to WFP only |
| — | WFP DataBridges | — | not used (unverified credentials) | |
| — | Morocco | — | **no coverage in any of the trusted sources** | Checked: no WFP HDX dataset, 0 WFP markets, FEWS returns an empty file, no GMM rows |

### 6.2 Pipeline, in order

1. **Fetch:** downloads are saved unchanged in `data/raw` with a `.meta.json` file. `raw_ref` is
   `URL|last_modified|fetched_at`.
2. **Point-in-time filter** (only with `--as-of`): rows whose `period_end + 1 month > as_of` are removed.
3. **Validation:** non-positive or NaN prices are dropped (never zero-filled). For WFP, when an "actual" and
   an "aggregate" row cover the same market/item/month, the actual one wins.
4. **Hampel outlier filter, on log price, per `commodity × unit × currency × price_type` group (never
   globally):**
   - The window is trailing: all markets in the group over the last 6 months, so it never looks forward.
   - A price is flagged when it is more than 3 × scale from the window median, where scale = 1.4826 × MAD.
   - The scale has a floor of 0.10, so a sticky price only gets flagged if it moves by more than ~35%.
   - At least 5 values must be in the window.
   - ZW: 1,222 flagged with currencies split, against the earlier 6,149 false positives.
5. **ZW currency:** ZWL rows are dropped after the outlier filter (demo decision). Only `usdprice` is used.
6. **Units:** `90 KG`, `500_ml`, `200 G` etc. are converted to price per kg or per litre, in local currency
   and USD. Units that can't be converted (Unit/Head/Bunch) keep their own unit.
7. **Commodity names** are standardised by a regex table (`maize_grain`, `maize_flour`, `beans`,
   `vegetable_oil`, …). Retail and wholesale are kept apart and **never pooled**. When a unit-month has
   both, retail is preferred for the signal.
8. **Location** (§6.4).
9. **Signals:** monthly median across the unit's markets per series (`unit × commodity × price_type × uom`),
   then:
   - **Anomaly:** (price − median of the same month over the previous 5 years) / median × 100. It needs
     ≥3 of those 5 years, otherwise it is absent.
   - **Aggregation:** one value per unit-month is the median across food series. Fuel and non-food items are
     excluded.
   - The derived signals of §4.1.
10. **Write:** SQLite upsert, TinyDB `signals` upsert (live runs only), CSV export, freshness JSON with smoke
    tests.

### 6.3 Confidence rules (source level)

- **Freshness factor**, based on the age of the source's latest data relative to the run date:
  - ≤ 4 months (1-month lag + 3): ×1.0
  - ≤ 12 months: ×0.5
  - older: ×0.25
- **Location factor:**

  | Match method | Factor |
  |---|---|
  | point in polygon | 1.0 |
  | nearest polygon within 10 km | 0.9 |
  | name match or gazetteer | 0.8 |

- **Per signal type:**

  | Signal | Confidence |
  |---|---|
  | `price_anom_*` | fresh × loc × min(1, baseline_years/5) × (0.5 + 0.5 × freshness_share) |
  | `price_velocity_anom` | fresh × loc × min(1, baseline_years/5) |
  | `price_dispersion_anom` | fresh × loc × min(1, n_markets/5) |
  | `market_coverage_drop` | fresh × (1.0 if \|change\| ≥ 5 else 0.5) |
  | `data_freshness_share` | fresh × loc |
  | `basket_cost_change` | fresh × (0.7 when a region-level value is copied to its districts, 1.0 national) |

### 6.4 Admin units and location matching

- **Boundaries:** geoBoundaries gbOpen, simplified, cached in `data/geo`.
  - **KE:** ADM1, 47 counties.
  - **ZW:** ADM2, 91 districts (ADM1 provinces are used only to copy GMM values down to districts).
- **unit_id format:** `<CC>-<ADM level>-<slug(shapeName)>`; national is `<CC>-ADM0-<country>`.
- **Location matching**, tried in order (markets that still don't match are dropped and listed in the
  freshness report, never filled):
  1. Point in polygon.
  2. Nearest polygon within 10 km.
  3. Gazetteer: `ZW_GAZETTEER`, 8 ZW markets WFP publishes without coordinates. **These were assigned by
     hand and are worth checking.**
  4. Name match on admin2, then admin1, then a name in brackets, then the market name.
- **GMM Kenya** reports by pre-2013 province. A static province → county map (`KE_PROVINCES`) copies those
  values to the counties.

### 6.5 Smoke tests (printed every run; exit code ≠ 0 on failure)

- **Zero hard failures:** the FEWS ZW skip doesn't count.
- **No values from missing data:** no NaN values; any zeros are real 0% changes.
- **`price_anom_real` populated.**
- **Derived signals populated:** a WARN, not a failure, when dispersion is empty. ZW has no district with
  ≥3 markets per item-month over ≥3 years.
- **Idempotent upsert:** row counts before and after.
- **ZW only:** the outlier count is below the earlier 6,149 false positives.

---

## 7. Ingestion: IPC ground truth (`scripts/ingest_ipc.py`)

- **Dataset:** HDX `<country>-acute-food-insecurity-country-data`, published by IPC. It falls back to
  `ipc-country-data`, which is legacy 2017–2023 XLSX and recorded as unusable. The slugs in the original
  spec (`ipc-zimbabwe-food-security-analysis`) don't exist.
- **Output:** `source_id = signal_type = "ipc_phase"`, value = phase 1–5, `period_start/end` = the IPC
  validity period, confidence 1.0.
- **Only "current" periods are kept.** Projections are forecasts, not ground truth, so they are skipped.
- **Phase:** the CSVs publish population per phase, not the classified area phase. The phase is derived with
  the IPC area rule: the worst phase that ≥20% of the population is in or above, using rounded
  percentages. For the latest analysis, the official `overall_phase` from the GeoJSON is used instead.
  Derived and official phases agree on all 83 cases where both exist (23 KE, 60 ZW).
- **Area matching:**
  - Exact name, then the `IPC_GAZETTEER`, then names with "county" removed.
  - Sub-areas (e.g. "Turkana north") are combined into their county by adding up the populations.
  - Refugee camps (Dadaab, Kakuma, Kalobeyei) have no admin unit. They are listed under
    `unmatched_districts` in the freshness report, not silently dropped.
- **Coverage:**
  - **KE:** 276 rows, 23 counties, analyses through 2026-07.
  - **ZW:** 240 rows, 60 districts, **last analysis Oct 2020** (periods up to 2020-12).

---

## 8. Engine layer

### 8.1 Features (`scripts/engineer_features.py`)

Steps, for each `(unit_id, signal_type, source_id)` series:
1. **Baseline** = median of the same calendar month over the previous 5 years. At least 3 values are
   required; otherwise there is **no feature**.
2. **MAD** of those same-month values.
3. **`z_score`** = (value − baseline) / max(MAD × 1.4826, **0.10**). The floor stops sticky signals from
   giving huge z-scores.
4. **`risk_score`** = z × `DIRECTION[signal_type]`, so positive always means bad.
5. **`threshold_flag`** = 1 when `risk_score > 1.5`.
6. **`source_family`** comes from the `source_id` prefix, falling back to the `signal_type` name (§9.1).
7. When several sources report the same `(unit, signal_type, period)`, the most alarming one is kept
   (max risk_score), and `n_source_series` records how many there were.

Other rules:
- `ipc_phase` is excluded (`GROUND_TRUTH`).
- **Point-in-time:** with `--as-of`, only signals with `period_end ≤ as_of − 1 month` are used.
- A feature never changes once its period is published, because baselines only look backwards.
  `backtest.py` relies on this by computing series features once (`series_features` + `collapse_sources`);
  this was checked to give identical results to recomputing at every tick.

Feature document fields: `unit_id, signal_type, period_start, period_end, value, baseline, mad,
baseline_years, z_score, risk_score, threshold_flag, source_family, source_id, n_source_series,
signal_confidence, feature_version`.

### 8.2 Fusion (`scripts/fuse_risk.py`, `model_version = "rules-v1"`)

**Raw level** for each `(unit_id, period_start)`. IPC and signals with direction 0 are ignored.

| Level | Rule |
|---|---|
| L0 Normal | no flagged feature |
| L1 Watch | ≥1 distinct source family flagged |
| L2 Warning | ≥2 distinct source families flagged |
| L3 Emergency | L2 **and** an `announcement_severity` feature with value ≥ 3 in the same unit-period |

**Hysteresis** (per unit, replayed over its periods up to `--as-of`):
- A new **higher** level must be seen on **2 consecutive periods** before it applies.
- A new **lower** level must be seen on **3 consecutive periods** before it applies.
- A gap of **> 90 days** between evidence periods resets the state.
- One "run" means one monthly period of evidence, not one script execution. So re-running the script never
  double-counts a period, and weekly back-test ticks don't confirm a level from the same monthly data.
- Live runs write `hysteresis_state.json`; back-dated runs write their own `_asof_` file.

**p_shortage** (n = number of families agreeing):

| Level | p_shortage |
|---|---|
| L0 | 0.05 |
| L1 | 0.20 + 0.10 × (n − 1), capped at 0.45 |
| L2 | 0.50 + 0.10 × (n − 2), capped at 0.75 |
| L3 | 0.80 |

**Confidence:** `min(n/4, 1) × min(baseline_years/5, 1) × freshness_share`, clipped to [0, 1].
- `baseline_years` is the mean over flagged features, or over all features if none are flagged.
- A missing freshness_share counts as 0.5.
- **This means L0 assessments always have confidence 0.**

**Drivers:** `drivers_json` holds the top 3 features by risk_score, as `{signal_type, value, z_score}`.

**Output:** one document per `(unit_id, run_at)`, for the unit's latest period, and **only when that period
is ≤ 6 months older than the reference date** (as-of, or today). Fields: `unit_id, run_at, period_start,
horizon_weeks (8), p_shortage, level, raw_level, confidence, drivers_json, n_sources_agreeing, n_total,
model_version, as_of`.
- `run_at` is the current UTC time, or `<as_of>T120000Z` for back-dated runs.
- Every live run adds new documents.

### 8.3 SMS composition (`scripts/compose_sms.py`)

**Which assessments get SMS:** for each unit, only the **latest** risk assessment with level ≥ 1. Pairs of
(assessment, audience) that already have an alert are skipped.

**Audiences and allowed actions** (`CAPABILITIES`):

| Audience | Allowed actions |
|---|---|
| farmer | store_grain_2m, sell_livestock_early, plant_drought_tolerant, avoid_selling_below_market, reduce_grain_sales |
| government | release_strategic_reserves, adjust_subsidies, declare_disaster, coordinate_response, request_international_aid |
| ngo | trigger_anticipatory_cash, pre_position_food, dispatch_field_teams, target_households, support_cooperatives |

Each code has a plain-language phrase in `ACTION_TEXT`. Language defaults to `en` for KE and ZW.

**Provider, first match wins:**
1. `OPENAI_API_KEY` set: `OpenAI().chat.completions.create(model="gpt-4o-mini", JSON mode)`. This is the
   openai ≥ 1.0 API. **Never tested against the real API**, because no key or package is installed.
2. `ANTHROPIC_API_KEY` set: `client.messages.create(model=SMS_MODEL or "claude-haiku-5-5",
   output_config={"effort": "low", "format": json_schema})`. Used live.
3. Neither set: `[MOCK] ...` text with `status = "mocked"`.

**Validation** (an SMS failing any check is rejected):
- ≤155 characters.
- GSM-7 characters only.
- Starts with WATCH / WARNING / EMERGENCY.
- Farmer SMS end with `Reply 1 ok 2 poor 3 failed. STOP to quit`.
- No raw capability codes (`_`).
- No `%`, probabilities or forecasts.

A rejected answer gets **one stricter retry**. After that, or after an API error, the SMS comes from a
hardcoded template per `(level, audience)`.

**Alert document fields:** `assessment_id, unit_id, level, audience, template_text, char_count, language,
status, approved_by, generator, generation_notes, run_at, created_at, expires_at`.

| Field | Values |
|---|---|
| `status` | `approved` for L1 (`approved_by = "auto:level1"`); `drafted` for L2+ (needs human approval, `approved_by = null`); `mocked` when no API key |
| `generator` | `anthropic:first`, `anthropic:retry`, `openai:...`, `template_fallback`, or `mock` |
| `expires_at` | `created_at` + horizon (8 weeks) |

### 8.4 Back-test (`scripts/backtest.py`)

`run_backtest(country, from, to, out_dir)` works like this:
1. Takes **weekly ticks** from `--from` to `--to`.
2. At each tick, recomputes features and fusion in memory with `as_of = tick`. Nothing is written to the
   live collections or the live hysteresis file.
3. Records each unit's level per tick.

It then scores those levels against IPC:
- **Transition:** the first IPC current period starting inside the window with phase ≥ 3, whose previous IPC
  period for that unit was < 3. Units with no earlier analysis are not counted.
- **hit_rate** = transitions where the first warning came on or before the transition date / all
  transitions.
- **false_alarm_rate:** among warned units whose next 8 weeks are covered by IPC, the share with no
  Phase 3+ period overlapping those 8 weeks.
- **median_weeks_gained** = median of (transition − first warning) over hits.
- **Persistence baseline:** the same metrics, using the engine's level from 4 weeks earlier.
- Reported for **level ≥ 2** (spec) and **level ≥ 1**.
- Rates are `None` (undefined) when their denominator is 0. They are never reported as 0.
- `notes` in the report explain empty results.

---

## 9. Integration guide

### 9.1 Adding a new data source (EO, weather, conflict, announcements, crowd)

1. **Write `signal_observation` rows** (§4) to the `signals` collection with `db.upsert_signals(rows)`.
   Optionally mirror them to SQLite with `ingest_prices.upsert(path, df)`.
2. **Pick a `source_id` prefix that maps to your family.** The table is `SOURCE_FAMILY` in
   `engineer_features.py`:

   | Family | Prefixes |
   |---|---|
   | `market_prices` | `wfp_hdx`, `fewsnet`, `wfp_gmm` |
   | `ipc` | `ipc` |
   | `eo` | `ndvi`, `evi`, `spi`, `chirps` |
   | `weather` | `soil_moist` |
   | `text` | `announcement` |
   | `crowd` | `crowd` |
   | `conflict` | `acled`, `conflict` |

   - If the `source_id` prefix doesn't match, the `signal_type` name is tried next. Anything else becomes
     `other`.
   - **SPI is `eo`, not `weather`, in the current table.** Two signals in the same family count as one for
     agreement.
   - A new family must be added to this table.
3. **Add your `signal_type` to `DIRECTION`** in `engineer_features.py` with +1 or −1. Unknown types get
   risk 0 and a warning.
4. **Use monthly periods** (`period_start` = the 1st of the month) and a truthful `period_end`. For
   point-event signals such as announcements, put the event in its month, so it lines up with the other
   signals of that unit-period. Otherwise it forms its own period and never combines with the rest.
5. **Provide history:** at least 3 earlier years of the same months, or the signal produces no features.
   Already-standardised z-scores are fine. The engine computes a z-score of your values against their own
   history.
6. **Use the existing `unit_id`s** (`KE-ADM1-*`, `ZW-ADM2-*`). `ingest_prices.load_units(cc, COUNTRIES[cc])`
   returns the polygons, and `assign_units` / `norm` / `slug` are reusable for matching.
7. **Never write rows for missing data.** Put quality into `confidence`. Make `raw_ref` traceable.
8. **For L3:** the rule needs an `announcement_severity` *feature* with value ≥ 3 in the same unit-period as
   two other flagged families. That means it also needs history, e.g. earlier years at severity 0. Then two
   consecutive L3 periods are needed because of hysteresis.

### 9.2 For the SMS / dispatch team

- Read the `alerts` collection with `db.query_alerts()`.
- **`status`:** send `approved`; route `drafted` (L2/L3) to a human reviewer and set `approved_by`; never
  send `mocked`.
- Respect `expires_at`. `assessment_id` points at the `risk_assessment` document ID.
- The farmer reply footer (`Reply 1 ok 2 poor 3 failed`) is meant to collect crowd crop-status feedback.
  It fits the reserved `crowd_crop_status` signal type, family `crowd`.
- Wording rules, audiences and allowed actions are constants at the top of `compose_sms.py`.

### 9.3 For a bigger orchestration system

- Every script is a CLI with `--country` (and `--as-of` where it applies), and also importable:

  | Function | Script |
  |---|---|
  | `compute_features` | engineer_features |
  | `assess`, `raw_levels`, `apply_hysteresis`, `p_shortage`, `confidence` | fuse_risk |
  | `Composer`, `build_alert`, `validate` | compose_sms |
  | `run_backtest` | backtest |

- Run order: ingest → engineer_features → fuse_risk → compose_sms. Each step is idempotent, so a scheduler
  can safely retry.
- Paths are module constants (`db.DB_PATH`, `db.HYSTERESIS_PATH`, `ingest_prices.OUT`); the tests override
  them.

---

## 10. Tests (`tests/`)

- **92 passing** (`python -m pytest tests/ -v`, ~30 s, every test < 5 s).
- They use a synthetic country `TEST` with districts D1–D5 and monthly data for 2019–2024 (history years are
  needed for baselines). They use a temporary TinyDB file and output folder.
- **API keys are removed and the network is blocked, so no real API calls are possible.**

| File | Covers |
|---|---|
| `conftest.py` | Synthetic world: D1 = price shock + drought + announcement (→ L3), D2 = price shock only (→ L1), D3–D5 normal; IPC phase 3 for D1/D2 from 2024-10 |
| `test_engineer_features.py` | z-score maths, direction, the 1.5 threshold, ≥3-year baseline, no look-ahead, IPC exclusion, the 0.10 floor |
| `test_fuse_risk.py` | L0–L3 rules, IPC ignored, escalation / de-escalation / 90-day reset, state persistence, p_shortage, confidence, drivers |
| `test_compose_sms.py` | mock mode, OpenAI and Anthropic calls with fake clients, validation, retry, template fallback, farmer footer, allowed actions, statuses |
| `test_backtest.py` | metrics present and in range, weeks gained > 0, undefined rates with no transitions, JSON output |
| `test_end_to_end.py` | full pipeline, alerts written, idempotent re-run, the real DB untouched |

---

## 11. Current results (real data, 2026-10-16)

**TinyDB contents:** `signals` 15,794 · `features` 6,965 · `risk_assessment` 25 · `alerts` 6.

**Signal rows per country:**

| signal_type | KE | ZW |
|---|---|---|
| price_anom_real | 1,779 | 468 |
| price_anom_nominal | 1,791 | 468 |
| price_velocity_anom | 1,500 | 316 |
| price_dispersion_anom | 36 | 0 |
| market_coverage_drop | 2,194 | 956 |
| data_freshness_share | 2,662 | 1,652 |
| basket_cost_change | 165 | 1,291 |
| ipc_phase | 276 | 240 |

**Latest fusion (KE):** 6 counties have recent enough evidence. Kisumu is at L1 Watch; the other 5 are at
L0. ZW features end in 2018 (except one basket-cost row), so ZW has no current assessment.

**Alerts:** 3 Claude-written SMS for Kisumu (farmer / government / ngo, status `approved`). They all passed
validation.

**Back-test KE, 2022-01 → 2026-06:**
- 14 Phase 3+ transitions; **0 Warnings**, because only one source family exists.
- At level ≥ 1: hit rate 0.36, false-alarm rate 0.29.
- The "161 weeks gained" figure is **not a real lead time**. The metric uses the first-ever warning in the
  window, which comes years before the transition.

**Back-test ZW, 2023-10 → 2024-04:** nothing to score. There are no recent ZW features and no IPC after
2020.

---

## 12. Known limitations and open decisions

| # | Issue | Effect | Options |
|---|---|---|---|
| 1 | Only the `market_prices` family is ingested | Fusion can't reach L2/L3 on real data | Ingest a second family (EO / weather / conflict / text) |
| 2 | The lead-time metric uses the first warning in the window | Long windows give inflated "weeks gained" | Measure from the latest warning within e.g. 26 weeks before each transition |
| 3 | 1-month publication lag + 2-period escalation hysteresis | A warning appears ~2 months after the anomaly starts. It can't come before an IPC transition dated in the same month as the shock | Shorter lag for fast sources, or escalation on "any higher level" instead of "the same level twice" |
| 4 | Hysteresis needs the *same* raw level twice | Alternating raw levels (1, 2, 1, 3) never escalate | Same as #3 |
| 5 | Assessments only for evidence ≤ 6 months old (relative to as-of / today) | Historical data needs `--as-of` | `MAX_STALENESS_MONTHS` in `fuse_risk.py` |
| 6 | Confidence = 0 for L0 (spec formula) | The confidence of "no risk" is not meaningful | Separate confidence definition for L0 |
| 7 | ZW: ZWL rows dropped and the USD series is sparse for 2019–2023 | Few ZW anomalies; no dispersion | Use `usdprice` of ZWL rows instead of dropping them |
| 8 | "Real" price = USD-converted, not deflated with a food CPI | US inflation is included | Add a food-CPI deflator |
| 9 | Back-tests use today's data version (HDX keeps no revision history) | Small look-ahead through later corrections | Keep the dated raw files from now on |
| 10 | The spec's false-alarm and hit definitions overlap (a hit can also be a false alarm) | Metrics can look contradictory | Settle one definition |
| 11 | `market_coverage_drop` leaves out months with zero reports | A district going completely silent isn't flagged | Needs the publication-lag rules settled first |
| 12 | Manual gazetteers (8 ZW markets, a few IPC area names) | Possible misplacements | Review `ZW_GAZETTEER`, `IPC_GAZETTEER` |
| 13 | OpenAI branch untested | — | Test once a key is available |

### Where the code deliberately departs from the original specs

- The Hampel scale has a 10% floor, so stable prices aren't flagged for small real steps.
- ZW outliers: 1,222 against 6,149. The 6,149 figure was not reproduced exactly; mixing currencies under
  this method gives 1,676.
- IPC datasets, the derived phase rule, and dropping projections (§7).
- `market_coverage_drop` counts distinct markets, not the sum across items.
- Dispersion needs ≥3 markets per item-month.
- Hysteresis counts monthly periods, not script runs.
- The back-test reports undefined rates as `None`, not 0.
- The model is `claude-haiku-5-5` instead of the retired `claude-3-5-haiku-latest`. The OpenAI branch uses the
  1.x client, because `openai.ChatCompletion` no longer exists.
- Extra SMS checks: GSM-7 characters, no capability codes, no probabilities.
- The engine store is TinyDB, and SQLite is kept as a mirror, so earlier smoke tests still hold.

---

## 13. Work in progress: simulated El Niño data (not built)

A script `scripts/inject_mock_sources.py` was requested. It would inject simulated EO, weather,
announcement and IPC data for the 2023-24 Zimbabwe drought, tagged with
`raw_ref = SIMULATED_ELNino_2023_24_PENDING_TEAMMATE_DELIVERY_<ts>`, so the full L0 → L3 escalation can be
demonstrated. **It has not been written yet.** The analysis done so far found that, under the current engine
rules, the spec as written would not produce the expected results:

1. **History is required.** Injecting only Oct 2023 – Apr 2024 gives no features (§9.1 step 5). Simulated
   "normal" seasons must be added as history (e.g. 5 seasons).
2. **The announcement can't reach L3 as specified:**
   - It must sit in the April period (2024-04-01), not 2024-04-03.
   - It needs history ("no declaration" in earlier years) to become a feature.
   - L3 needs two consecutive L3 months because of hysteresis.
3. **`fuse_risk` must run with `--as-of`** (e.g. 2024-07-01), because of the 6-month staleness rule.
4. **The back-test would show hit rate 0.** IPC Phase 3 starting 2024-02-01 comes before the engine can
   warn: the Feb data isn't published until end of March, and escalation needs 2 periods.
5. **`--remove` must also clean derived data.** Removing the simulated signals alone would leave their
   features (and any risk/alerts built on them) in the store.
6. **Any back-test metric on simulated signals scored against simulated IPC labels is circular.** It shows
   the pipeline works, but it is **not** evidence of predictive skill and must not be presented as such.

District matching notes from the store:
- There are 91 ZW-ADM2 units plus ZW-ADM0.
- "mt-darwin" is stored as `ZW-ADM2-mount-darwin`.
- "centenary" and "muzarabani" are one unit, `ZW-ADM2-centenary-muzarabani`.
- Urban variants exist, e.g. `chiredzi-urban`, `masvingo-urban`, `beitbridge-urban`.

---

## 14. Configuration reference

| Constant | Value | File |
|---|---|---|
| `PUB_LAG_MONTHS` | 1 | ingest_prices (used by all) |
| `FRESH_MONTHS` | 3 (stale beyond lag + 3) | ingest_prices |
| `BASELINE_YEARS` / `MIN_BASELINE_YEARS` | 5 / 3 | ingest_prices (used by features) |
| `HAMPEL_WINDOW_MONTHS`, `HAMPEL_T`, `HAMPEL_MIN_N`, `HAMPEL_MIN_SCALE` | 6, 3.0, 5, 0.10 | ingest_prices |
| `LOC_FACTOR`, `FANOUT_FACTOR` | 1.0/0.9/0.8, 0.7 | ingest_prices |
| `Z_SCALE_FLOOR`, `FLAG_THRESHOLD` | 0.10, 1.5 | engineer_features |
| `ESCALATE_RUNS`, `DEESCALATE_RUNS`, `RESET_GAP_DAYS` | 2, 3, 90 | fuse_risk |
| `MAX_STALENESS_MONTHS`, `MISSING_FRESHNESS`, `HORIZON_WEEKS` | 6, 0.5, 8 | fuse_risk |
| `MAX_CHARS`, `ANTHROPIC_MODEL`, `OPENAI_MODEL` | 155, `claude-haiku-5-5` (env `SMS_MODEL`), `gpt-4o-mini` | compose_sms |
| `PERSISTENCE_LAG_WEEKS` | 4 | backtest |

---

## 15. File inventory

```
scripts/
  ingest_prices.py      price ingestion + shared utilities
  ingest_ipc.py         IPC ground-truth ingestion
  db.py                 TinyDB helpers (all engine DB access)
  engineer_features.py  features
  fuse_risk.py          fusion + hysteresis
  compose_sms.py        SMS drafting
  backtest.py           weekly point-in-time back-test
  show_results.py       read-only summary
tests/                  92 pytest tests (synthetic data)
test_price_apis.py      original source smoke-check script (pre-existing)
data/{raw,clean,geo,out}/
```
