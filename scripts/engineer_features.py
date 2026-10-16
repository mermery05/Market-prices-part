#!/usr/bin/env python3
"""
Turn `signals` documents into per-signal risk features (TinyDB `features` collection).

Per (unit_id, signal_type, source_id) series, for each period_start:
  baseline   = median of `value` in the same calendar month over the previous 5 years (>= 3 obs, trailing only)
  mad        = median absolute deviation of those same-month values
  z_score    = (value - baseline) / max(mad * 1.4826, 0.10)
  risk_score = z_score * DIRECTION[signal_type]      (positive = bad for every signal)
  threshold_flag = 1 if risk_score > 1.5
ipc_phase is ground truth and never becomes a feature.
Rows without enough history get no feature (absent, never zero-filled). When several sources report the same
(unit, signal_type, period), the most alarming one (max risk_score) is kept; `n_source_series` records how many.

Point-in-time: with --as-of, only signals published by then are used (period_end + 1-month publication lag
<= as-of). Baselines only look backwards, so a feature never changes once its period is published.

Usage:
  python scripts/engineer_features.py --country KE
  python scripts/engineer_features.py --country KE --as-of 2024-02-15
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from ingest_prices import COUNTRIES, PUB_LAG_MONTHS, now_utc, same_month_stats  # noqa: E402

FEATURE_VERSION = "features-v1"
Z_SCALE_FLOOR = 0.10
FLAG_THRESHOLD = 1.5
GROUND_TRUTH = {"ipc_phase"}   # evaluation targets: never turned into features

DIRECTION = {
    "ndvi_z": -1, "evi_z": -1, "spi_1m": -1, "spi_3m": -1, "soil_moist_z": -1,  # lower = worse
    "price_anom_real": +1, "price_anom_nominal": +1, "price_velocity_anom": +1,
    "price_dispersion_anom": +1, "market_coverage_drop": -1,  # negative YoY = worse
    "basket_cost_change": +1, "data_freshness_share": -1,  # lower freshness = worse
    "fx_devaluation_3m": +1, "fuel_price_anom": +1, "food_cpi_yoy": +1,
    "conflict_events_30d": +1,
    "announcement_severity": +1, "crowd_crop_status": +1,
    "ipc_phase": 0,  # ground truth, NOT used in fusion
}

# source_id prefix -> source_family (first match wins)
SOURCE_FAMILY = [
    ("wfp_hdx", "market_prices"), ("fewsnet", "market_prices"), ("wfp_gmm", "market_prices"),
    ("ipc", "ipc"), ("ndvi", "eo"), ("evi", "eo"), ("spi", "eo"), ("chirps", "eo"),
    ("soil_moist", "weather"), ("announcement", "text"), ("crowd", "crowd"),
    ("acled", "conflict"), ("conflict", "conflict"),
]


def source_family(source_id, signal_type=None):
    for prefix, fam in SOURCE_FAMILY:
        if str(source_id).startswith(prefix):
            return fam
    for prefix, fam in SOURCE_FAMILY:          # fall back on the signal name (e.g. ndvi_z, spi_3m)
        if str(signal_type).startswith(prefix):
            return fam
    return "other"


def pit_cutoff(as_of):
    """Latest period_end that would have been published by `as_of`."""
    return (pd.Timestamp(as_of) - pd.DateOffset(months=PUB_LAG_MONTHS)).strftime("%Y-%m-%d")


def compute_features(signals, as_of=None):
    """signals: list of signal dicts (or DataFrame). Returns a DataFrame of feature documents."""
    s = pd.DataFrame(signals)
    if s.empty:
        return pd.DataFrame()
    if as_of is not None:
        s = s[s["period_end"] <= pit_cutoff(as_of)]
    return collapse_sources(series_features(s))


def series_features(s):
    """Per-source-series features. A row depends only on earlier periods of its own series, so computing this
    once on all signals and filtering later gives the same rows as filtering first (used by backtest.py)."""
    s = s[~s["signal_type"].isin(GROUND_TRUTH)].dropna(subset=["value"]).copy()
    if s.empty:
        return pd.DataFrame()
    s["period_start"] = pd.to_datetime(s["period_start"])
    key = ["unit_id", "signal_type", "source_id"]
    s = s.sort_values(key + ["period_start"]).reset_index(drop=True)
    s["baseline"], s["mad"], s["baseline_years"] = same_month_stats(s, key, "value")
    s = s.dropna(subset=["baseline"])                     # not enough same-month history -> absent
    s["z_score"] = (s["value"] - s["baseline"]) / np.maximum(s["mad"] * 1.4826, Z_SCALE_FLOOR)
    s["direction"] = s["signal_type"].map(DIRECTION).fillna(0)
    s["risk_score"] = s["z_score"] * s["direction"]
    s["threshold_flag"] = (s["risk_score"] > FLAG_THRESHOLD).astype(int)
    s["source_family"] = [source_family(a, b) for a, b in zip(s["source_id"], s["signal_type"])]
    return s


def collapse_sources(s):
    """One feature per (unit, signal_type, period): keep the most alarming source."""
    if s.empty:
        return pd.DataFrame()
    s = s.copy()
    k = ["unit_id", "signal_type", "period_start"]
    s["n_source_series"] = s.groupby(k)["source_id"].transform("size")
    s = s.sort_values("risk_score").drop_duplicates(k, keep="last")
    s["period_start"] = s["period_start"].dt.strftime("%Y-%m-%d")
    cols = ["unit_id", "signal_type", "period_start", "period_end", "value", "baseline", "mad", "baseline_years",
            "z_score", "risk_score", "threshold_flag", "source_family", "source_id", "n_source_series",
            "confidence"]
    f = s[cols].rename(columns={"confidence": "signal_confidence"}).copy()
    for c in ["value", "baseline", "mad", "z_score", "risk_score"]:
        f[c] = f[c].astype(float).round(4)
    f["baseline_years"] = f["baseline_years"].astype(int)
    f["feature_version"] = FEATURE_VERSION
    return f.sort_values(k).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--as-of", help="simulated date YYYY-MM-DD (point-in-time)")
    args = ap.parse_args()

    cutoff = pit_cutoff(args.as_of) if args.as_of else None
    signals = db.query_signals(country=args.country, period_end_le=cutoff)
    feats = compute_features(signals, args.as_of)
    unknown = sorted(set(pd.DataFrame(signals)["signal_type"]) - set(DIRECTION)) if signals else []
    res = db.upsert_features(feats.to_dict("records")) if len(feats) else (0, 0, None)

    print(f"Features {args.country}  run_at {now_utc():%Y%m%dT%H%M%SZ}" + (f"  as_of {args.as_of}" if args.as_of else ""))
    print(f"  signals read: {len(signals)}   features computed: {len(feats)}   upsert (updated, inserted, total): {res}")
    if unknown:
        print(f"  WARNING signal types without a DIRECTION entry (risk_score=0): {unknown}")
    if len(feats):
        summary = feats.groupby("signal_type").agg(n=("z_score", "size"), flagged=("threshold_flag", "sum"),
                                                   latest=("period_start", "max"))
        print(summary.to_string())
    sys.exit(0 if len(feats) else 1)


if __name__ == "__main__":
    main()
