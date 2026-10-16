#!/usr/bin/env python3
"""
Back-test the rules engine against IPC Phase 3+ transitions.

For every weekly tick in [--from, --to], features and fusion are recomputed in-process with as_of = tick
(point-in-time: only signals published by then, trailing baselines, hysteresis replayed). Nothing is written
to the live features / risk_assessment collections or the live hysteresis file.

Ground truth (signals collection, signal_type='ipc_phase'):
  transition  first IPC 'current' period starting inside [from, to] with phase >= 3 whose previous IPC
              period for that unit was < 3 (units with no earlier analysis are not counted)
Metrics, for warnings = first tick with level >= threshold:
  hit_rate             transitions preceded (or met on the day) by a warning / all transitions
  false_alarm_rate     warned units with no Phase 3+ IPC period overlapping the 8 weeks after the warning /
                       warned units whose next 8 weeks are covered by IPC at all
  median_weeks_gained  median of (transition date - warning date) over hits, in weeks
Persistence baseline: the same metrics when the predictor is the level the engine had 4 weeks earlier.
The spec threshold is level >= 2 (Warning); level >= 1 is reported alongside.

Usage:
  python scripts/backtest.py --country ZW --from 2023-10-01 --to 2024-04-03
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from engineer_features import collapse_sources, compute_features, pit_cutoff, series_features  # noqa: E402,F401
from fuse_risk import HORIZON_WEEKS, assess  # noqa: E402
from ingest_prices import COUNTRIES, OUT, now_utc  # noqa: E402

PERSISTENCE_LAG_WEEKS = 4


def ipc_transitions(ipc, start, end):
    out = {}
    for unit, g in ipc.sort_values("period_start").groupby("unit_id"):
        prev = None
        for r in g.itertuples():
            if start <= r.period_start <= end and r.value >= 3 and prev is not None and prev < 3:
                out[unit] = r.period_start
                break
            prev = r.value
    return out


def phase3_within(ipc_unit, t0, weeks):
    """(covered, any_phase3) for IPC periods overlapping [t0, t0 + weeks]."""
    t1 = t0 + pd.Timedelta(weeks=weeks)
    ov = ipc_unit[(ipc_unit["period_start"] <= t1) & (ipc_unit["period_end"] >= t0)]
    return len(ov) > 0, bool((ov["value"] >= 3).any())


def metrics(warn_dates, transitions, ipc):
    hits, gained = 0, []
    for unit, tdate in transitions.items():
        w = warn_dates.get(unit)
        if w is not None and w <= tdate:
            hits += 1
            gained.append((tdate - w).days / 7)
    fa_den = fa = 0
    for unit, w in warn_dates.items():
        covered, p3 = phase3_within(ipc[ipc["unit_id"] == unit], w, HORIZON_WEEKS)
        if covered:
            fa_den += 1
            fa += 0 if p3 else 1
    return {
        "hit_rate": round(hits / len(transitions), 3) if transitions else None,
        "false_alarm_rate": round(fa / fa_den, 3) if fa_den else None,
        "median_weeks_gained": round(float(np.median(gained)), 1) if gained else None,
        "n_hits": hits, "n_warned_units": len(warn_dates), "n_warned_units_with_ipc_coverage": fa_den,
    }


def first_dates(levels, threshold):
    """levels: DataFrame tick x unit -> {unit: first tick with level >= threshold}."""
    out = {}
    for unit in levels.columns:
        hit = levels.index[levels[unit].fillna(-1) >= threshold]
        if len(hit):
            out[unit] = hit[0]
    return out


def run_backtest(country, start_date, end_date, out_dir=None):
    """Run the weekly back-test and write the JSON report. Returns (report, path)."""
    start, end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    run_at = now_utc().strftime("%Y%m%dT%H%M%SZ")

    signals = pd.DataFrame(db.query_signals(country=country))
    # equivalent to compute_features(signals, as_of=t) per tick, without recomputing every baseline each time
    series = series_features(signals) if len(signals) else pd.DataFrame(columns=["period_end"])
    ticks = pd.date_range(start, end, freq="7D")
    per_tick = {}
    for t in ticks:
        feats = collapse_sources(series[series["period_end"] <= pit_cutoff(t)])
        _, latest, _ = assess(feats, as_of=t) if len(feats) else (None, pd.DataFrame(), None)
        per_tick[t] = dict(zip(latest["unit_id"], latest["level"])) if len(latest) else {}
    levels = pd.DataFrame.from_dict(per_tick, orient="index").sort_index()
    if levels.empty:
        levels = pd.DataFrame(index=ticks)
    persistence = levels.shift(PERSISTENCE_LAG_WEEKS)    # weekly ticks -> level "last month"

    ipc = pd.DataFrame(db.query_signals(country=country, signal_type="ipc_phase"))
    if len(ipc):
        ipc["period_start"] = pd.to_datetime(ipc["period_start"])
        ipc["period_end"] = pd.to_datetime(ipc["period_end"])
    else:
        ipc = pd.DataFrame(columns=["unit_id", "period_start", "period_end", "value"])
    transitions = ipc_transitions(ipc, start, end)
    in_window = ipc[(ipc["period_start"] <= end) & (ipc["period_end"] >= start)]

    notes = []
    if in_window.empty:
        last = ipc["period_end"].max()
        notes.append(f"no IPC analysis overlaps the window (latest IPC period ends {last.date() if pd.notna(last) else None}); "
                     "hit/false-alarm rates cannot be computed")
    if not levels.notna().any().any():
        notes.append("no unit had recent enough features for an assessment at any tick")
    elif levels.max().max() < 2:
        notes.append("no unit reached level 2 (needs flags from two different source families)")

    report = {
        "country": country, "from": str(start_date), "to": str(end_date), "run_at": run_at,
        "n_ticks": len(ticks), "n_districts": int(in_window["unit_id"].nunique()),
        "n_units_assessed": int(levels.notna().any().sum()) if len(levels.columns) else 0,
        "n_phase_3_plus_transitions": len(transitions),
        "transitions": {u: d.strftime("%Y-%m-%d") for u, d in transitions.items()},
        "model_metrics": metrics(first_dates(levels, 2), transitions, ipc),
        "persistence_baseline": metrics(first_dates(persistence, 2), transitions, ipc),
        "model_metrics_level1": metrics(first_dates(levels, 1), transitions, ipc),
        "persistence_baseline_level1": metrics(first_dates(persistence, 1), transitions, ipc),
        "level_counts_over_ticks": {str(int(k)): int(v) for k, v in levels.stack().value_counts().sort_index().items()},
        "notes": notes,
    }
    path = Path(out_dir or OUT) / f"backtest_{country}_{run_at}.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    return report, path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    args = ap.parse_args()
    report, path = run_backtest(args.country, args.start, args.end)
    ticks, transitions, notes = range(report["n_ticks"]), report["transitions"], report["notes"]

    m, m1 = report["model_metrics"], report["model_metrics_level1"]
    print(f"Backtest {args.country} {args.start} -> {args.end}: {len(ticks)} ticks, "
          f"{report['n_units_assessed']} units assessed, {report['n_districts']} districts with IPC in window, "
          f"{len(transitions)} Phase 3+ transitions")
    print(f"  level>=2  model {m}\n            persistence {report['persistence_baseline']}")
    print(f"  level>=1  model {m1}\n            persistence {report['persistence_baseline_level1']}")
    for n in notes:
        print(f"  NOTE: {n}")
    print(f"median_weeks_gained: {m['median_weeks_gained']}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
