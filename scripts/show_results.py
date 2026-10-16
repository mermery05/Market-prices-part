#!/usr/bin/env python3
"""
Show actual pipeline output - SMS samples, risk assessments, backtest metrics.
Reads data/out/signal_observation.json (via scripts/db.py, read-only) and prints a visual summary.
Handles None values, deduplicates districts (latest run per unit), shows the newest backtest per country.

Usage:
  python scripts/show_results.py
  python scripts/show_results.py --country KE
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402

OUT_DIR = db.DB_PATH.parent
LEVEL_NAMES = {0: "Normal", 1: "Watch", 2: "Warning", 3: "Emergency"}
LEVEL_CHARS = {0: ".", 1: "o", 2: "O", 3: "X"}


def safe_fmt(val, suffix="", fmt=".2f"):
    """Safely format a value that might be None."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val) + suffix


def num(val):
    """None-safe number for sorting/comparison."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def latest_per_unit(rows):
    """Keep the most recent run_at per unit_id."""
    best = {}
    for r in rows:
        uid = r.get("unit_id", "?")
        if uid not in best or str(r.get("run_at", "")) > str(best[uid].get("run_at", "")):
            best[uid] = r
    return list(best.values())


def print_metrics(title, m):
    m = m or {}
    print(f"\n  {title}")
    print(f"    Hit rate:             {safe_fmt(m.get('hit_rate'))}")
    print(f"    False alarm rate:     {safe_fmt(m.get('false_alarm_rate'))}")
    print(f"    Median weeks gained:  {safe_fmt(m.get('median_weeks_gained'), ' weeks', '.1f')}")
    print(f"    Hits / warned units:  {m.get('n_hits', 'N/A')} / {m.get('n_warned_units', 'N/A')}")


def show_backtest(path):
    bt = json.loads(path.read_text())
    print(f"\n  File: {path.name}")
    print(f"  Country: {bt.get('country')}   Period: {bt.get('from')} to {bt.get('to')}")
    print(f"  Districts with IPC in window: {bt.get('n_districts', '?')}   "
          f"Units assessed: {bt.get('n_units_assessed', '?')}   "
          f"IPC Phase 3+ transitions: {bt.get('n_phase_3_plus_transitions', '?')}")

    model = bt.get("model_metrics") or {}
    base = bt.get("persistence_baseline") or {}
    print_metrics("YOUR MODEL, Warning (level >= 2):", model)
    print_metrics("PERSISTENCE BASELINE (level 4 weeks earlier), level >= 2:", base)
    if bt.get("model_metrics_level1"):
        print_metrics("YOUR MODEL, Watch (level >= 1):", bt.get("model_metrics_level1"))
        print_metrics("PERSISTENCE BASELINE, level >= 1:", bt.get("persistence_baseline_level1"))

    model_wg, base_wg = model.get("median_weeks_gained"), base.get("median_weeks_gained")
    if model_wg is not None and base_wg is not None:
        diff = model_wg - base_wg
        verdict = "BEATS" if diff > 0 else "DOES NOT BEAT"
        print(f"\n  *** Level >= 2: model {verdict} the baseline ({model_wg:.1f} vs {base_wg:.1f} weeks, "
              f"{diff:+.1f}) ***")
        if diff > 0:
            print(f"  *** Pitch number: {model_wg:.1f} weeks gained ***")
    else:
        print("\n  (no level >= 2 lead time: metric undefined - see notes below)")
    for note in bt.get("notes") or []:
        print(f"  NOTE: {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", help="only show one country, e.g. KE or ZW")
    args = ap.parse_args()
    cc = args.country

    if not db.DB_PATH.exists():
        print("=" * 70)
        print("DB not found. Run the pipeline first:")
        print("=" * 70)
        c = cc or "KE"
        print(f"  python scripts/ingest_prices.py --country {c}")
        print(f"  python scripts/ingest_ipc.py --country {c}")
        print(f"  python scripts/engineer_features.py --country {c}")
        print(f"  python scripts/fuse_risk.py --country {c}")
        print(f"  python scripts/compose_sms.py --country {c} --audience all")
        print(f"  python scripts/backtest.py --country {c} --from 2022-01-01 --to 2026-06-30")
        return

    print()
    print("=" * 78)
    print("  SMS SHORTAGE EARLY WARNING - PIPELINE OUTPUT SUMMARY" + (f"  [{cc}]" if cc else ""))
    print("=" * 78)

    in_country = (lambda r: str(r.get("unit_id", "")).startswith(f"{cc}-")) if cc else (lambda r: True)

    # 1. COLLECTION COUNTS
    print("\n[1] COLLECTION COUNTS")
    print("-" * 78)
    tables = {
        "signals": db.query_signals(country=cc),
        "features": db.query_features(country=cc),
        "risk_assessment": db.query_risk(country=cc),
        "alerts": [a for a in db.query_alerts() if in_country(a)],
    }
    for name, rows in tables.items():
        bar = "#" * min(len(rows) // 500, 30)
        print(f"  {name:<20} {len(rows):>6}  {bar}")

    # 2. RISK ASSESSMENT BREAKDOWN
    print("\n[2] RISK LEVEL DISTRIBUTION (latest run per district)")
    print("-" * 78)
    risks = latest_per_unit(tables["risk_assessment"])
    if not risks:
        print("  (no risk assessments - run fuse_risk.py first)")
    level_counts = Counter(int(num(r.get("level"))) for r in risks)
    for level in sorted(level_counts):
        count = level_counts[level]
        print(f"  L{level} {LEVEL_NAMES.get(level, '?'):<10} {count:>5}  {LEVEL_CHARS.get(level, '?') * min(count, 50)}")

    # 3. TOP 5 MOST AT-RISK DISTRICTS
    print("\n[3] TOP 5 MOST AT-RISK DISTRICTS (latest run per district)")
    print("-" * 78)
    high_risk = sorted((r for r in risks if num(r.get("level")) >= 1),
                       key=lambda r: (-num(r.get("level")), -num(r.get("p_shortage")), -num(r.get("confidence"))))
    if not high_risk:
        print("  (no districts at Watch or above - check fuse_risk.py output)")
        print("  Note: If only L0 shown, the 2-source agreement rule may be blocking escalation.")
        print("  You need 2+ source families flagged for L2 (Warning).")
    for r in high_risk[:5]:
        level = int(num(r.get("level")))
        try:
            drivers = json.loads(r.get("drivers_json") or "[]")
        except json.JSONDecodeError:
            drivers = []
        print(f"\n  >> {r.get('unit_id', '?')}   (period {r.get('period_start', '?')}, run {r.get('run_at', '?')})")
        print(f"     Level {level} ({LEVEL_NAMES.get(level, '?')})  |  p_shortage={safe_fmt(r.get('p_shortage'))}  |  "
              f"confidence={safe_fmt(r.get('confidence'))}  |  sources={r.get('n_sources_agreeing', 'N/A')}")
        if drivers:
            print("     Top drivers:")
            for d in drivers[:3]:
                print(f"       - {d.get('signal_type', '?'):<25}  value={safe_fmt(d.get('value'), fmt='>8.2f')}  "
                      f"z={safe_fmt(d.get('z_score'), fmt='>6.2f')}")

    # 4. SAMPLE SMS PER AUDIENCE x LEVEL (newest first)
    print("\n\n[4] SAMPLE SMS PER AUDIENCE x LEVEL (newest per audience/level/status)")
    print("=" * 78)
    alerts = sorted(tables["alerts"], key=lambda a: str(a.get("created_at", "")), reverse=True)
    if not alerts:
        print("  (no alerts in DB - run compose_sms.py first)")
    seen, samples = set(), []
    for a in alerts:
        key = (a.get("audience", "?"), a.get("level", 0), a.get("status", "?"))
        if key not in seen:
            seen.add(key)
            samples.append(a)
    for a in sorted(samples, key=lambda x: (num(x.get("level")), str(x.get("audience", "")))):
        sms = a.get("template_text") or ""
        print()
        print(f"+--- {str(a.get('audience', '?')).upper()} | Level {a.get('level', '?')} | Status: {a.get('status', '?')}"
              f" | {a.get('generator', '?')} ---")
        print(f"|  District: {a.get('unit_id', '?')}   created {a.get('created_at', '?')}")
        print(f"|  SMS ({len(sms)} chars):")
        for i in range(0, len(sms), 70):
            print(f"|    {sms[i:i + 70]}")
        print(f"+{'-' * 76}")

    # 5. BACKTEST SUMMARY (newest file per country)
    print("\n\n[5] BACKTEST RESULTS (newest per country)")
    print("=" * 78)
    files = sorted(OUT_DIR.glob(f"backtest_{cc or '*'}_*.json"), key=lambda p: p.stat().st_mtime)
    newest = {}
    for p in files:
        newest[p.name.split("_")[1]] = p
    if not newest:
        c = cc or "KE"
        print(f"  (no backtest results - run: python scripts/backtest.py --country {c} --from 2022-01-01 --to 2026-06-30)")
    for country in sorted(newest):
        show_backtest(newest[country])

    print()
    print("=" * 78)
    print("  Done. Copy this output for your pitch rehearsal.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
