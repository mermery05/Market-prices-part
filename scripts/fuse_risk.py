#!/usr/bin/env python3
"""
Rule-based fusion of features into tiered risk assessments (TinyDB `risk_assessment` collection).

Per (unit_id, period_start):
  raw level  L0 no flagged feature | L1 >= 1 flagged source_family | L2 >= 2 families |
             L3 = L2 and an announcement_severity value >= 3
  (ipc_phase is ground truth and never takes part in fusion.)
Hysteresis, per unit, one step per monthly period of evidence: a new level must be seen on 2 consecutive
periods before escalating and 3 before de-escalating. It is replayed from the unit's history up to --as-of,
so re-running never double-counts a period; data/out/hysteresis_state.json keeps the resulting state.
  p_shortage  L0 0.05 | L1 0.20 + 0.10*(n-1) | L2 0.50 + 0.10*(n-2) | L3 0.80   (n = families agreeing)
  confidence  min(n/4, 1) * min(baseline_years/5, 1) * freshness_share     (clipped to [0, 1])
A gap of more than 90 days between a unit's evidence periods resets its hysteresis state.
One document per (unit_id, run_at) for the unit's latest period, if that period is recent enough.

Usage:
  python scripts/fuse_risk.py --country KE
  python scripts/fuse_risk.py --country KE --as-of 2024-02-15
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from engineer_features import DIRECTION  # noqa: E402
from ingest_prices import COUNTRIES, OUT, now_utc  # noqa: E402

MODEL_VERSION = "rules-v1"
HORIZON_WEEKS = 8
ESCALATE_RUNS, DEESCALATE_RUNS = 2, 3
MAX_STALENESS_MONTHS = 6       # no assessment when a unit's newest evidence is older than this
MISSING_FRESHNESS = 0.5        # freshness_share used when a unit-period has none
RESET_GAP_DAYS = 90            # longer silence between evidence periods resets hysteresis state


def raw_levels(features):
    """features DataFrame -> one row per (unit_id, period_start) with raw level and supporting info."""
    f = features[features["signal_type"].map(DIRECTION).fillna(0) != 0]
    f = f[f["source_family"] != "ipc"]
    if f.empty:
        return pd.DataFrame()
    k = ["unit_id", "period_start"]
    flagged = f[f["threshold_flag"] == 1]
    out = f.groupby(k).agg(n_total=("source_family", "nunique"), by_all=("baseline_years", "mean")).reset_index()
    fl = (flagged.groupby(k).agg(n_sources_agreeing=("source_family", "nunique"), by_flag=("baseline_years", "mean"),
                                 families_flagged=("source_family", lambda x: sorted(set(x))))
                 .reset_index())
    fr = f[f["signal_type"] == "data_freshness_share"].groupby(k)["value"].mean().rename("freshness_share").reset_index()
    ann = f[(f["signal_type"] == "announcement_severity") & (f["value"] >= 3)].groupby(k).size().rename("ann3").reset_index()
    top = f.sort_values("risk_score", ascending=False).groupby(k).head(3)
    top = top.assign(d=[{"signal_type": a, "value": round(float(b), 3), "z_score": round(float(c), 2)}
                        for a, b, c in zip(top["signal_type"], top["value"], top["z_score"])])
    drv = top.groupby(k)["d"].agg(list).rename("drivers").reset_index()
    for part in (fl, fr, ann, drv):
        out = out.merge(part, on=k, how="left")
    out["n_sources_agreeing"] = out["n_sources_agreeing"].fillna(0).astype(int)
    out["families_flagged"] = [v if isinstance(v, list) else [] for v in out["families_flagged"]]
    out["baseline_years"] = out["by_flag"].fillna(out["by_all"])
    out["freshness_share"] = out["freshness_share"].fillna(MISSING_FRESHNESS)   # missing -> lower confidence
    n = out["n_sources_agreeing"]
    out["raw_level"] = np.select([n >= 2, n == 1], [2, 1], 0)
    out.loc[(out["raw_level"] == 2) & out["ann3"].notna(), "raw_level"] = 3
    return out.drop(columns=["by_flag", "by_all", "ann3"]).sort_values(k).reset_index(drop=True)


def apply_hysteresis(levels):
    """Replay hysteresis per unit over its periods in order. Adds `level`; returns (df, state per unit).
    A gap of more than RESET_GAP_DAYS between evidence periods resets the state (as if seen for the first time)."""
    out, state = [], {}
    for unit, g in levels.sort_values(["unit_id", "period_start"]).groupby("unit_id"):
        cur, pending, count, last = 0, None, 0, None
        for r in g.itertuples():
            ps = pd.Timestamp(r.period_start)
            if last is not None and (ps - last).days > RESET_GAP_DAYS:
                cur, pending, count = 0, None, 0
            last = ps
            raw = r.raw_level
            if raw == cur:
                pending, count = None, 0
            else:
                count = count + 1 if raw == pending else 1
                pending = raw
                if count >= (ESCALATE_RUNS if raw > cur else DEESCALATE_RUNS):
                    cur, pending, count = raw, None, 0
            out.append(cur)
        state[unit] = {"level": cur, "pending_level": pending, "pending_count": count,
                       "period_start": g["period_start"].max(), "raw_level": int(g.iloc[-1]["raw_level"])}
    levels = levels.sort_values(["unit_id", "period_start"]).copy()
    levels["level"] = out
    return levels, state


def p_shortage(level, n):
    if level == 0:
        return 0.05
    if level == 1:
        return min(0.20 + 0.10 * (max(n, 1) - 1), 0.45)
    if level == 2:
        return min(0.50 + 0.10 * (max(n, 2) - 2), 0.75)
    return 0.80


def confidence(n_sources_agreeing, baseline_years, freshness_share):
    """min(n/4, 1) * min(baseline_years/5, 1) * freshness_share, clipped to [0, 1]."""
    c = min(n_sources_agreeing / 4, 1.0) * min(baseline_years / 5, 1.0) * freshness_share
    return round(min(max(c, 0.0), 1.0), 3)


def assess(features, as_of=None, run_at=None):
    """features -> (all unit-period levels, latest assessment per unit, hysteresis state)."""
    f = pd.DataFrame(features)
    if f.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    if as_of is not None:
        f = f[f["period_start"] <= pd.Timestamp(as_of).strftime("%Y-%m-%d")]
    lv = raw_levels(f)
    if lv.empty:
        return lv, pd.DataFrame(), {}
    lv, state = apply_hysteresis(lv)
    ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()
    latest = lv.sort_values("period_start").groupby("unit_id").tail(1).copy()
    age = (ref.year - pd.to_datetime(latest["period_start"]).dt.year) * 12 + (ref.month - pd.to_datetime(latest["period_start"]).dt.month)
    latest = latest[age <= MAX_STALENESS_MONTHS]
    latest["p_shortage"] = [round(p_shortage(l, n), 2) for l, n in zip(latest["level"], latest["n_sources_agreeing"])]
    latest["confidence"] = [confidence(n, by, fr) for n, by, fr in
                            zip(latest["n_sources_agreeing"], latest["baseline_years"], latest["freshness_share"])]
    latest["drivers_json"] = [json.dumps(d) for d in latest["drivers"]]
    latest["run_at"] = run_at
    latest["horizon_weeks"] = HORIZON_WEEKS
    latest["model_version"] = MODEL_VERSION
    latest["as_of"] = pd.Timestamp(as_of).strftime("%Y-%m-%d") if as_of is not None else None
    return lv, latest, state


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--as-of", help="simulated date YYYY-MM-DD (point-in-time)")
    args = ap.parse_args()

    run_at = (pd.Timestamp(args.as_of).strftime("%Y%m%dT120000Z") if args.as_of else now_utc().strftime("%Y%m%dT%H%M%SZ"))
    feats = db.query_features(country=args.country, period_start_le=args.as_of)
    _, latest, state = assess(feats, args.as_of, run_at)

    # live runs keep data/out/hysteresis_state.json; back-dated runs must not overwrite it
    hpath = None if not args.as_of else OUT / f"hysteresis_state_asof_{pd.Timestamp(args.as_of):%Y%m%d}.json"
    prev = db.load_hysteresis(hpath)
    for unit, s in state.items():
        s["previous_level"] = prev.get(unit, {}).get("level")
        s["run_at"] = run_at
    db.save_hysteresis({**prev, **state}, hpath)

    cols = ["unit_id", "run_at", "period_start", "horizon_weeks", "p_shortage", "level", "raw_level", "confidence",
            "drivers_json", "n_sources_agreeing", "n_total", "model_version", "as_of"]
    docs = latest[cols].to_dict("records") if len(latest) else []
    res = db.upsert_risk(docs) if docs else (0, 0, None)

    print(f"Risk {args.country}  run_at {run_at}  features read: {len(feats)}  assessments: {len(docs)}  "
          f"upsert (updated, inserted, total): {res}")
    if docs:
        lv = pd.DataFrame(docs)
        print("  levels:", lv["level"].value_counts().sort_index().to_dict(),
              "  raw levels:", lv["raw_level"].value_counts().sort_index().to_dict())
        changed = [u for u, s in state.items() if s["previous_level"] is not None and s["previous_level"] != s["level"]]
        print(f"  level changes vs previous run: {len(changed)}")
        print(lv.sort_values(["level", "p_shortage"], ascending=False)
                [["unit_id", "period_start", "level", "p_shortage", "confidence", "n_sources_agreeing"]].head(10).to_string(index=False))
    print(f"  hysteresis state: {hpath or db.HYSTERESIS_PATH}")


if __name__ == "__main__":
    main()
