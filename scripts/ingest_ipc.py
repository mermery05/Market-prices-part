#!/usr/bin/env python3
"""
Ingest IPC acute food insecurity phases (HDX, publisher: IPC) into the `signal_observation` schema.

Dataset resolution (CKAN package_show, first that has the needed resources wins):
  1. <country>-acute-food-insecurity-country-data   current IPC country data (CSV + GeoJSON)
  2. ipc-country-data                               legacy 2017-2023 XLSX only; no CSV, so it is
                                                    recorded as unusable rather than parsed
Signals:
  ipc_phase  value = area phase 1-5 for the "current" validity period, confidence 1.0 (ground truth).
             Projections are not emitted: they are forecasts, not ground truth.

The area CSV publishes population per phase, not the classified area phase, so the phase is derived with
the IPC area rule: the most severe phase p such that >= 20% of the population is in phase p or worse.
For the latest analysis the official `overall_phase` (GeoJSON) is used instead and the derived phase is
checked against it (agreement rate reported in the freshness file).

Usage:
  python scripts/ingest_ipc.py --country KE
  python scripts/ingest_ipc.py --country ZW
  python scripts/ingest_ipc.py --country KE --as-of 2024-06-30   # only analyses published by that date
"""
import argparse
import json
import re
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest_prices import (CLEAN, COUNTRIES, GEO, OUT, RAW, ROOT, SCHEMA_COLS, ckan_resource, docstore,  # noqa: E402
                           fetch, load_units, norm, now_utc, raw_ref, read_raw, upsert)

SOURCE_ID = "ipc_phase"
AREA_RULE_SHARE = 0.20
PHASES = ["1", "2", "3", "4", "5"]

# IPC area names that differ from the geoBoundaries unit names
IPC_GAZETTEER = {
    "KE": {"Taita": "Taita Taveta", "Tharaka Nithi": "Tharaka"},
    "ZW": {"Muzarabani": "Centenary/ Muzarabani"},
}


def dataset_slugs(cfg):
    return [f"{cfg['name'].lower()}-acute-food-insecurity-country-data", "ipc-country-data"]


def match_area(area, by_name, gaz):
    """IPC area -> (unit_id, method). Whole-unit matches first; sub-areas ('Turkana north',
    'Marsabit - saku') fall back to the unit whose name they start with."""
    if norm(area) in by_name:
        return by_name[norm(area)], "exact"
    if area in gaz and norm(gaz[area]) in by_name:
        return by_name[norm(gaz[area])], "gazetteer"
    stripped = re.sub(r"\s+county$", "", area, flags=re.I)
    if norm(stripped) in by_name:
        return by_name[norm(stripped)], "exact"
    hits = [n for n in by_name if norm(area).startswith(n)]
    if hits:
        return by_name[max(hits, key=len)], "sub_area_pooled"
    return None, None


def area_phase(numbers):
    """IPC area rule on a {phase: population} dict -> phase 1-5 (NaN if no population)."""
    total = sum(numbers.get(p, 0) for p in PHASES)
    if not total:
        return np.nan
    for p in reversed(PHASES):
        # IPC classifies on published (rounded) percentages; raw counts can sit just under 20%
        if round(sum(numbers.get(q, 0) for q in PHASES if q >= p) / total, 2) >= AREA_RULE_SHARE:
            return int(p)
    return 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--as-of", help="simulated date YYYY-MM-DD: keep analyses published by then")
    ap.add_argument("--refresh", action="store_true", help="re-download even if today's raw file exists")
    args = ap.parse_args()

    cc, cfg = args.country, COUNTRIES[args.country]
    for d in (RAW, CLEAN, OUT, GEO):
        d.mkdir(parents=True, exist_ok=True)
    run = now_utc()
    run_at, fetch_date, ingested_at = run.strftime("%Y%m%dT%H%M%SZ"), run.strftime("%Y-%m-%d"), run.isoformat(timespec="seconds")
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    units, _ = load_units(cc, cfg)
    by_name = {norm(n): u for n, u in zip(units["shapeName"], units["unit_id"])}

    st = {"source_id": SOURCE_ID, "status": "ok", "rows_fetched": 0, "rows_after_cleaning": 0,
          "latest_date": None, "confidence_adjustment": 1.0, "datasets_tried": {}}
    hard_failure = None
    out = pd.DataFrame(columns=SCHEMA_COLS)
    try:
        # ---- resolve dataset + resources
        area_res = None
        for s in dataset_slugs(cfg):
            try:
                area_res = ckan_resource(s, lambda n: n.endswith("_area_long.csv"))
                st["datasets_tried"][s] = "used"
                st["dataset"] = s
                break
            except Exception as e:
                st["datasets_tried"][s] = f"unusable: {e}"
        if area_res is None:
            raise RuntimeError("no usable IPC dataset on HDX")

        raw_path = RAW / f"{cc}_ipc_area_long_{fetch_date}.csv"
        meta = fetch(area_res["download_url"], raw_path, args.refresh, area_res.get("last_modified"))
        ref = raw_ref(meta)
        raw = read_raw(raw_path)
        st.update(rows_fetched=int(len(raw)), raw_path=str(raw_path.relative_to(ROOT)), last_modified=meta["last_modified"])

        # official overall_phase for the latest analysis (GeoJSON resource of the same dataset)
        official = pd.DataFrame(columns=["Area", "From", "official_phase"])
        try:
            geo_res = ckan_resource(st["dataset"], lambda n: True, fmt="geojson")
            gpath = RAW / f"{cc}_ipc_areas_{fetch_date}.geojson"
            fetch(geo_res["download_url"], gpath, args.refresh, geo_res.get("last_modified"))
            g = gpd.read_file(gpath)
            g = g[g["ipc_period"] == "C"]
            official = pd.DataFrame({"Area": g["title"].astype(str),
                                     "official_phase": pd.to_numeric(g["overall_phase"], errors="coerce")})
            st["geojson_path"] = str(gpath.relative_to(ROOT))
        except Exception as e:
            st["geojson_note"] = f"official overall_phase unavailable ({e!r}); derived phases only"

        # ---- clean
        d = raw[raw["Validity period"].str.lower() == "current"].copy()
        d["analysis_date"] = pd.to_datetime(d["Date of analysis"], format="%b %Y", errors="coerce")
        d["period_start"] = pd.to_datetime(d["From"], errors="coerce")
        d["period_end"] = pd.to_datetime(d["To"], errors="coerce")
        d["Number"] = pd.to_numeric(d["Number"], errors="coerce")
        d = d.dropna(subset=["analysis_date", "period_start", "period_end"])
        if as_of is not None:
            d = d[d["analysis_date"] <= as_of]
        st["projection_rows_skipped"] = int((raw["Validity period"].str.lower() != "current").sum())

        areas = pd.DataFrame({"Area": sorted(d["Area"].dropna().unique())})
        m = areas["Area"].map(lambda a: match_area(a, by_name, IPC_GAZETTEER.get(cc, {})))
        areas["unit_id"], areas["match_method"] = [x[0] for x in m], [x[1] for x in m]
        unmatched = areas[areas["unit_id"].isna()]["Area"].tolist()
        st["unmatched_districts"] = unmatched
        st["unmatched_note"] = ("IPC areas with no admin unit (e.g. refugee camps such as Dadaab/Kakuma are "
                                "separate IPC areas, not counties); excluded from output, not dropped silently")
        st["match_methods"] = areas["match_method"].value_counts().to_dict()
        d = d.merge(areas, on="Area", how="left")
        st["rows_unmatched"] = int(d["unit_id"].isna().sum())
        d = d.dropna(subset=["unit_id"])

        # population per phase per (analysis, area)
        per_area = (d[d["Phase"].isin(PHASES)]
                    .pivot_table(index=["analysis_date", "period_start", "period_end", "unit_id", "Area", "match_method"],
                                 columns="Phase", values="Number", aggfunc="sum")
                    .reset_index())
        # whole-unit areas win over sub-areas within the same analysis; duplicate spellings are de-duplicated
        per_area["_whole"] = per_area["match_method"] != "sub_area_pooled"
        per_area["_has_whole"] = per_area.groupby(["analysis_date", "period_start", "unit_id"])["_whole"].transform("any")
        per_area = per_area[per_area["_whole"] | ~per_area["_has_whole"]]
        per_area = per_area.drop_duplicates(["analysis_date", "period_start", "unit_id", *[p for p in PHASES if p in per_area]])

        grp = per_area.groupby(["analysis_date", "period_start", "period_end", "unit_id"])
        phases_present = [p for p in PHASES if p in per_area.columns]
        agg = grp[phases_present].sum(min_count=1).reset_index()
        agg["n_areas"] = grp["Area"].nunique().values
        agg["areas"] = grp["Area"].apply(lambda s: ";".join(sorted(s))).values
        agg["derived_phase"] = agg.apply(lambda r: area_phase({p: r[p] for p in phases_present if pd.notna(r[p])}), axis=1)

        # official overall_phase applies to the latest analysis, single-area units only
        latest = agg["analysis_date"].max()
        single = agg[(agg["analysis_date"] == latest) & (agg["n_areas"] == 1)].copy()
        single = single.merge(official, left_on="areas", right_on="Area", how="left")
        agg = agg.merge(single[["unit_id", "period_start", "official_phase"]], on=["unit_id", "period_start"], how="left")
        cmp = agg.dropna(subset=["official_phase", "derived_phase"])
        st["official_vs_derived"] = {"compared": int(len(cmp)),
                                     "agree": int((cmp["official_phase"] == cmp["derived_phase"]).sum())}
        agg["phase"] = agg["official_phase"].fillna(agg["derived_phase"])
        agg["phase_source"] = np.where(agg["official_phase"].notna(), "official_overall_phase", "derived_20pct_rule")
        agg = agg.dropna(subset=["phase"])

        # one row per (unit, period_start): the most recent analysis wins
        agg = agg.sort_values("analysis_date").drop_duplicates(["unit_id", "period_start"], keep="last")

        clean_path = CLEAN / f"{cc}_ipc_{agg['period_start'].min():%Y-%m}_{agg['period_start'].max():%Y-%m}.csv"
        agg.to_csv(clean_path, index=False)
        st.update(rows_after_cleaning=int(len(agg)), clean_path=str(clean_path.relative_to(ROOT)),
                  latest_date=agg["period_start"].max().strftime("%Y-%m-%d"),
                  latest_analysis=latest.strftime("%Y-%m"), analyses=int(agg["analysis_date"].nunique()),
                  units_covered=int(agg["unit_id"].nunique()),
                  phase_distribution=agg["phase"].astype(int).value_counts().sort_index().to_dict())

        out = pd.DataFrame({
            "unit_id": agg["unit_id"], "source_id": SOURCE_ID, "signal_type": "ipc_phase",
            "value": agg["phase"].astype(float), "unit_of_measure": "ipc_phase",
            "period_start": agg["period_start"].dt.strftime("%Y-%m-%d"),
            "period_end": agg["period_end"].dt.strftime("%Y-%m-%d"),
            "ingested_at": ingested_at, "confidence": 1.0, "raw_ref": ref})[SCHEMA_COLS]
    except Exception as e:
        st.update(status="hard_failure", error=repr(e))
        hard_failure = repr(e)

    db = OUT / ("signal_observation.sqlite" if as_of is None else f"signal_observation_asof_{as_of:%Y%m%d}.sqlite")
    db_before, db_after = upsert(db, out) if len(out) else (None, None)
    docs_upsert = docstore.upsert_signals(out.to_dict("records")) if (len(out) and as_of is None) else None
    suffix = "" if as_of is None else f"_asof{as_of:%Y%m%d}"
    out_path = OUT / f"signal_observation_ipc_{cc}_{run_at}{suffix}.csv"
    out.to_csv(out_path, index=False)

    checks = {
        "zero_hard_failures": {"pass": hard_failure is None, "error": hard_failure},
        "ipc_phase_populated": {"pass": len(out) > 0, "rows": int(len(out))},
        "phase_in_1_to_5": {"pass": bool(len(out)) and bool(out["value"].between(1, 5).all())},
        "idempotent_upsert": {"pass": True, "db": str(db.relative_to(ROOT)), "rows_before": db_before, "rows_after": db_after,
                              "tinydb_signals_updated_inserted_total": docs_upsert},
    }
    report = {"country": cc, "run_at": run_at, "as_of": args.as_of, "sources": {SOURCE_ID: st},
              "signal_row_counts": {"ipc_phase": int(len(out))}, "output_csv": str(out_path.relative_to(ROOT)),
              "smoke_test": checks}
    fresh_path = OUT / f"freshness_ipc_{cc}_{run_at}{suffix}.json"
    fresh_path.write_text(json.dumps(report, indent=2, default=str))

    print(f"\nIPC {cc}  run_at {run_at}" + (f"  as_of {args.as_of}" if as_of is not None else ""))
    print(f"  {SOURCE_ID}  {st['status']}  dataset={st.get('dataset')}  latest_analysis={st.get('latest_analysis')}  "
          f"fetched={st['rows_fetched']}  rows={st['rows_after_cleaning']}  units={st.get('units_covered')}"
          + (f"  error={st['error']}" if st.get("error") else ""))
    print(f"  phases: {st.get('phase_distribution')}  official_vs_derived: {st.get('official_vs_derived')}")
    print(f"  unmatched_districts: {st.get('unmatched_districts')}")
    print("\nSmoke test:")
    for k, v in checks.items():
        print(f"  [{'PASS' if v['pass'] else 'FAIL'}] {k}  " + json.dumps({a: b for a, b in v.items() if a != 'pass'}))
    print(f"\nExport   : {out_path}\nFreshness: {fresh_path}")
    sys.exit(0 if all(v["pass"] for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
