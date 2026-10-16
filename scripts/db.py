"""
Shared TinyDB (JSON document store) helpers for the engine layer.

One file, several collections (TinyDB tables):
  signals          written by ingest_prices.py / ingest_ipc.py   key: unit_id, source_id, signal_type, period_start
  features         written by engineer_features.py              key: unit_id, signal_type, period_start
  risk_assessment  written by fuse_risk.py                      key: unit_id, run_at
  alerts           written by compose_sms.py                    key: assessment_id, audience
Hysteresis state lives in a plain JSON file next to it (easy to inspect/debug).

All upserts are idempotent and done in bulk: one pass to update existing documents in place (doc_ids are
preserved, so `assessment_id` = risk_assessment doc_id stays stable) and one insert for new documents.
"""
import json
from collections import Counter
from pathlib import Path

from tinydb import TinyDB

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "out" / "signal_observation.json"
HYSTERESIS_PATH = ROOT / "data" / "out" / "hysteresis_state.json"

KEYS = {
    "signals": ("unit_id", "source_id", "signal_type", "period_start"),
    "features": ("unit_id", "signal_type", "period_start"),
    "risk_assessment": ("unit_id", "run_at"),
    "alerts": ("assessment_id", "audience"),
}


def _open(path=None):
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    return TinyDB(p, ensure_ascii=False)


def _bulk_upsert(table_name, docs, path=None):
    """Insert or update docs keyed on KEYS[table_name]. Returns (updated, inserted, total_after)."""
    key = KEYS[table_name]
    docs = [dict(d) for d in docs]
    incoming = {}
    for d in docs:
        incoming[tuple(d.get(k) for k in key)] = d          # last one wins inside a batch
    with _open(path) as db:
        t = db.table(table_name)
        existing = {tuple(doc.get(k) for k in key): doc.doc_id for doc in t.all()}
        upd_ids = [existing[k] for k in incoming if k in existing]
        new = [d for k, d in incoming.items() if k not in existing]
        if upd_ids:
            def apply(doc):
                doc.update(incoming[tuple(doc.get(k) for k in key)])
            t.update(apply, doc_ids=upd_ids)
        if new:
            t.insert_multiple(new)
        return len(upd_ids), len(new), len(t)


def _query(table_name, path=None, **eq):
    """Fetch all docs of a table, filtered in Python on equality (None = no filter).
    Each returned dict carries its TinyDB id as `_doc_id`."""
    with _open(path) as db:
        rows = db.table(table_name).all()
    out = []
    for doc in rows:
        if all(v is None or doc.get(k) == v for k, v in eq.items()):
            d = dict(doc)
            d["_doc_id"] = doc.doc_id
            out.append(d)
    return out


def _country_ok(doc, country):
    return country is None or str(doc.get("unit_id", "")).startswith(f"{country}-")


# ----------------------------------------------------------------- signals
def upsert_signals(docs, path=None):
    return _bulk_upsert("signals", docs, path)


def query_signals(country=None, signal_type=None, source_id=None, period_end_le=None, path=None):
    """period_end_le: ISO date string; keeps docs with period_end <= it (point-in-time filter)."""
    rows = _query("signals", path, signal_type=signal_type, source_id=source_id)
    return [d for d in rows if _country_ok(d, country)
            and (period_end_le is None or str(d.get("period_end")) <= str(period_end_le))]


def count_by(table_name, field, path=None):
    return Counter(d.get(field) for d in _query(table_name, path))


# ----------------------------------------------------------------- features
def upsert_features(docs, path=None):
    return _bulk_upsert("features", docs, path)


def query_features(country=None, period_start_le=None, path=None):
    rows = _query("features", path)
    return [d for d in rows if _country_ok(d, country)
            and (period_start_le is None or str(d.get("period_start")) <= str(period_start_le))]


# ----------------------------------------------------------------- risk
def upsert_risk(docs, path=None):
    return _bulk_upsert("risk_assessment", docs, path)


def query_risk(country=None, level_ge=None, path=None):
    rows = _query("risk_assessment", path)
    return [d for d in rows if _country_ok(d, country) and (level_ge is None or d.get("level", 0) >= level_ge)]


# ----------------------------------------------------------------- alerts
def upsert_alerts(docs, path=None):
    return _bulk_upsert("alerts", docs, path)


def query_alerts(assessment_id=None, audience=None, path=None):
    return _query("alerts", path, assessment_id=assessment_id, audience=audience)


# ----------------------------------------------------------------- hysteresis
def load_hysteresis(path=None):
    p = Path(path or HYSTERESIS_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def save_hysteresis(state, path=None):
    p = Path(path or HYSTERESIS_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(p)
