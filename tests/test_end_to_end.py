import json
from collections import Counter

from tinydb import TinyDB

import backtest
import compose_sms
import db
import engineer_features
import fuse_risk
from conftest import REAL_DB

AS_OF = "2025-01-15"     # synthetic data ends 2024-12; assessments need recent evidence


def run_pipeline(run_main):
    run_main(engineer_features, "--country", "TEST")
    run_main(fuse_risk, "--country", "TEST", "--as-of", AS_OF)
    run_main(compose_sms, "--country", "TEST", "--audience", "all")
    # window covers D1's June warning and the October IPC transition; metrics are tested in test_backtest.py
    run_main(backtest, "--country", "TEST", "--from", "2024-05-01", "--to", "2024-10-31")


def snapshot():
    with TinyDB(db.DB_PATH) as t:
        return {name: sorted((json.dumps(dict(d), sort_keys=True) for d in t.table(name).all()))
                for name in ("signals", "features", "risk_assessment", "alerts")}


def test_full_pipeline_runs_end_to_end(mock_db, run_main, isolated):
    real_mtime = REAL_DB.stat().st_mtime if REAL_DB.exists() else None
    run_pipeline(run_main)
    snap = snapshot()
    assert all(snap[name] for name in snap), {k: len(v) for k, v in snap.items()}
    levels = {r["unit_id"]: r["level"] for r in db.query_risk(country="TEST")}
    assert levels == {"TEST-D1": 3, "TEST-D2": 1, "TEST-D3": 0, "TEST-D4": 0, "TEST-D5": 0}
    assert list(isolated.glob("backtest_TEST_*.json"))
    assert (isolated / "hysteresis_state_asof_20250115.json").exists()
    assert (REAL_DB.stat().st_mtime if REAL_DB.exists() else None) == real_mtime, "real DB must not be touched"


def test_alerts_written_to_db(mock_db, run_main):
    run_pipeline(run_main)
    alerts = db.query_alerts()
    assert len(alerts) == 6                                        # D1 (L3) + D2 (L1), 3 audiences each
    assert Counter(a["audience"] for a in alerts) == {"farmer": 2, "government": 2, "ngo": 2}
    assert {a["status"] for a in alerts} == {"mocked"}             # no API key in tests
    assert {a["unit_id"] for a in alerts} == {"TEST-D1", "TEST-D2"}
    risk_ids = {r["_doc_id"] for r in db.query_risk(country="TEST", level_ge=1)}
    assert {a["assessment_id"] for a in alerts} <= risk_ids


def test_idempotent_re_run(mock_db, run_main, isolated):
    run_pipeline(run_main)
    first = snapshot()
    state_file = isolated / "hysteresis_state_asof_20250115.json"
    first_state = json.loads(state_file.read_text())
    run_pipeline(run_main)
    assert snapshot() == first                                     # no duplicates, nothing changed
    # same hysteresis state; only `previous_level` (what the prior run concluded) is filled in now
    second_state = json.loads(state_file.read_text())
    strip = lambda st: {u: {k: v for k, v in s.items() if k != "previous_level"} for u, s in st.items()}  # noqa: E731
    assert strip(second_state) == strip(first_state)
    assert all(s["previous_level"] == first_state[u]["level"] for u, s in second_state.items())
