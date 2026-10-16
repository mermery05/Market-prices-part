import json

import pytest

from backtest import run_backtest
from conftest import IPC_TRANSITION, REAL_DB

WINDOW = ("2024-01-01", "2024-12-31")


@pytest.fixture
def report(mock_db, isolated):
    rep, path = run_backtest("TEST", *WINDOW, out_dir=isolated)
    return rep, path


def test_persistence_baseline_computed(report):
    rep, _ = report
    for key in ("model_metrics", "persistence_baseline", "model_metrics_level1", "persistence_baseline_level1"):
        assert set(rep[key]) >= {"hit_rate", "false_alarm_rate", "median_weeks_gained"}
    # persistence = the engine's level 4 weeks earlier, so it can only warn later, never earlier
    assert rep["persistence_baseline"]["median_weeks_gained"] < rep["model_metrics"]["median_weeks_gained"]


def test_hit_rate_in_range_0_1(report):
    rep, _ = report
    assert rep["n_phase_3_plus_transitions"] == 2                 # TEST-D1 and TEST-D2 on 2024-10-01
    assert set(rep["transitions"]) == {"TEST-D1", "TEST-D2"}
    for key in ("model_metrics", "persistence_baseline", "model_metrics_level1", "persistence_baseline_level1"):
        for rate in ("hit_rate", "false_alarm_rate"):
            v = rep[key][rate]
            assert v is None or 0.0 <= v <= 1.0
    # D1 gets a Warning (prices + drought), D2 only a Watch (prices alone)
    assert rep["model_metrics"]["hit_rate"] == pytest.approx(0.5)
    assert rep["model_metrics_level1"]["hit_rate"] == pytest.approx(1.0)


def test_median_weeks_gained_positive(report):
    rep, _ = report
    m = rep["model_metrics"]
    assert m["n_hits"] > 0
    assert m["median_weeks_gained"] > 0
    # March shock -> published end of April -> escalated after 2 periods -> warning in early June
    assert 15 <= m["median_weeks_gained"] <= (IPC_TRANSITION - IPC_TRANSITION.replace(month=3)).days / 7


def test_no_phase_3_plus_returns_zero_metrics(mock_db_no_phase3, isolated):
    """No Phase 3+ transition -> zero counts. Rates that divide by the number of transitions are
    undefined (None) rather than 0, so a missing ground truth is never reported as a 0% hit rate."""
    rep, _ = run_backtest("TEST", *WINDOW, out_dir=isolated)
    assert rep["n_phase_3_plus_transitions"] == 0 and rep["transitions"] == {}
    for key in ("model_metrics", "persistence_baseline", "model_metrics_level1", "persistence_baseline_level1"):
        m = rep[key]
        assert m["n_hits"] == 0
        assert m["hit_rate"] is None and m["median_weeks_gained"] is None
    # warnings were still issued (D1), and with IPC phase 2 throughout every one is a false alarm
    assert rep["model_metrics"]["n_warned_units"] == 1
    assert rep["model_metrics"]["false_alarm_rate"] == pytest.approx(1.0)


def test_backtest_writes_json_output(report, isolated):
    rep, path = report
    assert path.parent == isolated and path.name.startswith("backtest_TEST_") and path.suffix == ".json"
    on_disk = json.loads(path.read_text())
    assert on_disk["model_metrics"] == rep["model_metrics"]
    assert on_disk["country"] == "TEST" and on_disk["from"] == WINDOW[0] and on_disk["to"] == WINDOW[1]
    assert not str(path).startswith(str(REAL_DB.parent))
