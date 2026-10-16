import math

import pandas as pd
import pytest

import db
from conftest import same_month_series, signal, synthetic_signals
from engineer_features import FLAG_THRESHOLD, compute_features, pit_cutoff

SCALE = 1.4826


def feature_for(rows, period_start):
    f = compute_features(rows)
    hit = f[f["period_start"] == period_start]
    assert len(hit) == 1, f"expected one feature for {period_start}, got {len(hit)}"
    return hit.iloc[0]


def test_z_score_computation():
    # same-month history 10, 12, 14 -> baseline 12, MAD 2; value 20 -> z = 8 / (2 * 1.4826)
    rows = same_month_series({2019: 10, 2020: 12, 2021: 14, 2022: 20})
    f = feature_for(rows, "2022-03-01")
    assert f["baseline"] == pytest.approx(12)
    assert f["mad"] == pytest.approx(2)
    assert f["z_score"] == pytest.approx(8 / (2 * SCALE), abs=1e-3)
    assert f["baseline_years"] == 3


def test_direction_of_risk_ndvi_lower_is_worse():
    # history 0, 0.5, -0.5 -> baseline 0, MAD 0.5; value -2 * scale -> z = -2 -> risk +2
    rows = same_month_series({2019: 0.0, 2020: 0.5, 2021: -0.5, 2022: -2 * 0.5 * SCALE},
                             signal_type="ndvi_z", source_id="ndvi_modis")
    f = feature_for(rows, "2022-03-01")
    assert f["z_score"] == pytest.approx(-2, abs=1e-3)
    assert f["risk_score"] == pytest.approx(2, abs=1e-3)
    assert f["source_family"] == "eo"


def test_direction_of_risk_price_higher_is_worse():
    rows = same_month_series({2019: 0.0, 2020: 0.5, 2021: -0.5, 2022: 2 * 0.5 * SCALE})
    f = feature_for(rows, "2022-03-01")
    assert f["z_score"] == pytest.approx(2, abs=1e-3)
    assert f["risk_score"] == pytest.approx(2, abs=1e-3)
    assert f["source_family"] == "market_prices"


@pytest.mark.parametrize("target_risk, expected_flag", [(0.5, 0), (1.0, 0), (1.49, 0), (1.51, 1), (3.0, 1)])
def test_threshold_flag_at_1_5(target_risk, expected_flag):
    rows = same_month_series({2019: 0.0, 2020: 0.5, 2021: -0.5, 2022: target_risk * 0.5 * SCALE})
    f = feature_for(rows, "2022-03-01")
    assert f["threshold_flag"] == expected_flag
    assert (f["risk_score"] > FLAG_THRESHOLD) == bool(expected_flag)


@pytest.mark.parametrize("n_prior_years, has_feature", [(0, False), (1, False), (2, False), (3, True), (4, True)])
def test_baseline_requires_3_observations(n_prior_years, has_feature):
    years = list(range(2022 - n_prior_years, 2023))
    rows = same_month_series({y: 10 + (y % 3) for y in years})
    f = compute_features(rows)
    got = f[f["period_start"] == "2022-03-01"] if len(f) else f
    assert (len(got) == 1) == has_feature
    if len(f):
        assert (f["baseline_years"] >= 3).all()


def test_point_in_time_no_lookahead(mock_db):
    as_of = "2024-06-01"
    f = compute_features(synthetic_signals(), as_of=as_of)
    assert len(f)
    assert f["period_start"].max() < "2024-06-01"
    assert not (f["period_start"] >= "2024-07-01").any(), "July 2024 must not be visible on 2024-06-01"
    # 1-month publication lag: May 2024 (ends 2024-05-31) is not published yet either
    assert f["period_start"].max() == "2024-04-01"
    # the DB query used by the CLI applies the same cutoff
    rows = db.query_signals(country="TEST", period_end_le=pit_cutoff(as_of))
    assert rows and max(r["period_end"] for r in rows) <= "2024-05-01"


def test_ipc_phase_never_in_features():
    f = compute_features(synthetic_signals())
    assert len(f) and "ipc_phase" not in set(f["signal_type"])
    only_ipc = [signal("TEST-D1", "ipc_phase", "ipc_phase", f"{y}-07-01", 3) for y in range(2018, 2025)]
    assert compute_features(only_ipc).empty


def test_sticky_signal_floor():
    # identical history -> MAD 0; a tiny move must not explode into a flag (scale floored at 0.10)
    rows = same_month_series({2019: 10.0, 2020: 10.0, 2021: 10.0, 2022: 10.05})
    f = feature_for(rows, "2022-03-01")
    assert f["mad"] == 0
    assert math.isfinite(f["z_score"])
    assert f["z_score"] == pytest.approx(0.5, abs=1e-3)
    assert f["threshold_flag"] == 0
