import json

import pandas as pd
import pytest

import db
import engineer_features
import fuse_risk
from fuse_risk import apply_hysteresis, assess, confidence, p_shortage, raw_levels

FAMILY = {"price_anom_real": "market_prices", "price_velocity_anom": "market_prices", "ndvi_z": "eo",
          "spi_3m": "eo", "announcement_severity": "text", "ipc_phase": "ipc", "data_freshness_share": "market_prices"}


def feat(signal_type, risk, unit="TEST-D1", period="2024-03-01", value=None, baseline_years=5, flag=None):
    """A minimal feature document (z_score = risk_score; direction handled by the feature step)."""
    return {"unit_id": unit, "signal_type": signal_type, "period_start": period, "period_end": period,
            "value": risk if value is None else value, "baseline": 0.0, "mad": 1.0, "baseline_years": baseline_years,
            "z_score": risk, "risk_score": risk, "threshold_flag": int(risk > 1.5) if flag is None else flag,
            "source_family": FAMILY[signal_type], "source_id": "test", "n_source_series": 1}


def level_of(features):
    lv = raw_levels(pd.DataFrame(features))
    assert len(lv) == 1
    return lv.iloc[0]


def run_hysteresis(raws, periods=None):
    periods = periods or [str(p.date()) for p in pd.date_range("2024-01-01", periods=len(raws), freq="MS")]
    df = pd.DataFrame({"unit_id": "TEST-D1", "period_start": periods, "raw_level": raws})
    out, state = apply_hysteresis(df)
    return out["level"].tolist(), state


# ----------------------------------------------------------------- raw levels
def test_level_0_no_thresholds():
    r = level_of([feat("price_anom_real", 1.0), feat("ndvi_z", 1.4), feat("announcement_severity", -3.0)])
    assert r["raw_level"] == 0 and r["n_sources_agreeing"] == 0


def test_level_1_one_source_family():
    # two flagged signals, but both market_prices -> still one family
    r = level_of([feat("price_anom_real", 3.0), feat("price_velocity_anom", 2.5), feat("ndvi_z", 0.2)])
    assert r["raw_level"] == 1 and r["n_sources_agreeing"] == 1
    assert r["families_flagged"] == ["market_prices"]


def test_level_2_two_source_families():
    r = level_of([feat("price_anom_real", 3.0), feat("ndvi_z", 2.0)])
    assert r["raw_level"] == 2 and r["n_sources_agreeing"] == 2


@pytest.mark.parametrize("announcement, expected", [
    (None, 2),          # no announcement at all
    (2.0, 2),           # announcement present but severity < 3
    (3.0, 3),           # severity exactly 3 -> emergency
    (4.0, 3),
])
def test_level_3_requires_announcement(announcement, expected):
    fs = [feat("price_anom_real", 3.0), feat("ndvi_z", 2.0)]
    if announcement is not None:
        fs.append(feat("announcement_severity", 0.5, value=announcement))   # not itself flagged
    assert level_of(fs)["raw_level"] == expected


def test_level_3_needs_level_2_first():
    # a severe announcement on top of a single flagged family is not an emergency
    r = level_of([feat("price_anom_real", 3.0), feat("announcement_severity", 0.5, value=5.0)])
    assert r["raw_level"] == 1


@pytest.mark.parametrize("others, expected", [([], 0), (["price_anom_real"], 1), (["price_anom_real", "ndvi_z"], 2)])
def test_ipc_phase_does_not_trigger_fusion(others, expected):
    fs = [feat("ipc_phase", 9.0, flag=1)] + [feat(s, 3.0) for s in others]
    lv = raw_levels(pd.DataFrame(fs))
    got = lv.iloc[0]["raw_level"] if len(lv) else 0
    assert got == expected
    if len(lv):
        assert "ipc" not in lv.iloc[0]["families_flagged"]


# ----------------------------------------------------------------- hysteresis
@pytest.mark.parametrize("raws, expected", [
    ([0, 2, 2], [0, 0, 2]),
    ([0, 1, 1], [0, 0, 1]),
    ([0, 2, 0, 2, 2], [0, 0, 0, 0, 2]),     # interrupted run restarts the count
])
def test_escalation_requires_2_consecutive_runs(raws, expected):
    assert run_hysteresis(raws)[0] == expected


@pytest.mark.parametrize("raws, expected", [
    ([2, 2, 0, 0, 0], [0, 2, 2, 2, 0]),
    ([2, 2, 0, 0, 2, 0, 0, 0], [0, 2, 2, 2, 2, 2, 2, 0]),   # a relapse resets the de-escalation count
])
def test_de_escalation_requires_3_consecutive_runs(raws, expected):
    assert run_hysteresis(raws)[0] == expected


def test_hysteresis_state_persisted(mock_db, run_main):
    run_main(engineer_features, "--country", "TEST")
    run_main(fuse_risk, "--country", "TEST")
    first = db.load_hysteresis()
    assert db.HYSTERESIS_PATH.exists()
    assert set(first) == {f"TEST-D{i}" for i in range(1, 6)}
    assert first["TEST-D1"]["level"] == 3 and first["TEST-D2"]["level"] == 1 and first["TEST-D5"]["level"] == 0
    assert all(s["previous_level"] is None for s in first.values())

    run_main(fuse_risk, "--country", "TEST")
    second = db.load_hysteresis()
    for unit, s in second.items():
        assert s["previous_level"] == first[unit]["level"]       # read back from the file
        assert s["level"] == first[unit]["level"]                # same evidence -> same level


@pytest.mark.parametrize("gap_days, expected_last", [(31, 2), (90, 2), (91, 0), (180, 0)])
def test_hysteresis_resets_on_long_gap(gap_days, expected_last):
    # L2 established, then a single L0 period: held at L2 after a normal gap, reset after > 90 days
    p3 = (pd.Timestamp("2024-02-01") + pd.Timedelta(days=gap_days)).strftime("%Y-%m-%d")
    levels, _ = run_hysteresis([2, 2, 0], periods=["2024-01-01", "2024-02-01", p3])
    assert levels[:2] == [0, 2]
    assert levels[2] == expected_last


# ----------------------------------------------------------------- p_shortage
def test_p_shortage_level_0():
    assert p_shortage(0, 0) == pytest.approx(0.05)


@pytest.mark.parametrize("n, expected", [(1, 0.20), (2, 0.30), (3, 0.40)])
def test_p_shortage_level_1(n, expected):
    assert p_shortage(1, n) == pytest.approx(expected)


@pytest.mark.parametrize("n, expected", [(2, 0.50), (3, 0.60), (4, 0.70)])
def test_p_shortage_level_2(n, expected):
    assert p_shortage(2, n) == pytest.approx(expected)


@pytest.mark.parametrize("n", [2, 3, 5])
def test_p_shortage_level_3(n):
    assert p_shortage(3, n) == pytest.approx(0.80)


# ----------------------------------------------------------------- confidence
def test_confidence_increases_with_more_sources():
    assert confidence(2, 5, 1.0) > confidence(1, 5, 1.0) > confidence(0, 5, 1.0)
    assert confidence(1, 5, 1.0) == pytest.approx(0.25)


def test_confidence_drops_with_short_baseline():
    assert confidence(2, 2, 1.0) < confidence(2, 5, 1.0)
    assert confidence(2, 5, 1.0) == confidence(2, 9, 1.0)      # capped at 5 years


@pytest.mark.parametrize("n, years, fresh", [(4, 5, 1.0), (10, 50, 1.0), (10, 50, 3.0)])
def test_confidence_clipped_at_1_0(n, years, fresh):
    assert 0.0 <= confidence(n, years, fresh) <= 1.0


# ----------------------------------------------------------------- drivers
def test_drivers_top_3_by_risk_score():
    fs = [feat("price_anom_real", 0.5), feat("ndvi_z", 3.0), feat("spi_3m", 2.0),
          feat("announcement_severity", 4.0), feat("price_velocity_anom", 1.0)]
    drivers = level_of(fs)["drivers"]
    assert [d["signal_type"] for d in drivers] == ["announcement_severity", "ndvi_z", "spi_3m"]


def test_drivers_format():
    fs = [feat("price_anom_real", 3.0, value=40.0), feat("ndvi_z", 2.0, value=-3.0)]
    _, latest, _ = assess(fs, as_of="2024-04-15", run_at="20240415T120000Z")
    drivers = json.loads(latest.iloc[0]["drivers_json"])
    assert 1 <= len(drivers) <= 3
    for d in drivers:
        assert set(d) == {"signal_type", "value", "z_score"}
        assert isinstance(d["signal_type"], str)
        assert isinstance(d["value"], float) and isinstance(d["z_score"], float)
    assert drivers[0] == {"signal_type": "price_anom_real", "value": 40.0, "z_score": 3.0}
