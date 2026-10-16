"""
Shared fixtures for the engine-layer tests.

Everything runs against a temporary TinyDB file and a temporary output directory; API keys are removed and
network access is blocked, so no test can touch data/out, call an LLM or fetch anything.

Synthetic world (country "TEST", districts TEST-D1 .. TEST-D5), monthly 2019-01 .. 2024-12:
  history needs >= 3 earlier years of the same month for a baseline, so 2019-2021 are baseline-only years.
  Normal values wiggle by a fixed per-year offset (0, +1, -1, 0, +1, -1) -> MAD > 0, z-scores stay small.
  TEST-D1  from 2024-03: price shock (market_prices) + drought (eo); from 2024-06 also announcement >= 3 (text)
  TEST-D2  from 2024-03: price shock only (one family)
  TEST-D3..D5 normal throughout
  IPC (quarterly, ground truth): phase 2 everywhere, TEST-D1 and TEST-D2 move to phase 3 on 2024-10-01
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backtest  # noqa: E402
import compose_sms  # noqa: E402
import db  # noqa: E402
import fuse_risk  # noqa: E402
import ingest_prices  # noqa: E402

COUNTRY = "TEST"
DISTRICTS = [f"TEST-D{i}" for i in range(1, 6)]
NAMES = {u: f"District {u[-1]}" for u in DISTRICTS}
YEAR_OFFSET = {2019: 0, 2020: 1, 2021: -1, 2022: 0, 2023: 1, 2024: -1}
SHOCK_START = pd.Timestamp("2024-03-01")
ANNOUNCE_START = pd.Timestamp("2024-06-01")
IPC_TRANSITION = pd.Timestamp("2024-10-01")
REAL_DB = ROOT / "data" / "out" / "signal_observation.json"


def month_end(ts):
    return (pd.Timestamp(ts) + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d")


def signal(unit, source_id, signal_type, period_start, value, period_end=None, confidence=1.0):
    """One signal_observation document."""
    ps = pd.Timestamp(period_start)
    return {"unit_id": unit, "source_id": source_id, "signal_type": signal_type, "value": float(value),
            "unit_of_measure": "test", "period_start": ps.strftime("%Y-%m-%d"),
            "period_end": period_end or month_end(ps), "ingested_at": "2025-01-01T00:00:00+00:00",
            "confidence": confidence, "raw_ref": "synthetic"}


def same_month_series(values_by_year, signal_type="price_anom_real", source_id="wfp_hdx_test",
                      unit="TEST-D1", month=3):
    """Rows for one series, one value per year in the same calendar month."""
    return [signal(unit, source_id, signal_type, f"{y}-{month:02d}-01", v) for y, v in values_by_year.items()]


def synthetic_signals(ipc_phase3=True):
    rows = []
    for ps in pd.date_range("2019-01-01", "2024-12-01", freq="MS"):
        o = YEAR_OFFSET[ps.year]
        shocked = ps >= SHOCK_START
        for u in DISTRICTS:
            price = 40.0 if (u in ("TEST-D1", "TEST-D2") and shocked) else 5 + o
            ndvi = -3.0 if (u == "TEST-D1" and shocked) else 0.2 * o
            spi = -2.5 if (u == "TEST-D1" and shocked) else 0.2 * o
            ann = 4.0 if (u == "TEST-D1" and ps >= ANNOUNCE_START) else 1 + 0.5 * o
            rows += [
                signal(u, "wfp_hdx_test", "price_anom_real", ps, price),
                signal(u, "wfp_hdx_test", "data_freshness_share", ps, 0.9 + 0.05 * o),
                signal(u, "ndvi_modis", "ndvi_z", ps, ndvi),
                signal(u, "chirps", "spi_3m", ps, spi),
                signal(u, "announcement_news", "announcement_severity", ps, ann),
            ]
    for ps in pd.date_range("2023-01-01", "2024-10-01", freq="QS"):
        for u in DISTRICTS:
            phase = 3 if (ipc_phase3 and u in ("TEST-D1", "TEST-D2") and ps >= IPC_TRANSITION) else 2
            rows.append(signal(u, "ipc_phase", "ipc_phase", ps, phase,
                               period_end=month_end(ps + pd.DateOffset(months=2))))
    return rows


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Temp DB + output dir, no API keys, no network, a 'TEST' country. Returns the temp output dir."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)

    import requests

    def no_network(*args, **kwargs):
        raise RuntimeError("network access is blocked in tests")
    monkeypatch.setattr(requests.Session, "request", no_network)

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(db, "DB_PATH", out / "test_signal_observation.json")
    monkeypatch.setattr(db, "HYSTERESIS_PATH", out / "hysteresis_state.json")
    for mod in (ingest_prices, fuse_risk, backtest):
        monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setitem(ingest_prices.COUNTRIES, COUNTRY,
                        {"iso3": "TST", "name": "Testland", "level": "ADM2", "gazetteer": {}, "fews": False,
                         "drop_currencies": [], "hdx_slug": None})
    monkeypatch.setattr(compose_sms, "load_units", lambda cc, cfg: (
        pd.DataFrame({"unit_id": DISTRICTS, "shapeName": [NAMES[u] for u in DISTRICTS]}), None))
    return out


@pytest.fixture
def mock_db(isolated):
    """Temp TinyDB populated with the synthetic signals (IPC transitions to phase 3 in D1, D2)."""
    db.upsert_signals(synthetic_signals())
    return str(db.DB_PATH)


@pytest.fixture
def mock_db_no_phase3(isolated):
    """Same world, but IPC never reaches phase 3."""
    db.upsert_signals(synthetic_signals(ipc_phase3=False))
    return str(db.DB_PATH)


@pytest.fixture
def run_main(monkeypatch):
    """Run a script's main() with CLI args; a SystemExit(0) counts as success."""
    def _run(module, *args):
        monkeypatch.setattr(sys, "argv", [module.__name__, *args])
        try:
            module.main()
        except SystemExit as e:
            assert e.code in (0, None), f"{module.__name__} exited with {e.code}"
    return _run
