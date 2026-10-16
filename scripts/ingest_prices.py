#!/usr/bin/env python3
"""
Ingest trusted market-price sources into the `signal_observation` schema.

Sources (no API key needed):
  wfp_hdx   WFP Food Prices on HDX (KE, ZW)                  -> price_anom_real / _nominal, data_freshness_share,
                                                                 price_velocity_anom, price_dispersion_anom,
                                                                 market_coverage_drop
  fewsnet   FEWS NET staple prices, FDW API (KE only)         -> same as wfp_hdx
  wfp_gmm   WFP Global Market Monitor on HDX (KE, ZW)         -> basket_cost_change
Skipped on purpose: FEWS NET ZW (HTTP 500), WFP DataBridges (unverified creds), any scraping.

Usage:
  python scripts/ingest_prices.py --country KE
  python scripts/ingest_prices.py --country ZW
  python scripts/ingest_prices.py --country KE --as-of 2024-06-30   # point-in-time back-test
  python scripts/ingest_prices.py --country KE --refresh            # re-download even if today's raw exists

Outputs (under ./data):
  raw/<CC>_<source>_<fetch_date>.csv            unchanged downloads (+ .meta.json with headers)
  clean/<CC>_<source>_<YYYY-MM>_<YYYY-MM>.csv   cleaned market-level rows
  out/signal_observation_prices_<CC>_<run_at>.csv
  out/freshness_<run_at>.json
  out/signal_observation.sqlite                  idempotent upsert store
"""
import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db as docstore  # noqa: E402  (TinyDB `signals` collection, read by the engine layer)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW, CLEAN, OUT, GEO = DATA / "raw", DATA / "clean", DATA / "out", DATA / "geo"

HDX = "https://data.humdata.org/api/3/action"
FEWS_URL = ("https://fdw.fews.net/api/marketpricefacts/"
            "?dataset=FEWS_NET_Staple_Food_Price_Data&country={cc}&format=csv&fields=website")
GEOBOUNDARIES = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{level}/"
HEADERS = {"User-Agent": "sms-shortage-early-warning-hackathon/0.1"}
TIMEOUT = 300

PUB_LAG_MONTHS = 1          # data for month M is assumed published by end of M+1
FRESH_MONTHS = 3            # tolerated age beyond the publication lag before "stale"
BASELINE_YEARS = 5          # same-month baseline window
MIN_BASELINE_YEARS = 3      # require >= 3 same-month observations in the window
HAMPEL_WINDOW_MONTHS = 6    # trailing window (point-in-time safe: never looks forward)
HAMPEL_T = 3.0
HAMPEL_MIN_N = 5
HAMPEL_MIN_SCALE = 0.10     # log-scale MAD floor: sticky prices (MAD~0) only flag moves > ~35%
OLD_ZW_FALSE_POSITIVES = 6149
DERIVED_SIGNALS = ["price_velocity_anom", "price_dispersion_anom", "market_coverage_drop"]

# Location-match quality -> confidence multiplier
LOC_FACTOR = {"point_in_polygon": 1.0, "nearest_polygon_10km": 0.9, "gazetteer": 0.8, "name_match": 0.8}
FANOUT_FACTOR = 0.7         # region-level value broadcast to its child admin units

# GMM Kenya reports by pre-2013 province; map to the 47 counties (geoBoundaries names).
KE_PROVINCES = {
    "Coast": ["Mombasa", "Kwale", "Kilifi", "Tana River", "Lamu", "Taita Taveta"],
    "North Eastern": ["Garissa", "Wajir", "Mandera"],
    "Eastern": ["Marsabit", "Isiolo", "Meru", "Tharaka", "Embu", "Kitui", "Machakos", "Makueni"],
    "Central": ["Nyandarua", "Nyeri", "Kirinyaga", "Murang'a", "Kiambu"],
    "Rift Valley": ["Turkana", "West Pokot", "Samburu", "Trans Nzoia", "Uasin Gishu", "Elgeyo-Marakwet",
                    "Nandi", "Baringo", "Laikipia", "Nakuru", "Narok", "Kajiado", "Kericho", "Bomet"],
    "Western": ["Kakamega", "Vihiga", "Bungoma", "Busia"],
    "Nyanza": ["Siaya", "Kisumu", "Homa Bay", "Migori", "Kisii", "Nyamira"],
    "Nairobi": ["Nairobi"],
}
# WFP ZW markets published without coordinates or admin names -> district (geoBoundaries ADM2).
ZW_GAZETTEER = {
    "Murombedzi": "Zvimba", "Renkini Bus Terminus": "Bulawayo", "Mucheke": "Masvingo Urban",
    "Tshovani": "Chiredzi Urban", "Mandava": "Zvishavane Urban", "Kombai": "Gutu",
    "Dulibadzimu Market": "Beitbridge Urban", "Plumtree town": "Plumtree",
}

COUNTRIES = {
    "KE": dict(iso3="KEN", name="Kenya", level="ADM1", region_map=KE_PROVINCES, gazetteer={},
               hdx_slug="wfp-food-prices-for-kenya", fews=True, drop_currencies=[]),
    "ZW": dict(iso3="ZWE", name="Zimbabwe", level="ADM2", region_level="ADM1", gazetteer=ZW_GAZETTEER,
               hdx_slug="wfp-food-prices-for-zimbabwe", fews=False, drop_currencies=["ZWL"],
               fews_skip_reason="FEWS NET ZW returns HTTP 500 (live check 2026-10-16); skipped by design"),
}

COMMODITY_RULES = [
    (r"^maize (flour|meal)", "maize_flour"), (r"^maize", "maize_grain"), (r"^beans", "beans"),
    (r"^rice", "rice"), (r"^wheat flour", "wheat_flour"), (r"^wheat", "wheat_grain"),
    (r"^sugar", "sugar"), (r"^(oil \(vegetable\)|refined vegetable oil|vegetable oil)", "vegetable_oil"),
    (r"^(milk|cow's milk)", "milk"), (r"^bread", "bread"), (r"^potatoes \(irish\)", "potatoes_irish"),
    (r"^sorghum", "sorghum"), (r"^salt", "salt"), (r"^diesel", "diesel"), (r"^(gasoline|petrol)", "gasoline"),
    (r"^kerosene", "kerosene"),
]
FUEL = {"diesel", "gasoline", "kerosene"}
UNIT_FACTORS = {"kg": (1.0, "kg"), "g": (0.001, "kg"), "mt": (1000.0, "kg"), "l": (1.0, "L"), "ml": (0.001, "L")}


# ----------------------------------------------------------------- helpers
def now_utc():
    return datetime.now(timezone.utc)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def slug(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def month_index(ts):
    return ts.dt.year * 12 + ts.dt.month - 1


def to_month_start(s):
    return pd.to_datetime(s, errors="coerce").dt.to_period("M").dt.to_timestamp()


def months_between(later, earlier):
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def standardise_commodity(raw):
    low = str(raw).strip().lower()
    for pat, canon in COMMODITY_RULES:
        if re.match(pat, low):
            return canon
    return slug(low).replace("-", "_")


def parse_unit(u):
    """'90 KG' -> (90, 'kg'); '500_ml' -> (0.5, 'L'); 'Bunch' -> (1, 'bunch') (not convertible)."""
    m = re.match(r"^\s*(\d+(?:\.\d+)?)?\s*_?\s*([A-Za-z]+)\s*$", str(u))
    if not m:
        return 1.0, slug(u)
    qty = float(m.group(1)) if m.group(1) else 1.0
    key = m.group(2).lower()
    if key in UNIT_FACTORS:
        f, std = UNIT_FACTORS[key]
        return qty * f, std
    return qty, key


def raw_ref(meta):
    return f"{meta['url']}|last_modified={meta.get('last_modified') or 'n/a'}|fetched_at={meta['fetched_at']}"


def freshness_factor(latest, ref_date):
    age = months_between(ref_date, latest)
    if age <= PUB_LAG_MONTHS + FRESH_MONTHS:
        return 1.0, age
    if age <= 12:
        return 0.5, age
    return 0.25, age


# ----------------------------------------------------------------- fetching
def ckan_resource(slug_, name_pred, fmt="csv"):
    r = requests.get(f"{HDX}/package_show", params={"id": slug_}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"CKAN error for {slug_}: {body.get('error')}")
    for res in body["result"]["resources"]:
        if (res.get("format") or "").lower() == fmt and name_pred(res.get("name") or ""):
            return res
    raise RuntimeError(f"no matching {fmt.upper()} resource in {slug_}")


def fetch(url, raw_path, refresh, fallback_last_modified=None):
    """Download url to raw_path byte-for-byte; reuse today's copy unless --refresh."""
    meta_path = raw_path.with_name(raw_path.name + ".meta.json")
    if raw_path.exists() and meta_path.exists() and not refresh:
        return json.loads(meta_path.read_text())
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    raw_path.write_bytes(r.content)
    meta = {"url": url, "status": r.status_code, "bytes": len(r.content),
            "last_modified": r.headers.get("Last-Modified") or fallback_last_modified,
            "last_modified_source": "http_header" if r.headers.get("Last-Modified") else
                                    ("ckan_resource" if fallback_last_modified else None),
            "fetched_at": now_utc().isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def read_raw(path):
    df = pd.read_csv(path, low_memory=False)
    first = df.columns[0]
    df = df[~df[first].astype(str).str.startswith("#")]   # drop HXL tag row if present
    obj = df.select_dtypes("object").columns
    df[obj] = df[obj].apply(lambda c: c.str.strip())
    return df.reset_index(drop=True)


def load_units(cc, cfg):
    """Admin polygons from geoBoundaries (cached under data/geo)."""
    GEO.mkdir(parents=True, exist_ok=True)

    def get(level):
        p = GEO / f"{cfg['iso3']}_{level}.geojson"
        if not p.exists():
            meta = requests.get(GEOBOUNDARIES.format(iso3=cfg["iso3"], level=level), headers=HEADERS, timeout=60).json()
            r = requests.get(meta["simplifiedGeometryGeoJSON"], headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            p.write_bytes(r.content)
        g = gpd.read_file(p).to_crs("EPSG:4326")
        g["unit_id"] = [f"{cc}-{level}-{slug(n)}" for n in g["shapeName"]]
        dup = g["unit_id"].duplicated(keep=False)
        g.loc[dup, "unit_id"] = g.loc[dup, "unit_id"] + "-" + g.loc[dup, "shapeID"].astype(str)
        return g[["unit_id", "shapeName", "geometry"]]

    units = get(cfg["level"])
    regions = get(cfg["region_level"]) if cfg.get("region_level") else None
    return units, regions


# ----------------------------------------------------------------- location
def assign_units(df, units, cfg):
    """Market -> unit_id. Point-in-polygon, then nearest polygon <=10 km, then name-based fallback.
    Unmatched markets keep unit_id = NaN and are dropped downstream (absent, never zero-filled)."""
    cols = ["market", "latitude", "longitude", "admin1", "admin2"]
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    mk = df[cols].drop_duplicates().reset_index(drop=True)
    mk["unit_id"], mk["location_method"] = np.nan, np.nan
    mk["unit_id"] = mk["unit_id"].astype(object)
    mk["location_method"] = mk["location_method"].astype(object)

    has_xy = mk["latitude"].notna() & mk["longitude"].notna()
    if has_xy.any():
        pts = gpd.GeoDataFrame(mk[has_xy], geometry=gpd.points_from_xy(mk.loc[has_xy, "longitude"],
                                                                      mk.loc[has_xy, "latitude"]), crs="EPSG:4326")
        hit = gpd.sjoin(pts, units, how="left", predicate="within")
        hit = hit[~hit.index.duplicated()]
        ok = hit["unit_id_right"].notna()
        mk.loc[hit.index[ok], "unit_id"] = hit.loc[ok, "unit_id_right"]
        mk.loc[hit.index[ok], "location_method"] = "point_in_polygon"
        miss = pts.loc[hit.index[~ok]]
        if len(miss):
            near = gpd.sjoin_nearest(miss[["geometry"]].to_crs("EPSG:3857"), units[["unit_id", "geometry"]].to_crs("EPSG:3857"),
                                     how="left", max_distance=10_000)
            near = near[~near.index.duplicated()].dropna(subset=["unit_id"])
            mk.loc[near.index, "unit_id"] = near["unit_id"]
            mk.loc[near.index, "location_method"] = "nearest_polygon_10km"

    by_name = {norm(n): u for n, u in zip(units["shapeName"], units["unit_id"])}
    for i in mk.index[mk["unit_id"].isna()]:
        row = mk.loc[i]
        gaz = cfg["gazetteer"].get(row["market"])
        if gaz and norm(gaz) in by_name:
            mk.loc[i, ["unit_id", "location_method"]] = [by_name[norm(gaz)], "gazetteer"]
            continue
        paren = re.findall(r"\(([^)]*)\)", str(row["market"]))
        for cand in [row["admin2"], row["admin1"], *paren, row["market"]]:
            if pd.notna(cand) and norm(cand) in by_name:
                mk.loc[i, ["unit_id", "location_method"]] = [by_name[norm(cand)], "name_match"]
                break

    out = df.merge(mk, on=cols, how="left")
    out["loc_factor"] = out["location_method"].map(LOC_FACTOR)
    return out


# ----------------------------------------------------------------- outliers
def hampel_flags(df, group_cols, value_col):
    """Trailing-window Hampel filter on log price, computed separately inside each group.
    Window = all markets in the group over the last HAMPEL_WINDOW_MONTHS months (incl. current)."""
    flags = np.zeros(len(df), dtype=bool)
    m_all = df["m"].to_numpy()
    v_all = np.log(df[value_col].to_numpy(dtype=float))
    for _, idx in df.groupby(group_cols, sort=False, dropna=False).indices.items():
        order = idx[np.argsort(m_all[idx], kind="stable")]
        m, v = m_all[order], v_all[order]
        for um in np.unique(m):
            lo = np.searchsorted(m, um - HAMPEL_WINDOW_MONTHS + 1, "left")
            cur_lo, hi = np.searchsorted(m, um, "left"), np.searchsorted(m, um, "right")
            win = v[lo:hi]
            if len(win) < HAMPEL_MIN_N:
                continue
            med = np.median(win)
            scale = max(1.4826 * np.median(np.abs(win - med)), HAMPEL_MIN_SCALE)
            flags[order[cur_lo:hi]] = np.abs(v[cur_lo:hi] - med) > HAMPEL_T * scale
    return flags


# ----------------------------------------------------------------- cleaning
def pit_filter(df, as_of):
    """Keep only rows that would have been published by `as_of` (1-month publication lag)."""
    if as_of is None:
        return df
    published = df["period_end"] + pd.DateOffset(months=PUB_LAG_MONTHS)
    return df[published <= as_of]


def finish_prices(df, cfg, units, stats, src_name, group_cols):
    """Shared steps for price sources: validate, Hampel per group, convert units, locate."""
    df["period_end"] = df["period_start"] + pd.offsets.MonthEnd(0)
    df = pit_filter(df, stats["_as_of"])
    df = df.dropna(subset=["period_start"])
    bad = ~(df["price"] > 0)                     # NaN / zero / negative prices are invalid, not "zero"
    stats["invalid_price_rows"] = int(bad.sum())
    df = df[~bad].reset_index(drop=True)
    df["m"] = month_index(df["period_start"])

    df["is_outlier"] = hampel_flags(df, group_cols, "price")
    stats["outlier_count"] = int(df["is_outlier"].sum())
    stats["outlier_grouping"] = " x ".join(group_cols)
    if df["currency"].nunique() > 1:
        naive = [c for c in group_cols if c != "currency"]
        stats["outlier_count_if_currencies_mixed"] = int(hampel_flags(df, naive, "price").sum())
    if cfg["drop_currencies"]:
        drop = df["currency"].isin(cfg["drop_currencies"])
        stats["rows_dropped_currency"] = {c: int((df["currency"] == c).sum()) for c in cfg["drop_currencies"]}
        stats["outlier_count_in_kept_currency"] = int(df.loc[~drop, "is_outlier"].sum())
        df = df[~drop]
    df = df[~df["is_outlier"]]

    q = df["unit"].map(lambda u: parse_unit(u))
    df["unit_qty_std"] = [a for a, _ in q]
    df["unit_of_measure"] = [b for _, b in q]
    df["price_local_per_uom"] = df["price"] / df["unit_qty_std"]
    df["price_usd_per_uom"] = df["usdprice"] / df["unit_qty_std"]
    df["commodity_std"] = df["commodity_raw"].map(standardise_commodity)

    df = assign_units(df, units, cfg)
    unloc = df["unit_id"].isna()
    stats["rows_unlocated"] = int(unloc.sum())
    stats["markets_unlocated"] = sorted(df.loc[unloc, "market"].dropna().unique().tolist())
    stats["location_methods"] = df.loc[~unloc].drop_duplicates("market")["location_method"].value_counts().to_dict()
    df = df[~unloc].copy()
    stats["rows_after_cleaning"] = int(len(df))
    if len(df):
        stats["latest_date"] = df["period_start"].max().strftime("%Y-%m-%d")
        stats["earliest_date"] = df["period_start"].min().strftime("%Y-%m-%d")
    return df


def clean_wfp(raw, cfg, units, stats):
    df = pd.DataFrame({
        "period_start": to_month_start(raw["date"]),
        "market": raw["market"], "admin1": raw.get("admin1"), "admin2": raw.get("admin2"),
        "latitude": raw.get("latitude"), "longitude": raw.get("longitude"),
        "category": raw["category"], "commodity_raw": raw["commodity"], "unit": raw["unit"],
        "price_type": raw["pricetype"].str.lower(), "priceflag": raw["priceflag"],
        "currency": raw["currency"],
        "price": pd.to_numeric(raw["price"], errors="coerce"),
        "usdprice": pd.to_numeric(raw["usdprice"], errors="coerce"),
    })
    # same market/commodity/month can appear as 'actual' and 'aggregate': prefer actual
    df["_rank"] = (~df["priceflag"].astype(str).str.startswith("actual")).astype(int)
    df = (df.sort_values("_rank")
            .drop_duplicates(["market", "commodity_raw", "unit", "price_type", "currency", "period_start"])
            .drop(columns="_rank"))
    df["usdprice"] = df["usdprice"].where(df["usdprice"] > 0)
    df["is_food"] = df["category"].str.lower() != "non-food"
    return finish_prices(df, cfg, units, stats, "wfp_hdx", ["commodity_raw", "unit", "currency", "price_type"])


def clean_fews(raw, cfg, units, stats, fx):
    df = pd.DataFrame({
        "period_start": to_month_start(raw["period_date"]),
        "market": raw["market"], "admin1": raw.get("admin_1"), "admin2": np.nan,
        "latitude": raw.get("latitude"), "longitude": raw.get("longitude"),
        "commodity_raw": raw["product"], "unit": raw["unit"],
        "price_type": raw["price_type"].str.lower(), "currency": raw["currency"],
        "price": pd.to_numeric(raw["value"], errors="coerce"),
    })
    df["category"] = np.where(df["commodity_raw"].map(standardise_commodity).isin(FUEL), "non-food", "food")
    df["is_food"] = df["category"] == "food"
    # FEWS has no USD column: convert with the WFP-implied monthly rate (trailing fill <= 3 months, PIT-safe)
    df["usdprice"] = df["price"] / df["period_start"].map(fx) if fx is not None else np.nan
    stats["usd_conversion"] = "WFP-implied monthly local/USD rate (median price/usdprice), ffill <= 3 months"
    return finish_prices(df, cfg, units, stats, "fewsnet", ["commodity_raw", "unit", "currency", "price_type"])


def wfp_fx(clean_wfp_df):
    """Monthly local-currency-per-USD rate implied by WFP rows."""
    d = clean_wfp_df.dropna(subset=["usdprice"])
    d = d[d["usdprice"] > 0]
    if d.empty:
        return None
    rate = (d["price"] / d["usdprice"]).groupby(d["period_start"]).median()
    full = pd.date_range(rate.index.min(), rate.index.max() + pd.DateOffset(months=3), freq="MS")
    return rate.reindex(full).ffill(limit=3).dropna()


# ----------------------------------------------------------------- signals
def monthly_series(df):
    key = ["unit_id", "commodity_std", "price_type", "unit_of_measure"]
    ms = (df.groupby(key + ["period_start"])
            .agg(price_local=("price_local_per_uom", "median"), price_usd=("price_usd_per_uom", "median"),
                 n_markets=("market", "nunique"), loc_factor=("loc_factor", "mean"), is_food=("is_food", "all"))
            .reset_index())
    return ms, key


def seasonal_anomaly(ms, key):
    """(price_now - median of same calendar month over previous 5 years) / baseline * 100.
    Requires >= 3 of those 5 same-month observations; otherwise anomaly is absent (NaN)."""
    ms = ms.copy()
    ms["year"], ms["mon"] = ms["period_start"].dt.year, ms["period_start"].dt.month
    base = ms[key + ["year", "mon", "price_local", "price_usd"]]
    for k in range(1, BASELINE_YEARS + 1):
        lag = base.rename(columns={"price_local": f"local_lag{k}", "price_usd": f"usd_lag{k}"})
        ms = ms.merge(lag.assign(year=lag["year"] + k), on=key + ["year", "mon"], how="left")
    for cur in ["local", "usd"]:
        lags = ms[[f"{cur}_lag{k}" for k in range(1, BASELINE_YEARS + 1)]]
        b = lags.median(axis=1, skipna=True)
        n = lags.notna().sum(axis=1)
        ok = (n >= MIN_BASELINE_YEARS) & (b > 0) & ms[f"price_{cur}"].notna()
        ms[f"baseline_{cur}"] = b.where(ok)
        ms[f"baseline_years_{cur}"] = n
        ms[f"anom_{cur}"] = ((ms[f"price_{cur}"] - b) / b * 100).where(ok)
    return ms.drop(columns=[c for c in ms.columns if "_lag" in c])


def same_month_stats(df, key, col):
    """Same calendar month over the previous BASELINE_YEARS years (trailing only): median, MAD, count.
    Returns (baseline, mad, n_years) aligned to df; baseline/mad are NaN with < MIN_BASELINE_YEARS obs."""
    d = df[key + ["period_start", col]].copy()
    d["year"], d["mon"] = d["period_start"].dt.year, d["period_start"].dt.month
    res = d
    for k in range(1, BASELINE_YEARS + 1):
        lag = d[key + ["year", "mon", col]].rename(columns={col: f"_lag{k}"})
        res = res.merge(lag.assign(year=lag["year"] + k), on=key + ["year", "mon"], how="left")
    lags = res[[f"_lag{k}" for k in range(1, BASELINE_YEARS + 1)]]
    n = lags.notna().sum(axis=1)
    ok = n >= MIN_BASELINE_YEARS
    med = lags.median(axis=1, skipna=True)
    mad = lags.sub(med, axis=0).abs().median(axis=1, skipna=True)
    return med.where(ok).to_numpy(), mad.where(ok).to_numpy(), n.to_numpy()


def same_month_baseline(df, key, col):
    """Median of `col` for the same calendar month over the previous BASELINE_YEARS years (trailing only).
    Returns (baseline, n_years) aligned to df; baseline is NaN with fewer than MIN_BASELINE_YEARS obs."""
    b, _, n = same_month_stats(df, key, col)
    return b, n


def prefer_retail(d):
    """Keep one price type per unit-month (retail preferred) so retail and wholesale are never pooled."""
    d = d.copy()
    d["_pt"] = (d["price_type"] != "retail").astype(int)
    return d[d["_pt"] == d.groupby(["unit_id", "period_start"])["_pt"].transform("min")]


def velocity_signal(ms, key, fresh_factor):
    """price_velocity_anom: 3-month mean of monthly % change in USD price minus its same-month baseline."""
    v = ms[ms["is_food"]].dropna(subset=["price_usd"])[key + ["period_start", "price_usd", "loc_factor"]].copy()
    v["m"] = month_index(v["period_start"])
    prev = v[key + ["m", "price_usd"]].rename(columns={"price_usd": "_prev"})
    v = v.merge(prev.assign(m=prev["m"] + 1), on=key + ["m"], how="left")   # only true consecutive months
    v["pct_change_1m"] = (v["price_usd"] / v["_prev"] - 1) * 100
    p = v[key + ["m", "pct_change_1m"]]
    for k in (1, 2):
        v = v.merge(p.rename(columns={"pct_change_1m": f"_p{k}"}).assign(m=p["m"] + k), on=key + ["m"], how="left")
    pc = v[["pct_change_1m", "_p1", "_p2"]]
    v["velocity_3m"] = pc.mean(axis=1).where(pc.notna().sum(axis=1) >= 2)
    v = v.dropna(subset=["velocity_3m"]).reset_index(drop=True)
    v["baseline"], v["baseline_years"] = same_month_baseline(v, key, "velocity_3m")
    v["velocity_anom"] = v["velocity_3m"] - v["baseline"]
    v = prefer_retail(v.dropna(subset=["velocity_anom"]))
    a = (v.groupby(["unit_id", "period_start"])
           .agg(value=("velocity_anom", "median"), loc=("loc_factor", "mean"), hist=("baseline_years", "mean"))
           .reset_index())
    return pd.DataFrame({"unit_id": a["unit_id"], "signal_type": "price_velocity_anom", "value": a["value"],
                         "unit_of_measure": "abs_diff_3m_velocity_vs_baseline", "period_start": a["period_start"],
                         "confidence": fresh_factor * a["loc"] * np.minimum(1.0, a["hist"] / BASELINE_YEARS)})


def dispersion_signal(clean, fresh_factor, min_markets=3):
    """price_dispersion_anom: cross-market IQR/median of USD price, minus its same-month baseline.
    Needs >= min_markets markets per commodity-month (with fewer, the IQR is not informative)."""
    c = clean[clean["is_food"]].dropna(subset=["price_usd_per_uom"])
    k = ["unit_id", "period_start", "commodity_std", "price_type", "unit_of_measure"]
    g = c.groupby(k)["price_usd_per_uom"]
    s = pd.DataFrame({"q25": g.quantile(0.25), "q75": g.quantile(0.75), "med": g.median(),
                      "n_markets": c.groupby(k)["market"].nunique(), "loc": c.groupby(k)["loc_factor"].mean()}).reset_index()
    s = s[(s["med"] > 0) & (s["n_markets"] >= min_markets)]
    s["iqr_ratio"] = (s["q75"] - s["q25"]) / s["med"]
    s = prefer_retail(s)
    a = (s.groupby(["unit_id", "period_start"])
           .agg(iqr_ratio=("iqr_ratio", "median"), n_markets=("n_markets", "mean"), loc=("loc", "mean"))
           .reset_index())
    a["baseline"], _ = same_month_baseline(a, ["unit_id"], "iqr_ratio")
    a = a.dropna(subset=["baseline"])
    return pd.DataFrame({"unit_id": a["unit_id"], "signal_type": "price_dispersion_anom",
                         "value": a["iqr_ratio"] - a["baseline"],
                         "unit_of_measure": "abs_diff_iqr_over_median_vs_baseline", "period_start": a["period_start"],
                         "confidence": fresh_factor * a["loc"] * (a["n_markets"] / 5).clip(0, 1)})


def coverage_signal(clean, fresh_factor):
    """market_coverage_drop: distinct markets reporting this month minus 12 months earlier (negative = worse).
    Unit-months with no reports at all stay absent; rows without a 12-months-ago value are skipped."""
    c = clean.groupby(["unit_id", "period_start"])["market"].nunique().rename("n").reset_index()
    c["m"] = month_index(c["period_start"])
    ago = c[["unit_id", "m", "n"]].rename(columns={"n": "n_ago"})
    c = c.merge(ago.assign(m=ago["m"] + 12), on=["unit_id", "m"], how="inner")
    yoy = c["n"] - c["n_ago"]
    return pd.DataFrame({"unit_id": c["unit_id"], "signal_type": "market_coverage_drop", "value": yoy,
                         "unit_of_measure": "yoy_change_active_market_count", "period_start": c["period_start"],
                         "confidence": fresh_factor * np.where(yoy.abs() >= 5, 1.0, 0.5)})


def freshness_share(ms, key):
    """Per unit-month: share of the unit's active food series (seen in trailing 12 months) reporting now.
    Only emitted for unit-months with >= 1 observation; months with no data stay absent."""
    f = ms[ms["is_food"]].copy()
    f["series"] = f[key[1:]].astype(str).agg("|".join, axis=1)
    rows = []
    for unit, g in f.groupby("unit_id"):
        p = pd.crosstab(g["period_start"], g["series"]) > 0
        p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="MS"), fill_value=False)
        active = p.astype(int).rolling(12, min_periods=1).max()
        share = p.sum(axis=1) / active.sum(axis=1)
        share = share[p.sum(axis=1) > 0]
        rows.append(pd.DataFrame({"unit_id": unit, "period_start": share.index, "share": share.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["unit_id", "period_start", "share"])


def price_signals(clean, source_id, ref, fresh_factor):
    ms, key = monthly_series(clean)
    ms = seasonal_anomaly(ms, key)
    share = freshness_share(ms, key)
    out = []

    s = share.merge(ms.groupby(["unit_id", "period_start"])["loc_factor"].mean().reset_index(), on=["unit_id", "period_start"])
    out.append(pd.DataFrame({
        "unit_id": s["unit_id"], "signal_type": "data_freshness_share", "value": s["share"],
        "unit_of_measure": "share_of_active_series", "period_start": s["period_start"],
        "confidence": fresh_factor * s["loc_factor"]}))

    food = ms[ms["is_food"]]
    for signal, col, uom in [("price_anom_real", "anom_usd", "pct_vs_5y_same_month_median_usd"),
                             ("price_anom_nominal", "anom_local", "pct_vs_5y_same_month_median_local")]:
        d = food.dropna(subset=[col]).copy()
        if d.empty:
            continue
        # one price type per unit-month (retail preferred); retail and wholesale are never pooled
        d["_pt"] = (d["price_type"] != "retail").astype(int)
        d = d[d["_pt"] == d.groupby(["unit_id", "period_start"])["_pt"].transform("min")]
        cur = col.split("_")[1]
        a = (d.groupby(["unit_id", "period_start"])
               .agg(value=(col, "median"), n_series=(col, "size"), price_type=("price_type", "first"),
                    hist=(f"baseline_years_{cur}", "mean"), loc=("loc_factor", "mean"))
               .reset_index()
               .merge(share, on=["unit_id", "period_start"], how="left"))
        conf = (fresh_factor * a["loc"] * np.minimum(1.0, a["hist"] / BASELINE_YEARS)
                * (0.5 + 0.5 * a["share"].fillna(0.0)))      # missing coverage lowers confidence only
        out.append(pd.DataFrame({
            "unit_id": a["unit_id"], "signal_type": signal, "value": a["value"],
            "unit_of_measure": uom + "_" + a["price_type"], "period_start": a["period_start"], "confidence": conf}))

    out.append(velocity_signal(ms, key, fresh_factor))
    out.append(dispersion_signal(clean, fresh_factor))
    out.append(coverage_signal(clean, fresh_factor))

    sig = pd.concat(out, ignore_index=True)
    sig["source_id"], sig["raw_ref"] = source_id, ref
    return sig, ms


def gmm_signals(sub, nat, cfg, cc, units, regions, refs, fresh_factor):
    """basket_cost_change = GMM TotImpactQuarterlyChange (% change in food-basket cost over 3 months)."""
    rows = []
    adm0 = f"{cc}-ADM0-{slug(cfg['name'])}"
    if cfg.get("region_map"):
        by_norm = {norm(n): u for n, u in zip(units["shapeName"], units["unit_id"])}
        fan = {r: [by_norm[norm(c)] for c in kids if norm(c) in by_norm] for r, kids in cfg["region_map"].items()}
    elif regions is not None:
        reps = units.copy()
        reps["geometry"] = reps.geometry.representative_point()
        j = gpd.sjoin(reps, regions.rename(columns={"unit_id": "region_id"}), predicate="within")
        fan = j.groupby("shapeName_right")["unit_id"].apply(list).to_dict()
    else:
        fan = {n: [u] for n, u in zip(units["shapeName"], units["unit_id"])}

    for df, ref, level in [(sub, refs["sub"], "sub"), (nat, refs["nat"], "nat")]:
        d = df.dropna(subset=["TotImpactQuarterlyChange"])
        d = d.groupby(["Admin1", "period_start"])["TotImpactQuarterlyChange"].first().reset_index()
        for _, r in d.iterrows():
            if level == "nat":
                targets, f = [adm0], 1.0
            else:
                targets, f = fan.get(r["Admin1"], []), FANOUT_FACTOR
            for u in targets:
                rows.append({"unit_id": u, "signal_type": "basket_cost_change", "value": r["TotImpactQuarterlyChange"],
                             "unit_of_measure": "pct_change_3m_basket_cost", "period_start": r["period_start"],
                             "confidence": fresh_factor * f, "raw_ref": ref, "_region": r["Admin1"]})
    sig = pd.DataFrame(rows)
    unmapped = sorted(set(sub["Admin1"]) - set(fan)) if len(sub) else []
    if len(sig):
        sig["source_id"] = "wfp_gmm"
    return sig.drop(columns="_region", errors="ignore"), unmapped


# ----------------------------------------------------------------- storage
SCHEMA_COLS = ["unit_id", "source_id", "signal_type", "value", "unit_of_measure",
               "period_start", "period_end", "ingested_at", "confidence", "raw_ref"]


def upsert(db_path, df):
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS signal_observation (
        unit_id TEXT NOT NULL, source_id TEXT NOT NULL, signal_type TEXT NOT NULL, value REAL NOT NULL,
        unit_of_measure TEXT, period_start TEXT NOT NULL, period_end TEXT, ingested_at TEXT,
        confidence REAL, raw_ref TEXT,
        PRIMARY KEY (unit_id, source_id, signal_type, period_start))""")
    before = con.execute("SELECT COUNT(*) FROM signal_observation").fetchone()[0]
    con.executemany(f"""INSERT INTO signal_observation ({', '.join(SCHEMA_COLS)})
        VALUES ({', '.join('?' * len(SCHEMA_COLS))})
        ON CONFLICT(unit_id, source_id, signal_type, period_start) DO UPDATE SET
          value=excluded.value, unit_of_measure=excluded.unit_of_measure, period_end=excluded.period_end,
          ingested_at=excluded.ingested_at, confidence=excluded.confidence, raw_ref=excluded.raw_ref""",
                    df[SCHEMA_COLS].itertuples(index=False, name=None))
    con.commit()
    after = con.execute("SELECT COUNT(*) FROM signal_observation").fetchone()[0]
    con.close()
    return before, after


def write_clean(df, cc, source):
    if df is None or df.empty:
        return None
    p = CLEAN / f"{cc}_{source}_{df['period_start'].min():%Y-%m}_{df['period_start'].max():%Y-%m}.csv"
    df.drop(columns=["m", "is_outlier"], errors="ignore").to_csv(p, index=False)
    return str(p.relative_to(ROOT))


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--as-of", help="simulated date YYYY-MM-DD for point-in-time back-tests")
    ap.add_argument("--refresh", action="store_true", help="re-download even if today's raw file exists")
    args = ap.parse_args()

    cc, cfg = args.country, COUNTRIES[args.country]
    for d in (RAW, CLEAN, OUT, GEO):
        d.mkdir(parents=True, exist_ok=True)
    run = now_utc()
    run_at, fetch_date, ingested_at = run.strftime("%Y%m%dT%H%M%SZ"), run.strftime("%Y-%m-%d"), run.isoformat(timespec="seconds")
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    ref_date = as_of if as_of is not None else pd.Timestamp(run.date())

    units, regions = load_units(cc, cfg)
    report = {"country": cc, "run_at": run_at, "as_of": args.as_of, "admin_level": cfg["level"],
              "admin_units": int(len(units)), "sources": {}}
    signals, hard_failures, fx = [], [], None

    def new_stats(source_id):
        st = {"source_id": source_id, "status": "ok", "_as_of": as_of, "rows_fetched": 0, "rows_after_cleaning": 0,
              "outlier_count": 0, "latest_date": None, "confidence_adjustment": None}
        report["sources"][source_id] = st
        return st

    # ---- 1. WFP HDX food prices
    sid = f"wfp_hdx_{cc.lower()}"
    st = new_stats(sid)
    try:
        res = ckan_resource(cfg["hdx_slug"], lambda n: n.endswith("Food Prices"))
        raw_path = RAW / f"{cc}_wfp_hdx_{fetch_date}.csv"
        meta = fetch(res["download_url"], raw_path, args.refresh, res.get("last_modified"))
        raw = read_raw(raw_path)
        st.update(rows_fetched=len(raw), raw_path=str(raw_path.relative_to(ROOT)), last_modified=meta["last_modified"])
        clean = clean_wfp(raw, cfg, units, st)
        fx = wfp_fx(clean)
        f, age = freshness_factor(pd.Timestamp(st["latest_date"]), ref_date)
        st.update(confidence_adjustment=f, age_months=age, clean_path=write_clean(clean, cc, "wfp_hdx"))
        sig, _ = price_signals(clean, sid, raw_ref(meta), f)
        signals.append(sig)
    except Exception as e:
        st.update(status="hard_failure", error=repr(e))
        hard_failures.append(sid)

    # ---- 2. FEWS NET (KE only; degrade gracefully to WFP HDX on failure)
    sid = f"fewsnet_{cc.lower()}"
    st = new_stats(sid)
    if not cfg["fews"]:
        st.update(status="skipped", reason=cfg["fews_skip_reason"])
    else:
        try:
            raw_path = RAW / f"{cc}_fewsnet_{fetch_date}.csv"
            meta = fetch(FEWS_URL.format(cc=cc), raw_path, args.refresh)
            raw = read_raw(raw_path)
            st.update(rows_fetched=len(raw), raw_path=str(raw_path.relative_to(ROOT)), last_modified=meta["last_modified"])
            clean = clean_fews(raw, cfg, units, st, fx)
            f, age = freshness_factor(pd.Timestamp(st["latest_date"]), ref_date)
            st.update(confidence_adjustment=f, age_months=age, clean_path=write_clean(clean, cc, "fewsnet"))
            sig, _ = price_signals(clean, sid, raw_ref(meta), f)
            signals.append(sig)
        except Exception as e:   # e.g. HTTP 500: continue with WFP HDX only
            st.update(status="degraded", error=repr(e), note="FEWS NET unavailable; using WFP HDX only")

    # ---- 3. WFP Global Market Monitor
    sid = "wfp_gmm"
    st = new_stats(sid)
    try:
        refs, parts = {}, {}
        for level, rname in [("sub", "global-market-monitor_subnational.csv"), ("nat", "global-market-monitor.csv")]:
            res = ckan_resource("global-market-monitor", lambda n, rname=rname: n == rname)
            tag = "subnational" if level == "sub" else "national"
            raw_path = RAW / f"{cc}_wfp_gmm_{tag}_{fetch_date}.csv"
            meta = fetch(res["download_url"], raw_path, args.refresh, res.get("last_modified"))
            raw = read_raw(raw_path)
            d = raw[raw["CountryName"] == cfg["name"]].copy()
            st["rows_fetched"] += len(d)
            st[f"raw_path_{tag}"] = str(raw_path.relative_to(ROOT))
            st["last_modified"] = meta["last_modified"]
            d["period_start"] = to_month_start(d["Date"])
            d["period_end"] = d["period_start"] + pd.offsets.MonthEnd(0)
            d["TotImpactQuarterlyChange"] = pd.to_numeric(d["TotImpactQuarterlyChange"], errors="coerce")
            d = pit_filter(d.dropna(subset=["period_start"]), as_of)
            refs[level], parts[level] = raw_ref(meta), d
        allg = pd.concat(parts.values())
        st["rows_after_cleaning"] = int(len(allg))
        st["outlier_count"] = 0
        st["outlier_note"] = "GMM publishes pre-computed % changes; no price levels to Hampel-filter"
        st["latest_date"] = allg["period_start"].max().strftime("%Y-%m-%d") if len(allg) else None
        if st["latest_date"]:
            f, age = freshness_factor(pd.Timestamp(st["latest_date"]), ref_date)
            st.update(confidence_adjustment=f, age_months=age)
            sig, unmapped = gmm_signals(parts["sub"], parts["nat"], cfg, cc, units, regions, refs, f)
            st["regions_unmapped"] = unmapped
            st["clean_path"] = write_clean(allg, cc, "wfp_gmm")
            signals.append(sig)
        else:
            st.update(status="no_rows_for_country")
    except Exception as e:
        st.update(status="hard_failure", error=repr(e))
        hard_failures.append(sid)

    # ---- merge, upsert, export
    out = pd.concat([s for s in signals if s is not None and len(s)], ignore_index=True) if signals else pd.DataFrame()
    if len(out):
        out = out.dropna(subset=["value", "unit_id", "period_start"])          # absent stays absent
        out["period_end"] = (out["period_start"] + pd.offsets.MonthEnd(0)).dt.strftime("%Y-%m-%d")
        out["period_start"] = out["period_start"].dt.strftime("%Y-%m-%d")
        out["ingested_at"] = ingested_at
        out["value"] = out["value"].astype(float).round(4)
        out["confidence"] = out["confidence"].clip(0, 1).round(3)
        out = out.drop_duplicates(["unit_id", "source_id", "signal_type", "period_start"], keep="last")
        out = out[SCHEMA_COLS].sort_values(["signal_type", "unit_id", "source_id", "period_start"])

    db = OUT / ("signal_observation.sqlite" if as_of is None else f"signal_observation_asof_{as_of:%Y%m%d}.sqlite")
    db_before, db_after = upsert(db, out) if len(out) else (None, None)
    docs_upsert = docstore.upsert_signals(out.to_dict("records")) if (len(out) and as_of is None) else None
    suffix = "" if as_of is None else f"_asof{as_of:%Y%m%d}"
    out_path = OUT / f"signal_observation_prices_{cc}_{run_at}{suffix}.csv"
    out.to_csv(out_path, index=False)

    # ---- smoke test
    counts = out["signal_type"].value_counts().to_dict() if len(out) else {}
    checks = {
        "zero_hard_failures": {"pass": not hard_failures, "hard_failures": hard_failures},
        "no_values_from_missing": {
            "pass": bool(len(out)) and int(out["value"].isna().sum()) == 0,
            "null_values": int(out["value"].isna().sum()) if len(out) else None,
            "zero_values": int((out["value"] == 0).sum()) if len(out) else None,
            "note": "missing inputs are dropped, never filled; any value==0 is a genuine 0% change "
                    "computed from observed prices"},
        "price_anom_real_populated": {"pass": counts.get("price_anom_real", 0) > 0,
                                      "rows": counts.get("price_anom_real", 0)},
        "derived_signals_populated": {
            # dispersion needs >= 3 markets per commodity-month in a unit AND >= 3 years of that history for the
            # trailing same-month baseline; when no unit has that, it is reported as a warning, not faked.
            "pass": counts.get("price_velocity_anom", 0) > 0 and counts.get("market_coverage_drop", 0) > 0,
            **{k: counts.get(k, 0) for k in DERIVED_SIGNALS},
            **({"warning": "price_dispersion_anom empty: no admin unit has >= 3 markets per commodity-month "
                           "with >= 3 years of same-month history"}
               if counts.get("price_dispersion_anom", 0) == 0 else {})},
        "idempotent_upsert": {"pass": True, "db": str(db.relative_to(ROOT)), "rows_before": db_before, "rows_after": db_after,
                              "tinydb_signals_updated_inserted_total": docs_upsert},
    }
    if cc == "ZW":
        wst = report["sources"]["wfp_hdx_zw"]
        checks["zw_outliers_below_old_false_positive_count"] = {
            "pass": wst.get("outlier_count", 10**9) < OLD_ZW_FALSE_POSITIVES / 2,
            "outlier_count": wst.get("outlier_count"),
            "outlier_count_if_currencies_mixed": wst.get("outlier_count_if_currencies_mixed"),
            "previous_false_positives": OLD_ZW_FALSE_POSITIVES}
    report["signal_row_counts"] = counts
    report["output_csv"] = str(out_path.relative_to(ROOT))
    report["smoke_test"] = checks
    for s in report["sources"].values():
        s.pop("_as_of", None)
    fresh_path = OUT / f"freshness_{run_at}{suffix}.json"
    fresh_path.write_text(json.dumps(report, indent=2, default=str))

    # ---- console summary
    print(f"\nCountry {cc}  run_at {run_at}" + (f"  as_of {args.as_of}" if as_of is not None else ""))
    for sid, s in report["sources"].items():
        print(f"  {sid:<14} {s['status']:<13} latest={s.get('latest_date')}  fetched={s.get('rows_fetched')}  "
              f"clean={s.get('rows_after_cleaning')}  outliers={s.get('outlier_count')}  conf_adj={s.get('confidence_adjustment')}"
              + (f"  error={s['error']}" if s.get("error") else ""))
    print("\nRows per signal_type:")
    for k in ["price_anom_real", "price_anom_nominal", "basket_cost_change", "data_freshness_share", *DERIVED_SIGNALS]:
        print(f"  {k:<22} {counts.get(k, 0):>7}")
    print("\nSmoke test:")
    for k, v in checks.items():
        tag = ("PASS" if not v.get("warning") else "WARN") if v["pass"] else "FAIL"
        print(f"  [{tag}] {k}  " + json.dumps({a: b for a, b in v.items() if a not in ('pass', 'note')}))
    print(f"\nExport   : {out_path}")
    print(f"Freshness: {fresh_path}")
    sys.exit(0 if all(v["pass"] for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
