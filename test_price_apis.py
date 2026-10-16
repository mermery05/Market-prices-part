#!/usr/bin/env python3
"""
Smoke test for market-price data sources, per country.

Tests:
  1. WFP food prices on HDX (CKAN API, no key needed)       -> main test
  2. FEWS NET staple food prices on HDX (no key needed)     -> history / gap check
  3. WFP Global Market Monitor on HDX (no key needed)       -> biweekly basket cost
  4. WFP DataBridges API (needs WFP_API_KEY / WFP_API_SECRET) -> optional, UNVERIFIED endpoints

Usage:
  pip install requests pandas
  python test_price_apis.py                      # Kenya + Zimbabwe
  python test_price_apis.py --countries Kenya Zimbabwe Malawi
  python test_price_apis.py --save               # also save the cleaned CSVs to ./data

Run it from your own machine (needs normal internet access).
"""
import argparse
import io
import os
import sys
from datetime import datetime

import pandas as pd
import requests

HDX = "https://data.humdata.org/api/3/action"
TIMEOUT = 60
FRESH_MONTHS_OK = 3        # latest price older than this -> flagged as stale
HEADERS = {"User-Agent": "sms-shortage-early-warning-hackathon/0.1"}

ISO3 = {"kenya": "KEN", "zimbabwe": "ZWE", "malawi": "MWI", "zambia": "ZMB"}


# ----------------------------------------------------------------- helpers
def months_ago(ts):
    now = pd.Timestamp.now()
    return (now.year - ts.year) * 12 + (now.month - ts.month)


def ckan(action, **params):
    r = requests.get(f"{HDX}/{action}", params=params, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"CKAN error: {body.get('error')}")
    return body["result"]


def find_dataset(slug, search_query):
    """Try the expected slug first, then fall back to a search."""
    try:
        return ckan("package_show", id=slug)
    except Exception:
        res = ckan("package_search", q=search_query, rows=5)
        if res["count"] == 0:
            raise RuntimeError(f"no dataset found for '{search_query}'")
        print(f"   (slug '{slug}' not found; using search hit '{res['results'][0]['name']}')")
        return res["results"][0]


def pick_csv(dataset, avoid=("quickcharts",)):
    for res in dataset["resources"]:
        name = (res.get("name") or "").lower()
        fmt = (res.get("format") or "").lower()
        if fmt == "csv" and not any(a in name for a in avoid):
            return res
    raise RuntimeError("no suitable CSV resource in dataset")


def read_csv_url(url):
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT * 3)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), low_memory=False)
    # HDX files often have a second row of '#tags' (HXL). Drop it.
    first_col = df.columns[0]
    df = df[~df[first_col].astype(str).str.startswith("#")].copy()
    return df


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def verdict(ok, msg):
    print(f"   [{'PASS' if ok else 'WARN'}] {msg}")


# ----------------------------------------------------------------- test 1
def test_wfp_prices(country, save):
    slug = f"wfp-food-prices-for-{country.lower()}"
    ds = find_dataset(slug, f"WFP food prices {country}")
    res = pick_csv(ds)
    print(f"   dataset      : {ds['name']}")
    print(f"   metadata mod : {ds.get('metadata_modified')}")
    print(f"   dataset_date : {ds.get('dataset_date')}")
    print(f"   update freq  : {ds.get('data_update_frequency')} (days; 30 = monthly)")
    print(f"   licence      : {ds.get('license_title')}")
    print(f"   resource     : {res.get('name')}  (modified {res.get('last_modified')})")

    df = read_csv_url(res["download_url"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["usdprice"] = pd.to_numeric(df.get("usdprice"), errors="coerce")
    df = df.dropna(subset=["date", "price"])

    latest = df["date"].max()
    age = months_ago(latest)
    print(f"   rows         : {len(df):,}")
    print(f"   date range   : {df['date'].min().date()} -> {latest.date()}")
    print(f"   markets      : {df['market'].nunique()}  | commodities: {df['commodity'].nunique()}")
    print(f"   currencies   : {sorted(df['currency'].dropna().unique())}")
    if "pricetype" in df:
        print(f"   price types  : {df['pricetype'].value_counts().to_dict()}")
    if "priceflag" in df:
        print(f"   price flags  : {df['priceflag'].value_counts().to_dict()}")

    verdict(age <= FRESH_MONTHS_OK, f"latest price is {age} month(s) old (limit {FRESH_MONTHS_OK})")

    coords = df[["market", "latitude", "longitude"]].drop_duplicates("market")
    coords = coords.assign(
        latitude=pd.to_numeric(coords["latitude"], errors="coerce"),
        longitude=pd.to_numeric(coords["longitude"], errors="coerce"),
    )
    n_coords = coords.dropna(subset=["latitude", "longitude"]).shape[0]
    verdict(n_coords == len(coords), f"{n_coords}/{len(coords)} markets have coordinates (needed for district join)")

    # staple check: last 12 months of the data, top commodities by number of observations
    recent = df[df["date"] >= latest - pd.DateOffset(months=12)]
    top = recent["commodity"].value_counts().head(5)
    print("   top commodities (last 12 months of data):")
    for name, n in top.items():
        print(f"      {name:<35} {n:>6} obs")

    # coverage per month for the last 12 months
    cov = recent.groupby(recent["date"].dt.to_period("M"))["market"].nunique()
    print("   markets reporting per month (last 12):", cov.to_dict())
    verdict(cov.min() >= 3 if len(cov) else False, "at least 3 markets report every month")

    # outlier sanity: price per commodity vs its median
    med = df.groupby(["commodity", "unit"])["price"].transform("median")
    ratio = df["price"] / med
    n_out = int(((ratio > 20) | (ratio < 0.05)).sum())
    print(f"   suspicious outliers (>20x or <0.05x of series median): {n_out}")

    if save:
        os.makedirs("data", exist_ok=True)
        path = f"data/wfp_prices_{country.lower()}.csv"
        df.to_csv(path, index=False)
        print(f"   saved -> {path}")
    return df


# ----------------------------------------------------------------- test 2
def test_fewsnet(country):
    res = ckan("package_search", q=f"FEWS NET staple food price {country}", rows=5)
    hits = [d for d in res["results"] if "fews" in d["name"].lower() or "fews" in d["title"].lower()]
    if not hits:
        verdict(False, "no FEWS NET price dataset found on HDX")
        return
    ds = hits[0]
    print(f"   dataset      : {ds['title']}")
    print(f"   dataset_date : {ds.get('dataset_date')}")
    print(f"   metadata mod : {ds.get('metadata_modified')}")
    print("   note: FEWS NET collection was paused in early 2025; check for a hole in recent months.")
    try:
        csv = pick_csv(ds)
        df = read_csv_url(csv["download_url"])
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        if date_col:
            d = pd.to_datetime(df[date_col], errors="coerce").dropna()
            months = d.dt.to_period("M")
            verdict(months_ago(d.max()) <= FRESH_MONTHS_OK,
                    f"latest FEWS NET record {d.max().date()}")
            recent_2025 = months[months >= "2025-01"].nunique()
            print(f"   distinct months with data since 2025-01: {recent_2025}")
    except Exception as e:
        print(f"   could not parse a CSV ({e}); inspect the dataset page manually.")


# ----------------------------------------------------------------- test 3
def test_market_monitor():
    ds = find_dataset("global-market-monitor", "WFP Global Market Monitor")
    print(f"   dataset      : {ds['name']}")
    print(f"   dataset_date : {ds.get('dataset_date')}")
    print(f"   update freq  : {ds.get('data_update_frequency')} (days; 14 = biweekly)")
    print(f"   resources    : {[r['name'] for r in ds['resources']][:6]}")


# ----------------------------------------------------------------- test 4
def test_databridges(countries):
    """
    OPTIONAL and UNVERIFIED. The base host and the endpoint paths below come from the
    DataBridges API repo README; the token URL and query parameter names are my best
    recollection and may need adjusting against the official docs:
      https://databridges.vam.wfp.org/  and  https://github.com/WFP-VAM/DataBridgesAPI
    Credentials: email wfp.vaminfo@wfp.org (external users).
    """
    key, secret = os.getenv("WFP_API_KEY"), os.getenv("WFP_API_SECRET")
    if not (key and secret):
        print("   skipped: set WFP_API_KEY and WFP_API_SECRET to test DataBridges.")
        return
    token_url = "https://api.wfp.org/token"          # UNVERIFIED
    base = "https://api.wfp.org/vam-data-bridges/5.0.0"
    r = requests.post(token_url, data={"grant_type": "client_credentials"},
                      auth=(key, secret), headers=HEADERS, timeout=TIMEOUT)
    print(f"   token request: HTTP {r.status_code}")
    if r.status_code != 200:
        print("   token failed; check credentials or token URL in the official docs.")
        return
    auth = {"Authorization": f"Bearer {r.json()['access_token']}", **HEADERS}
    for c in countries:
        iso = ISO3.get(c.lower())
        for path in ("/Commodities/List", "/MarketPrices/PriceMonthly"):
            resp = requests.get(base + path, headers=auth,
                                params={"countryCode": iso, "page": 1},   # param names UNVERIFIED
                                timeout=TIMEOUT)
            print(f"   {c:<10} GET {path:<28} -> HTTP {resp.status_code}  {resp.text[:120]!r}")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["Kenya", "Zimbabwe"])
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    print(f"Run time: {datetime.now():%Y-%m-%d %H:%M}")
    failures = 0

    for country in args.countries:
        header(f"1. WFP food prices (HDX) - {country}")
        try:
            test_wfp_prices(country, args.save)
        except Exception as e:
            failures += 1
            print(f"   [FAIL] {e}")

        header(f"2. FEWS NET staple prices (HDX) - {country}")
        try:
            test_fewsnet(country)
        except Exception as e:
            failures += 1
            print(f"   [FAIL] {e}")

    header("3. WFP Global Market Monitor (HDX)")
    try:
        test_market_monitor()
    except Exception as e:
        failures += 1
        print(f"   [FAIL] {e}")

    header("4. WFP DataBridges API (optional)")
    try:
        test_databridges(args.countries)
    except Exception as e:
        print(f"   [FAIL] {e}")

    print(f"\nDone. Hard failures: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
