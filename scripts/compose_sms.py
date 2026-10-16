#!/usr/bin/env python3
"""
Compose audience-specific SMS alerts from risk assessments (TinyDB `alerts` collection).

Reads the latest risk_assessment per unit with level >= 1 (older runs are superseded), skips assessments
that already have an alert for that audience, and drafts one SMS per (assessment, audience).

LLM provider, first match wins:
  OPENAI_API_KEY     -> OpenAI chat.completions, gpt-4o-mini, JSON mode
  ANTHROPIC_API_KEY  -> Claude (SMS_MODEL env, default claude-haiku-5-5), JSON-schema structured output
  neither            -> "[MOCK] ..." placeholder, status='mocked'
Every SMS is validated (<= 155 chars, GSM-7 only, starts with the level word, farmer reply footer); one
stricter retry, then a hardcoded template per (level, audience).

Usage:
  python scripts/compose_sms.py --country KE --audience all
  python scripts/compose_sms.py --country ZW --audience farmer
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import db  # noqa: E402
from ingest_prices import COUNTRIES, load_units, now_utc  # noqa: E402

MAX_CHARS = 155
LEVEL_WORD = {1: "WATCH", 2: "WARNING", 3: "EMERGENCY"}
FARMER_FOOTER = "Reply 1 ok 2 poor 3 failed. STOP to quit"
ANTHROPIC_MODEL = os.environ.get("SMS_MODEL", "claude-haiku-5-5")
OPENAI_MODEL = "gpt-4o-mini"

CAPABILITIES = {
    "farmer": [
        "store_grain_2m", "sell_livestock_early", "plant_drought_tolerant",
        "avoid_selling_below_market", "reduce_grain_sales"
    ],
    "government": [
        "release_strategic_reserves", "adjust_subsidies", "declare_disaster",
        "coordinate_response", "request_international_aid"
    ],
    "ngo": [
        "trigger_anticipatory_cash", "pre_position_food", "dispatch_field_teams",
        "target_households", "support_cooperatives"
    ],
}
LANGUAGE_DEFAULT = {"KE": "en", "ZW": "en"}

ACTION_TEXT = {
    "store_grain_2m": "Store enough grain for 2 months", "sell_livestock_early": "Sell weak livestock early",
    "plant_drought_tolerant": "Plant drought-tolerant seed", "avoid_selling_below_market": "Do not sell below market price",
    "reduce_grain_sales": "Hold back grain sales",
    "release_strategic_reserves": "Prepare strategic reserve release", "adjust_subsidies": "Review food subsidies",
    "declare_disaster": "Prepare disaster declaration", "coordinate_response": "Convene response coordination",
    "request_international_aid": "Prepare international aid request",
    "trigger_anticipatory_cash": "Prepare anticipatory cash", "pre_position_food": "Pre-position food stocks",
    "dispatch_field_teams": "Dispatch field teams to verify", "target_households": "Update household targeting",
    "support_cooperatives": "Support local cooperatives",
}
# action used by the hardcoded fallback template, per (audience, level)
FALLBACK_ACTION = {
    ("farmer", 1): "reduce_grain_sales", ("farmer", 2): "store_grain_2m", ("farmer", 3): "store_grain_2m",
    ("government", 1): "coordinate_response", ("government", 2): "release_strategic_reserves",
    ("government", 3): "declare_disaster",
    ("ngo", 1): "dispatch_field_teams", ("ngo", 2): "trigger_anticipatory_cash", ("ngo", 3): "pre_position_food",
}
DRIVER_TEXT = {
    "price_anom_real": "food prices above normal", "price_anom_nominal": "local food prices above normal",
    "price_velocity_anom": "food prices rising fast", "price_dispersion_anom": "prices uneven across markets",
    "market_coverage_drop": "fewer markets reporting", "data_freshness_share": "market reports missing",
    "basket_cost_change": "food basket cost rising",
}
# GSM 03.38 basic character set (+ extension table); anything else forces UCS-2 and 70-char segments
GSM7 = set("@£$¥èéùìòÇ\nØø\rÅåΔ_ΦΓΛΩΠΨΣΘΞÆæßÉ !\"#¤%&'()*+,-./0123456789:;<=>?¡ABCDEFGHIJKLMNOPQRSTUVWXYZ"
           "ÄÖÑÜ§¿abcdefghijklmnopqrstuvwxyzäöñüà") | set("^{}\\[~]|€")

PROMPT = """You are the SMS composer for an early-warning food shortage system.
Generate ONE SMS for the following risk assessment and audience.

RISK:
- District: {district} ({unit_id}), Country: {country}
- Level: {level} (1=Watch, 2=Warning, 3=Emergency)
- Confidence: {confidence}
- Horizon: {horizon_weeks} weeks
- p_shortage: {p_shortage}
- Top drivers: {drivers_json}
- Sources agreeing: {n_sources_agreeing}/{n_total}

AUDIENCE: {audience}
LANGUAGE: {language}
CAPABILITIES (only recommend actions from this list):
{capabilities}

CONSTRAINTS:
- ≤155 characters (one GSM-7 segment)
- Start with the level word: WATCH, WARNING, or EMERGENCY
- One concrete action from the capabilities list
- No probabilities, no forecasts — only the action
- For farmer audience only: end with "Reply 1 ok 2 poor 3 failed. STOP to quit"
- Write the action in plain words (the text after the colon above), never the code before it

Return JSON: {{"sms": "<text>", "char_count": <int>}}"""

STRICT_SUFFIX = """

Your previous answer was rejected: {errors}
Be stricter: count characters exactly, use only plain ASCII letters, digits and basic punctuation,
start with exactly "{word}", and keep the whole SMS at most 155 characters."""

SMS_SCHEMA = {"type": "object", "properties": {"sms": {"type": "string"}, "char_count": {"type": "integer"}},
              "required": ["sms", "char_count"], "additionalProperties": False}


def validate(sms, level, audience):
    errors = []
    if len(sms) > MAX_CHARS:
        errors.append(f"{len(sms)} characters (max {MAX_CHARS})")
    if not sms.startswith(LEVEL_WORD[level]):
        errors.append(f"does not start with {LEVEL_WORD[level]}")
    bad = sorted(set(sms) - GSM7)
    if bad:
        errors.append(f"non GSM-7 characters {bad}")
    if audience == "farmer" and not sms.rstrip().endswith(FARMER_FOOTER):
        errors.append("missing farmer reply footer")
    if "_" in sms or any(c in sms for caps in CAPABILITIES.values() for c in caps):
        errors.append("contains an internal capability code; use plain words")
    low = sms.lower()
    if "%" in sms or any(w in low for w in ("probab", "likel", "forecast", "horizon", "chance")):
        errors.append("mentions probabilities or forecasts; give only the action")
    return errors


def top_driver(drivers_json):
    d = json.loads(drivers_json or "[]")
    return DRIVER_TEXT.get(d[0]["signal_type"], d[0]["signal_type"].replace("_", " ")) if d else "risk signals"


def fit(parts, limit=MAX_CHARS):
    """Join parts; shorten the district name (parts[1]) if the result is too long."""
    text = "".join(parts)
    if len(text) > limit:
        over = len(text) - limit
        parts = list(parts)
        parts[1] = parts[1][: max(3, len(parts[1]) - over)]
        text = "".join(parts)
    return text[:limit]


def fallback_template(a, audience, district):
    word = LEVEL_WORD[a["level"]]
    action = ACTION_TEXT[FALLBACK_ACTION[(audience, a["level"])]]
    tail = f" {FARMER_FOOTER}" if audience == "farmer" else ""
    return fit([f"{word} ", district, f": {top_driver(a['drivers_json'])}. {action}.", tail])


def mock_sms(a, audience, district):
    first = CAPABILITIES[audience][0]
    return fit(["[MOCK] ", f"{LEVEL_WORD[a['level']]} {district}", f": {top_driver(a['drivers_json'])}. Action: {first}"])


class Composer:
    def __init__(self):
        self.provider = "openai" if os.environ.get("OPENAI_API_KEY") else (
            "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "mock")
        self.client = None
        try:
            if self.provider == "openai":
                from openai import OpenAI
                self.client = OpenAI()
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic()
        except ImportError as e:
            print(f"  {self.provider} SDK not installed ({e}); falling back to mock SMS")
            self.provider = "mock"

    def ask(self, prompt):
        if self.provider == "openai":
            r = self.client.chat.completions.create(
                model=OPENAI_MODEL, response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}])
            return json.loads(r.choices[0].message.content)
        r = self.client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=1024,
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SMS_SCHEMA}},
            messages=[{"role": "user", "content": prompt}])
        if r.stop_reason == "refusal":
            raise RuntimeError("model refused")
        return json.loads(next(b.text for b in r.content if b.type == "text"))

    def compose(self, a, audience, district, country):
        """-> (sms, generator, notes)"""
        if self.provider == "mock":
            return mock_sms(a, audience, district), "mock", []
        prompt = PROMPT.format(
            district=district, unit_id=a["unit_id"], country=country, level=a["level"],
            confidence=a["confidence"], horizon_weeks=a["horizon_weeks"], p_shortage=a["p_shortage"],
            drivers_json=a["drivers_json"], n_sources_agreeing=a["n_sources_agreeing"], n_total=a.get("n_total"),
            audience=audience, language=LANGUAGE_DEFAULT.get(country, "en"),
            capabilities="\n".join(f"- {c}: {ACTION_TEXT[c]}" for c in CAPABILITIES[audience]))
        notes = []
        for attempt in range(2):
            try:
                sms = str(self.ask(prompt).get("sms", "")).strip()
            except Exception as e:      # API/network/JSON errors: count as a failed attempt
                notes.append(f"attempt {attempt + 1}: {type(e).__name__}: {e}")
                continue
            errors = validate(sms, a["level"], audience)
            if not errors:
                return sms, f"{self.provider}:{'first' if attempt == 0 else 'retry'}", notes
            notes.append(f"attempt {attempt + 1}: {'; '.join(errors)}")
            prompt = prompt + STRICT_SUFFIX.format(errors="; ".join(errors), word=LEVEL_WORD[a["level"]])
        return fallback_template(a, audience, district), "template_fallback", notes


def build_alert(a, audience, district, country, composer):
    """One alert document for a risk assessment `a` (needs _doc_id, unit_id, level, run_at, ...)."""
    sms, gen, notes = composer.compose(a, audience, district, country)
    created = now_utc()
    status = "mocked" if gen == "mock" else ("approved" if a["level"] == 1 else "drafted")
    return {"assessment_id": a["_doc_id"], "unit_id": a["unit_id"], "level": a["level"], "audience": audience,
            "template_text": sms, "char_count": len(sms), "language": LANGUAGE_DEFAULT.get(country, "en"),
            "status": status, "approved_by": "auto:level1" if status == "approved" else None,
            "generator": gen, "generation_notes": notes, "run_at": a["run_at"],
            "created_at": created.strftime("%Y%m%dT%H%M%SZ"),
            "expires_at": (created + timedelta(weeks=int(a.get("horizon_weeks") or 8))).strftime("%Y%m%dT%H%M%SZ")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--country", required=True, choices=sorted(COUNTRIES))
    ap.add_argument("--audience", default="all", choices=["farmer", "government", "ngo", "all"])
    args = ap.parse_args()
    audiences = list(CAPABILITIES) if args.audience == "all" else [args.audience]

    units, _ = load_units(args.country, COUNTRIES[args.country])
    names = dict(zip(units["unit_id"], units["shapeName"]))
    risk = db.query_risk(country=args.country, level_ge=1)
    latest_run = {}
    for a in db.query_risk(country=args.country):          # newest run per unit, any level
        latest_run[a["unit_id"]] = max(latest_run.get(a["unit_id"], ""), a["run_at"])
    risk = [a for a in risk if a["run_at"] == latest_run[a["unit_id"]]]

    todo = []
    for a in risk:
        for aud in audiences:
            if not db.query_alerts(assessment_id=a["_doc_id"], audience=aud):
                todo.append((a, aud))
    composer = Composer()
    print(f"SMS {args.country}  provider={composer.provider}"
          + (f" model={ANTHROPIC_MODEL}" if composer.provider == "anthropic" else "")
          + f"  assessments L1+: {len(risk)}  alerts to draft: {len(todo)}")

    def work(item):
        a, aud = item
        return build_alert(a, aud, names.get(a["unit_id"], a["unit_id"]), args.country, composer)

    with ThreadPoolExecutor(max_workers=4) as pool:
        alerts = list(pool.map(work, todo))
    res = db.upsert_alerts(alerts) if alerts else (0, 0, None)
    for al in alerts:
        print(f"  [{al['status']:<8}|{al['generator']:<18}] {al['audience']:<10} {al['unit_id']:<22} "
              f"({al['char_count']}) {al['template_text']}")
        for n in al["generation_notes"]:
            print(f"      note: {n}")
    print(f"  upsert (updated, inserted, total): {res}")


if __name__ == "__main__":
    main()
