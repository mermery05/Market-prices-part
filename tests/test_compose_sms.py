"""
SMS composition. No real LLM call is ever made: the autouse fixture removes API keys, and the provider
tests swap in fake SDK clients.

Note: the spec mentions patching `openai.ChatCompletion.create`; that API was removed in openai>=1.0,
which is what compose_sms.py targets (`OpenAI().chat.completions.create`), so the fake mirrors that shape.
"""
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import compose_sms
from compose_sms import (ACTION_TEXT, CAPABILITIES, FARMER_FOOTER, LEVEL_WORD, MAX_CHARS, Composer, build_alert,
                         fallback_template, validate)

DISTRICT = "District 1"
AUDIENCES = ["farmer", "government", "ngo"]


def assessment(level=2):
    return {"_doc_id": 7, "unit_id": "TEST-D1", "level": level, "run_at": "20250115T120000Z", "horizon_weeks": 8,
            "confidence": 0.5, "p_shortage": 0.5, "n_sources_agreeing": 2, "n_total": 3,
            "drivers_json": json.dumps([{"signal_type": "price_anom_real", "value": 40.0, "z_score": 10.0}])}


def valid_sms(level, audience):
    action = ACTION_TEXT[CAPABILITIES[audience][0]]
    body = f"{LEVEL_WORD[level]} {DISTRICT}: food prices above normal. {action}."
    return f"{body} {FARMER_FOOTER}" if audience == "farmer" else body


def fake_composer(*answers):
    """Composer whose LLM returns the given SMS texts in order."""
    c = Composer()                          # no keys -> mock provider, no SDK client
    c.provider = "anthropic"
    c.ask = MagicMock(side_effect=[{"sms": a, "char_count": len(a)} for a in answers])
    return c


def test_mocks_llm_when_no_api_key():
    c = Composer()
    assert c.provider == "mock" and c.client is None
    alert = build_alert(assessment(2), "farmer", DISTRICT, "TEST", c)
    assert alert["status"] == "mocked"
    assert alert["generator"] == "mock"
    assert alert["template_text"].startswith("[MOCK] WARNING District 1")


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_calls_llm_when_api_key_set(provider, monkeypatch):
    answer = json.dumps({"sms": valid_sms(2, "government"), "char_count": 0})
    if provider == "openai":
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = MagicMock(return_value=client)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        c = Composer()
        call = client.chat.completions.create
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        client = MagicMock()
        client.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn", content=[SimpleNamespace(type="text", text=answer)])
        with patch("anthropic.Anthropic", return_value=client):
            c = Composer()
        call = client.messages.create

    assert c.provider == provider
    sms, gen, notes = c.compose(assessment(2), "government", DISTRICT, "TEST")
    assert sms == valid_sms(2, "government") and gen == f"{provider}:first" and notes == []
    call.assert_called_once()
    kwargs = call.call_args.kwargs
    prompt = kwargs["messages"][0]["content"]
    assert "District: District 1 (TEST-D1)" in prompt and "Level: 2" in prompt and "AUDIENCE: government" in prompt
    assert all(cap in prompt for cap in CAPABILITIES["government"])
    assert not any(cap in prompt for cap in CAPABILITIES["farmer"])
    if provider == "openai":
        assert kwargs["model"] == "gpt-4o-mini" and kwargs["response_format"] == {"type": "json_object"}
    else:
        assert kwargs["model"] == compose_sms.ANTHROPIC_MODEL
        assert kwargs["output_config"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_sms_starts_with_level_word(level):
    word = LEVEL_WORD[level]
    assert any("does not start with" in e for e in validate("Food shortage expected soon.", level, "ngo"))
    wrong_word = valid_sms(level, "ngo").replace(word, "ALERT", 1)
    c = fake_composer(wrong_word, valid_sms(level, "ngo"))
    sms, _, notes = c.compose(assessment(level), "ngo", DISTRICT, "TEST")
    assert sms.startswith(word) and "does not start with" in notes[0]
    assert fallback_template(assessment(level), "ngo", DISTRICT).startswith(word)


def test_sms_under_155_chars():
    too_long = valid_sms(2, "ngo").rstrip(".") + " " + "x" * 120 + "."
    assert len(too_long) > MAX_CHARS
    assert any("characters" in e for e in validate(too_long, 2, "ngo"))
    c = fake_composer(too_long, valid_sms(2, "ngo"))
    sms, gen, notes = c.compose(assessment(2), "ngo", DISTRICT, "TEST")
    assert c.ask.call_count == 2                       # retried exactly once
    assert gen == "anthropic:retry" and len(sms) <= MAX_CHARS
    retry_prompt = c.ask.call_args_list[1].args[0]
    assert "rejected" in retry_prompt and "characters" in retry_prompt


def test_falls_back_to_hardcoded_after_retry_failure():
    bad = "x" * 200
    c = fake_composer(bad, bad)
    sms, gen, notes = c.compose(assessment(2), "farmer", DISTRICT, "TEST")
    assert c.ask.call_count == 2 and gen == "template_fallback" and len(notes) == 2
    assert sms == fallback_template(assessment(2), "farmer", DISTRICT)
    assert validate(sms, 2, "farmer") == []


def test_falls_back_when_the_api_errors():
    c = Composer()
    c.provider = "anthropic"
    c.ask = MagicMock(side_effect=RuntimeError("boom"))
    sms, gen, notes = c.compose(assessment(3), "ngo", DISTRICT, "TEST")
    assert gen == "template_fallback" and "boom" in notes[0] and validate(sms, 3, "ngo") == []


@pytest.mark.parametrize("level", [1, 2, 3])
def test_farmer_audience_gets_reply_instruction(level):
    sms = fallback_template(assessment(level), "farmer", DISTRICT)
    assert sms.endswith(FARMER_FOOTER)
    no_footer = valid_sms(level, "farmer").replace(" " + FARMER_FOOTER, "")
    assert "missing farmer reply footer" in validate(no_footer, level, "farmer")
    alert = build_alert(assessment(level), "farmer", DISTRICT, "TEST", fake_composer(valid_sms(level, "farmer")))
    assert alert["template_text"].endswith(FARMER_FOOTER)


@pytest.mark.parametrize("audience", ["government", "ngo"])
@pytest.mark.parametrize("level", [1, 2, 3])
def test_non_farmer_audience_no_reply_instruction(audience, level):
    sms = fallback_template(assessment(level), audience, DISTRICT)
    assert "Reply 1" not in sms and "STOP" not in sms
    assert validate(sms, level, audience) == []


@pytest.mark.parametrize("audience", AUDIENCES)
@pytest.mark.parametrize("level", [1, 2, 3])
def test_capability_action_in_sms(audience, level):
    sms = fallback_template(assessment(level), audience, DISTRICT)
    assert any(ACTION_TEXT[cap] in sms for cap in CAPABILITIES[audience])
    others = [cap for aud, caps in CAPABILITIES.items() if aud != audience for cap in caps]
    assert not any(ACTION_TEXT[cap] in sms for cap in others)
    # raw capability codes are rejected (plain words only)
    assert any("capability code" in e for e in validate(f"{LEVEL_WORD[level]} X: {CAPABILITIES[audience][0]}",
                                                        level, audience))


@pytest.mark.parametrize("level", [2, 3])
def test_l2_plus_status_drafted(level):
    alert = build_alert(assessment(level), "ngo", DISTRICT, "TEST", fake_composer(valid_sms(level, "ngo")))
    assert alert["status"] == "drafted" and alert["approved_by"] is None
    assert alert["assessment_id"] == 7 and alert["level"] == level


def test_l1_status_approved():
    alert = build_alert(assessment(1), "government", DISTRICT, "TEST", fake_composer(valid_sms(1, "government")))
    assert alert["status"] == "approved" and alert["approved_by"] == "auto:level1"
    assert alert["expires_at"] > alert["created_at"]
