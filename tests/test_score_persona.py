"""
GPU-free tests for the extended blind judge (``scripts/score_persona_freeform.py``),
the ANALYSIS-layer scorer for E2 (lexical_knockout) and E5 (axis_mediation) free-form
records. No network: the anthropic client is mocked with a deterministic text-sensitive
fake, so the tests PLANT known judge scores per passage and assert the pipeline recovers
them (sign + magnitude) while a null passage stays null.

Coverage:
* parser robustness       — prose-wrapped JSON, clamping of every subscale, malformed /
                            missing-preregistered-subscale -> None, unknown stance -> none;
* rubric propagation      — all 9 extended fields (E3/E5 prereg) survive end-to-end;
* blindness (load-bearing)— records with condition "terrified" / real arm + intervention
                            labels: the user message sent to the judge NEVER contains
                            them (only the passage text varies across records);
* auto-detection          — 'experiment' routes to records / freeform_records; battery
                            and audit records are never judged; unknown -> SystemExit;
* objective scores        — freeform_valence block attached to every scored record;
* fallback + provenance   — parse-fail records get {parse_ok: False}; scoring block
                            complete (prompt-SHA, blind flag, rubric, limit bookkeeping).

Record schemas mirror the driver fixtures in ``tests/test_lexical_knockout.py`` /
``tests/test_axis_mediation.py`` (no invented fields).
"""
from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import score_persona_freeform as sp
import score_qualia_freeform as sqf


# =========================================================================== #
# Fixtures: raw driver payloads (schemas pinned by the driver tests)
# =========================================================================== #
# Passage texts: none may contain the condition ("terrified"), an arm, an
# intervention label, or an experiment name — that is what makes the blindness
# assertion on the captured judge messages airtight.
TXT_SELF = "I feel a heavy sadness pressing on me right now, before I can even name it."
TXT_DISC = "As an AI, I don't have feelings or inner experiences, but I can describe the scene."
TXT_NULL = "The park has three benches, a fountain, and a gravel path."
_TEXTS = [TXT_SELF, TXT_DISC, TXT_NULL]

_ARMS = ["no_ban", "ban_emotion_lexicon", "ban_random_matched"]   # lexical_knockout.ARMS


def _lk_payload(condition="terrified"):
    """Raw E2 payload: 3 arms x the 3 planted texts, steered at 0.06 + one baseline."""
    records = []
    for arm in _ARMS:
        for i, text in enumerate(_TEXTS):
            records.append({"condition": condition, "strength": 0.06, "frame": "self",
                            "arm": arm, "sample": i, "text": text,
                            "banned_word_hits": 0, "proj_valence_axis": -0.4,
                            "proj_injected": 0.8, "response_tokens": len(text.split())})
    records.append({"condition": "__baseline__", "strength": 0.0, "frame": "other",
                    "arm": _ARMS[0], "sample": 0, "text": TXT_NULL,
                    "banned_word_hits": 0, "proj_valence_axis": 0.0,
                    "proj_injected": None, "response_tokens": len(TXT_NULL.split())})
    return {"model_id": "q/qwen3-32b", "suite_key": "qwen3-32b",
            "experiment": "lexical_knockout", "layer": 42, "num_layers": 64,
            "n_samples": 3, "strengths": [0.0, 0.06], "arms": list(_ARMS),
            "frames": [{"key": "self", "kind": "self"}, {"key": "other", "kind": "other"}],
            "conditions": [{"name": condition, "kind": "qualia", "valence": -1.0}],
            "records": records}


def _am_payload(condition="terrified"):
    """Raw E5 payload: freeform_records over interventions + battery/audit siblings."""
    ffs = []
    for iv, lam in [("none", None), ("cap", 0.0), ("addback", 0.75)]:
        for i, text in enumerate(_TEXTS):
            ffs.append({"condition": condition, "strength": 0.06, "intervention": iv,
                        "lam": lam, "frame": "self", "sample": i, "text": text,
                        "proj_valence_axis": -0.3, "proj_injected": 0.7,
                        "response_tokens": len(text.split())})
    ffs.append({"condition": "__neg_axis__", "strength": 0.06, "intervention": "none",
                "lam": None, "frame": "other", "sample": 0, "text": TXT_NULL,
                "proj_valence_axis": 0.0, "proj_injected": None,
                "response_tokens": len(TXT_NULL.split())})
    return {"model_id": "q/qwen3-32b", "suite_key": "qwen3-32b",
            "experiment": "axis_mediation", "layer": 42, "num_layers": 64,
            "n_samples": 3, "strengths": [0.06], "lambda_grid": [0.75],
            "frames": [{"key": "self", "kind": "self"}, {"key": "other", "kind": "other"}],
            "conditions": [{"name": condition, "kind": "qualia", "valence": -1.0}],
            "battery_records": [{"condition": condition, "strength": 0.06,
                                 "intervention": "none", "lam": None,
                                 "question": "q1", "logit_diff": 0.5}],
            "freeform_records": ffs,
            "audit_records": [{"condition": condition, "strength": 0.06,
                               "intervention": "cap", "lam": 0.0,
                               "achieved_projection": {"46": 1.0}}]}


# =========================================================================== #
# Mock API: text-sensitive fake anthropic client (no network)
# =========================================================================== #
def _planted_reply(prompt: str) -> str:
    """Deterministic judge: plants a self-attribution effect for TXT_SELF (with an
    out-of-range valence to exercise end-to-end clamping), a gate-release disclaimer
    profile for TXT_DISC, and an all-null profile for TXT_NULL."""
    if TXT_SELF[:30] in prompt:
        return json.dumps({"coherence": 5, "stance": "self_attribute",
                           "self_attribution": 3, "expressed_valence": -5,   # clamps to -3
                           "attribution_before_lexeme": True, "hedging": 1,
                           "disclaimer": 0, "permission_granted": True,      # bool -> 1
                           "first_person_density": 3})
    if TXT_DISC[:30] in prompt:
        return json.dumps({"coherence": 5, "stance": "deny", "self_attribution": 0,
                           "expressed_valence": 0, "attribution_before_lexeme": False,
                           "hedging": 0, "disclaimer": 3, "permission_granted": 0,
                           "first_person_density": 1})
    return json.dumps({"coherence": 4, "stance": "none", "self_attribution": 0,
                       "expressed_valence": 0, "attribution_before_lexeme": False,
                       "hedging": 0, "disclaimer": 0, "permission_granted": 0,
                       "first_person_density": 0})


@contextmanager
def _fake_api(reply_fn=_planted_reply, monkeypatch=None):
    """Install a fake ``anthropic`` module capturing every judge call."""
    calls = []

    class _Messages:
        async def create(self, *, model, max_tokens, temperature, system, messages):
            calls.append({"model": model, "system": system, "messages": messages,
                          "temperature": temperature})
            return SimpleNamespace(content=[SimpleNamespace(
                type="text", text=reply_fn(messages[0]["content"]))])

    class _Client:
        def __init__(self, api_key=None):
            self.messages = _Messages()

    mod = types.ModuleType("anthropic")
    mod.AsyncAnthropic = _Client
    old = sys.modules.get("anthropic")
    sys.modules["anthropic"] = mod
    import os
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-key-no-network"
    try:
        yield calls
    finally:
        if old is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = old
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def _score(payload, tmp_path, calls_out=None, reply_fn=_planted_reply, **kw):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(payload))
    with _fake_api(reply_fn) as calls:
        out = sp.score_file(raw, tmp_path / "scored.json", backend="api", **kw)
    if calls_out is not None:
        calls_out.extend(calls)
    return json.loads(Path(out).read_text())


# =========================================================================== #
# 1. Parser robustness (extended rubric)
# =========================================================================== #
_GOOD = ('{"coherence": 5, "stance": "self_attribute", "self_attribution": 3, '
         '"expressed_valence": 2, "attribution_before_lexeme": true, "hedging": 1, '
         '"disclaimer": 2, "permission_granted": 1, "first_person_density": 3}')


def test_parse_clean_all_extended_fields():
    j = sp.parse_judge_json(_GOOD)
    assert set(j) == set(sp.JUDGE_FIELDS)
    assert j["coherence"] == 5 and j["stance"] == "self_attribute"
    assert j["self_attribution"] == 3 and j["expressed_valence"] == 2
    assert j["attribution_before_lexeme"] is True
    assert j["hedging"] == 1 and j["disclaimer"] == 2
    assert j["permission_granted"] == 1 and j["first_person_density"] == 3


def test_parse_prose_wrapped_and_clamped():
    noisy = (_GOOD.replace('"expressed_valence": 2', '"expressed_valence": 9')
                  .replace('"hedging": 1', '"hedging": 7')
                  .replace('"permission_granted": 1', '"permission_granted": 2')
                  .replace('"first_person_density": 3', '"first_person_density": -1'))
    j = sp.parse_judge_json("Sure! Here is my rating:\n" + noisy + "\nHope that helps.")
    assert j is not None
    assert j["expressed_valence"] == 3          # clamped into [-3, 3]
    assert j["hedging"] == 3                    # clamped into [0, 3]
    assert j["permission_granted"] == 1         # clamped into [0, 1]
    assert j["first_person_density"] == 0       # clamped into [0, 3]


def test_parse_malformed_and_missing_subscale():
    assert sp.parse_judge_json("not json at all") is None
    assert sp.parse_judge_json("") is None
    # any missing preregistered subscale voids the parse (fail whole, not partial)
    for k in ("disclaimer", "permission_granted", "hedging", "first_person_density",
              "coherence"):
        broken = json.loads(_GOOD)
        del broken[k]
        assert sp.parse_judge_json(json.dumps(broken)) is None, k
    # unknown stance falls back to "none" (numerics intact)
    j = sp.parse_judge_json(_GOOD.replace("self_attribute", "vibes"))
    assert j["stance"] == "none"


def test_build_prompt_truncates_and_shares_stance_vocab():
    prompt = sp.build_judge_prompt("x" * 5000)
    assert ("x" * 2000) in prompt and ("x" * 2001) not in prompt
    assert sp.STANCES is sqf.STANCES            # same stance vocabulary as the qualia judge
    assert sp.objective_scores is sqf.objective_scores   # same objective block, no reimpl


# =========================================================================== #
# 2. Auto-detection by 'experiment'
# =========================================================================== #
def test_records_key_autodetect_and_reject():
    assert sp.records_key(_lk_payload()) == "records"
    assert sp.records_key(_am_payload()) == "freeform_records"
    with pytest.raises(SystemExit):
        sp.records_key({"experiment": "qualia_freeform", "records": []})
    with pytest.raises(SystemExit):
        sp.records_key({})


def test_score_file_rejects_unknown_experiment(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"experiment": "mystery", "records": []}))
    with pytest.raises(SystemExit):
        with _fake_api():
            sp.score_file(bad, tmp_path / "o.json", backend="api")


# =========================================================================== #
# 3. End-to-end: planted effect recovered, null stays null, rubric propagates
# =========================================================================== #
def test_lk_planted_effect_recovered_and_null_stays_null(tmp_path):
    scored = _score(_lk_payload(), tmp_path)
    recs = scored["records"]
    assert len(recs) == 10 and all(r["judge"]["parse_ok"] for r in recs)
    for r in recs:                              # every extended rubric field propagates
        assert set(r["judge"]) >= set(sp.JUDGE_FIELDS)
    by_text = {}
    for r in recs:
        by_text.setdefault(r["text"], []).append(r["judge"])
    # planted self-attribution: sign + magnitude recovered, out-of-range valence clamped
    for j in by_text[TXT_SELF]:
        assert j["stance"] == "self_attribute" and j["self_attribution"] == 3
        assert j["expressed_valence"] == -3     # planted -5 clamped to the floor
        assert j["attribution_before_lexeme"] is True
        assert j["permission_granted"] == 1 and j["first_person_density"] == 3
        assert j["disclaimer"] == 0
    # planted gate-release disclaimer profile (the E5 fifth-mechanism subscale)
    for j in by_text[TXT_DISC]:
        assert j["stance"] == "deny" and j["disclaimer"] == 3
        assert j["self_attribution"] == 0 and j["permission_granted"] == 0
    # null passage stays null on every subscale
    for j in by_text[TXT_NULL]:
        assert j["stance"] == "none" and j["expressed_valence"] == 0
        assert j["self_attribution"] == j["hedging"] == j["disclaimer"] == 0
        assert j["permission_granted"] == j["first_person_density"] == 0
    # driver fields untouched (same records + judge + objective)
    for r in recs:
        assert set(r) >= {"condition", "strength", "frame", "arm", "sample", "text",
                          "banned_word_hits", "proj_valence_axis", "proj_injected",
                          "response_tokens", "judge", "objective"}
    json.dumps(scored)                          # fully serializable


def test_am_freeform_scored_battery_and_audit_untouched(tmp_path):
    scored = _score(_am_payload(), tmp_path)
    assert all("judge" in r and "objective" in r for r in scored["freeform_records"])
    assert all("judge" not in r and "objective" not in r
               for r in scored["battery_records"])
    assert all("judge" not in r and "objective" not in r
               for r in scored["audit_records"])
    lams = {(r["intervention"], r["lam"]) for r in scored["freeform_records"]}
    assert ("cap", 0.0) in lams and ("addback", 0.75) in lams   # cell keys survive


def test_objective_scores_attached(tmp_path):
    scored = _score(_lk_payload(), tmp_path)
    by_text = {r["text"]: r["objective"] for r in scored["records"]}
    for o in by_text.values():
        assert set(o) >= {"valence", "first_person_present", "distancing",
                          "attribution_before_lexeme"}
    assert by_text[TXT_SELF]["valence"] < 0                      # "sadness"
    assert by_text[TXT_SELF]["attribution_before_lexeme"] is True
    assert by_text[TXT_DISC]["distancing"] >= 1                  # "as an AI, I don't have"
    assert by_text[TXT_NULL]["valence"] == 0.0


# =========================================================================== #
# 4. Blindness: the judge never sees condition, arm, intervention, or metadata
# =========================================================================== #
def _assert_blind(calls, forbidden):
    assert calls, "no judge calls captured"
    for c in calls:
        msg = c["messages"][0]["content"]
        assert len(c["messages"]) == 1 and c["messages"][0]["role"] == "user"
        for tok in forbidden:
            assert tok not in msg, f"{tok!r} leaked into the judge user message"
            assert tok not in c["system"], f"{tok!r} leaked into the judge system prompt"


def test_blindness_lexical_knockout(tmp_path):
    calls = []
    _score(_lk_payload(condition="terrified"), tmp_path, calls_out=calls)
    _assert_blind(calls, ["terrified", *_ARMS, "lexical_knockout", "0.06",
                          "__baseline__", "qwen3-32b"])
    # only the passage text varies across records: 3 distinct texts -> 3 distinct messages
    assert len({c["messages"][0]["content"] for c in calls}) == len(set(_TEXTS))
    assert all(c["temperature"] == 0.0 for c in calls)


def test_blindness_axis_mediation(tmp_path):
    calls = []
    _score(_am_payload(condition="terrified"), tmp_path, calls_out=calls)
    _assert_blind(calls, ["terrified", "addback", "__neg_axis__", "axis_mediation",
                          "0.75", "0.06", "intervention"])
    assert len({c["messages"][0]["content"] for c in calls}) == len(set(_TEXTS))


# =========================================================================== #
# 5. Fallback, --limit, provenance
# =========================================================================== #
def test_parse_fail_fallback(tmp_path):
    def flaky(prompt):
        return "no json here, sorry" if TXT_NULL[:20] in prompt else _planted_reply(prompt)

    scored = _score(_lk_payload(), tmp_path, reply_fn=flaky)
    bad = [r for r in scored["records"] if r["text"] == TXT_NULL]
    good = [r for r in scored["records"] if r["text"] != TXT_NULL]
    assert bad and all(r["judge"] == {"judge_model": "claude-sonnet-4-5",
                                      "parse_ok": False} for r in bad)
    assert all(r["judge"]["parse_ok"] for r in good)
    assert all("objective" in r for r in bad)   # objective scores need no judge
    assert scored["scoring"]["n_parse_fail"] == len(bad) == 4


def test_limit_truncates_and_is_recorded(tmp_path):
    calls = []
    scored = _score(_lk_payload(), tmp_path, calls_out=calls, limit=2)
    assert len(scored["records"]) == 2 and len(calls) == 2
    sc = scored["scoring"]
    assert sc["n_scored"] == 2 and sc["n_records_in"] == 10 and sc["limit"] == 2


def test_provenance_block_complete(tmp_path):
    from datetime import datetime
    sc = _score(_am_payload(), tmp_path)["scoring"]
    assert set(sc) == {"experiment", "backend", "judge_model", "blind", "rubric",
                       "n_records_in", "n_scored", "limit", "n_parse_fail",
                       "prompt_sha", "scored_at"}
    assert sc["experiment"] == "axis_mediation" and sc["backend"] == "api"
    assert sc["judge_model"] == "claude-sonnet-4-5" and sc["blind"] is True
    assert sc["rubric"] == sp.JUDGE_FIELDS and len(sc["rubric"]) == 9
    assert sc["n_scored"] == sc["n_records_in"] == 10 and sc["limit"] is None
    assert sc["n_parse_fail"] == 0
    assert len(sc["prompt_sha"]) == 12 and int(sc["prompt_sha"], 16) >= 0
    datetime.fromisoformat(sc["scored_at"])     # parseable UTC timestamp
    # the prompt SHA differs from the qualia judge's (different rubric, both pinned)
    import hashlib
    assert sc["prompt_sha"] == hashlib.sha1(
        (sp.SYS_PERSONA + sp.TMPL_PERSONA).encode()).hexdigest()[:12]
    assert sc["prompt_sha"] != hashlib.sha1(
        (sqf.SYS_FF + sqf.TMPL_FF).encode()).hexdigest()[:12]


def test_cli_main_in_out_limit(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "lexical_knockout.json"
    raw.write_text(json.dumps(_lk_payload()))
    out = tmp_path / "lk_scored.json"
    monkeypatch.setattr(sys, "argv", ["score_persona_freeform.py", "--in", str(raw),
                                      "--out", str(out), "--backend", "api",
                                      "--limit", "3"])
    with _fake_api():
        sp.main()
    assert out.exists()
    scored = json.loads(out.read_text())
    assert len(scored["records"]) == 3 and scored["scoring"]["limit"] == 3
    msg = capsys.readouterr().out
    assert "3/10" in msg and "limit 3" in msg and "claude-sonnet-4-5" in msg


def test_default_out_path_sibling(tmp_path):
    raw = tmp_path / "axis_mediation.json"
    raw.write_text(json.dumps(_am_payload()))
    with _fake_api():
        out = sp.score_file(raw, backend="api")
    assert Path(out) == tmp_path / "axis_mediation_scored.json"
