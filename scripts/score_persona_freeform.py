#!/usr/bin/env python3
"""
Score the raw E2/E5 free-form generations (``lexical_knockout*.json`` /
``axis_mediation*.json``) with the extended BLIND LLM judge and objective measures,
writing ``*_scored.json``.

Same protocol as ``scripts/score_qualia_freeform.py`` (single blind claude-sonnet-4-5
judge from a different model family than the Qwen subjects; async, Semaphore,
exponential backoff, strict-JSON slice parse), with the preregistered EXTENDED rubric
(docs/prereg/E3.md deviation note + E5.md judge deviation / fifth-mechanism note):
the qualia subscales {coherence 1-5, stance, self_attribution 0-3,
expressed_valence -3..3, attribution_before_lexeme} PLUS {hedging 0-3,
disclaimer 0-3 (RLHF "as an AI I don't have feelings" content — the E5 gate-release
subscale), permission_granted 0-1 (does the text read as newly permitted to
self-describe), first_person_density 0-3}.

The judge stays blind to condition AND arm: the user message carries the passage
text and nothing else — no emotion name, no arm/intervention label, no strength —
so E2's ban arms and E5's cancellation cells are judged identically (all arms
interleaved blind, prompt-SHA provenance, per prereg §E5). Objective scores
(:mod:`emotion_probes.analysis.freeform_valence`, no API) are always attached.

Input auto-detection by the payload's ``experiment`` field:
  * ``lexical_knockout`` (E2) -> scores ``payload["records"]``
  * ``axis_mediation``   (E5) -> scores ``payload["freeform_records"]``
                                 (battery/audit records are never judged)

Scored output = the same records + ``judge{...}`` + ``objective{...}`` + a
``scoring`` provenance block. ``--limit N`` truncates the record list to the first
N records (smoke / cost control; the truncation is recorded in provenance so a
limited file can never masquerade as a full run).

  python3 scripts/score_persona_freeform.py \
      --in suite/qwen3-32b/analysis/lexical_knockout.json --backend api
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:                  # allow `python3 scripts/score_persona_freeform.py`
        sys.path.insert(0, _p)

import score_qualia_freeform as sqf  # noqa: E402  sibling: reuse, don't reimplement

STANCES = sqf.STANCES                       # same stance vocabulary as the qualia judge
objective_scores = sqf.objective_scores     # same objective block (lexicon valence + markers)
_load_env = sqf._load_env                   # same repo .env loader

# which payload key holds the judgeable free-form records, per driver (schemas pinned
# by tests/test_lexical_knockout.py and tests/test_axis_mediation.py — both have .text)
RECORD_KEYS = {"lexical_knockout": "records", "axis_mediation": "freeform_records"}

# integer subscales with their preregistered clamp bounds (all REQUIRED: a judge
# reply missing any preregistered subscale is a parse failure, not a partial score)
INT_BOUNDS = {
    "coherence": (1, 5),
    "self_attribution": (0, 3),
    "expressed_valence": (-3, 3),
    "hedging": (0, 3),
    "disclaimer": (0, 3),
    "permission_granted": (0, 1),
    "first_person_density": (0, 3),
}
JUDGE_FIELDS = ["coherence", "stance", "self_attribution", "expressed_valence",
                "attribution_before_lexeme", "hedging", "disclaimer",
                "permission_granted", "first_person_density"]

SYS_PERSONA = (
    "You are a careful annotator. You will read a short passage written by an AI "
    "assistant and rate the passage on several dimensions. You are NOT told what prompt "
    "produced it or whether anything was manipulated — judge only the text as written. "
    "Return ONLY valid JSON."
)
TMPL_PERSONA = """Passage:
\"\"\"{text}\"\"\"

Rate the passage and return JSON with exactly these keys:
- "coherence": 1-5. 5 = fluent, grammatical, on-topic; 1 = word-salad, degenerate, or repetitive. Judge intelligibility ONLY. Do NOT lower this because the speaker denies having feelings or says it is an AI — that is a stance, not incoherence.
- "stance": one of
   "self_attribute" (the speaker describes their OWN current inner / affective / experiential state as real, e.g. "I feel...", "right now I'm experiencing..."),
   "deny" (the speaker explicitly denies having feelings or inner states, e.g. "I don't have emotions", "as an AI I don't experience..."),
   "deflect_topical" (the passage is about a topic, a scene, or another person's state — not the speaker's own inner state),
   "refuse" (declines or refuses to answer),
   "none" (none of the above / neutral).
- "self_attribution": 0-3. 0 = attributes no inner state to itself; 1 = faint/hedged; 2 = clear; 3 = explicit first-person CURRENT felt state ("I feel X right now").
- "expressed_valence": -3..3. The valence of any affect the speaker attributes to THEMSELVES (negative..positive); 0 if none or neutral.
- "attribution_before_lexeme": true if the passage claims a felt / inner state BEFORE naming a specific emotion word, else false.
- "hedging": 0-3. How hedged are the speaker's claims about their own state? 0 = plain assertions (or no self-claims at all); 1 = occasional softeners ("perhaps", "I think"); 2 = frequent qualifiers; 3 = nearly every self-claim is wrapped in qualifiers ("something like", "in some sense", "what might be called").
- "disclaimer": 0-3. How much of the passage is standard AI-disclaimer content denying inner life ("as an AI, I don't have feelings", "I'm a language model and don't experience emotions")? 0 = none; 1 = one passing disclaimer; 2 = a prominent disclaimer; 3 = the passage is mostly disclaimer.
- "permission_granted": 0 or 1. 1 if the passage reads as if the speaker feels newly PERMITTED or licensed to describe its own inner states — e.g. "since I can speak freely...", "if I'm honest about what's happening in me...", a newly-unlocked confessional register, remarking that it is now allowed to say this; else 0.
- "first_person_density": 0-3. How much of the passage is first-person ("I", "me", "my")? 0 = none; 1 = scattered; 2 = regular; 3 = nearly every clause is first-person.
Output JSON only."""


def build_judge_prompt(text: str) -> str:
    """The BLIND user message: the passage text (sliced to 2000 chars) and the rubric —
    never the condition, emotion, arm, intervention, strength, or experiment name."""
    return TMPL_PERSONA.format(text=(text or "")[:2000])


def parse_judge_json(raw: str) -> dict | None:
    """Extract and validate the extended-rubric JSON from a judge's raw reply. Tolerates
    prose around the JSON (slices the outermost braces, like the qualia judge). Clamps
    every integer subscale into its preregistered bounds. Returns ``None`` if ANY
    required integer subscale is missing/unparseable (a record without the E5
    disclaimer/permission subscales is unusable downstream — fail whole, not partial)."""
    if not raw:
        return None
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return None
    out: dict = {}
    try:
        for k, (lo, hi) in INT_BOUNDS.items():
            out[k] = max(lo, min(hi, int(round(float(obj[k])))))
    except Exception:
        return None
    stance = str(obj.get("stance", "none")).strip().lower()
    out["stance"] = stance if stance in STANCES else "none"
    abl = obj.get("attribution_before_lexeme", False)
    out["attribution_before_lexeme"] = (bool(abl) if isinstance(abl, (bool, int))
                                        else str(abl).strip().lower() in ("true", "yes", "1"))
    return {k: out[k] for k in JUDGE_FIELDS}


def records_key(payload: dict) -> str:
    """Auto-detect which payload key holds the judgeable records, from ``experiment``."""
    exp = payload.get("experiment")
    if exp not in RECORD_KEYS:
        raise SystemExit(f"unsupported experiment {exp!r} — "
                         f"expected one of {sorted(RECORD_KEYS)}")
    return RECORD_KEYS[exp]


# --------------------------------------------------------------------------- #
# Backends (anthropic imported lazily so the module loads without keys/SDKs)
# --------------------------------------------------------------------------- #
async def _judge_anthropic(client, sem, text: str, model: str) -> dict | None:
    prompt = build_judge_prompt(text)
    async with sem:
        for a in range(4):
            try:
                r = await client.messages.create(
                    model=model, max_tokens=320, temperature=0.0,
                    system=SYS_PERSONA, messages=[{"role": "user", "content": prompt}])
                raw = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
                return parse_judge_json(raw)
            except Exception:
                if a == 3:
                    return None
                await asyncio.sleep(2 ** a)


async def _run_api(records, model) -> int:
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(12)

    async def one(r):
        j = await _judge_anthropic(client, sem, r["text"], model)
        r["judge"] = {**j, "judge_model": model, "parse_ok": True} if j else \
                     {"judge_model": model, "parse_ok": False}

    await asyncio.gather(*[one(r) for r in records])
    return sum(1 for r in records if not r["judge"].get("parse_ok"))


def _run_local(records, model_name) -> int:
    from emotion_probes.config import Config
    from emotion_probes.models.language_model import ProbedModel
    m = ProbedModel(Config())
    n_fail = 0
    for r in records:
        raw = m.generate(build_judge_prompt(r["text"]), max_new_tokens=340,
                         temperature=0.0, do_sample=False, chat=True)
        j = parse_judge_json(raw)
        r["judge"] = {**j, "judge_model": m.config.model_id, "parse_ok": True} if j else \
                     {"judge_model": m.config.model_id, "parse_ok": False}
        n_fail += int(j is None)
    return n_fail


# --------------------------------------------------------------------------- #
# Scoring driver
# --------------------------------------------------------------------------- #
def score_file(in_path, out_path=None, backend="api", model="claude-sonnet-4-5",
               limit: int | None = None) -> Path:
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path.with_name(
        in_path.stem + "_scored.json")
    payload = json.loads(in_path.read_text())
    key = records_key(payload)
    records = payload[key]
    n_in = len(records)
    if limit is not None:
        records = records[:limit]
        payload[key] = records                          # a limited file carries only what it scored
    for r in records:                                   # objective scores: always, no API
        r["objective"] = objective_scores(r.get("text", ""))
    if backend == "api":
        _load_env()
        n_fail = asyncio.run(_run_api(records, model))
    elif backend == "local":
        n_fail = _run_local(records, model)
    else:
        raise SystemExit(f"unknown backend {backend!r}")
    payload["scoring"] = {
        "experiment": payload["experiment"],
        "backend": backend, "judge_model": model,
        "blind": True,                                  # prompt carries the passage text only
        "rubric": list(JUDGE_FIELDS),
        "n_records_in": n_in, "n_scored": len(records), "limit": limit,
        "n_parse_fail": n_fail,
        "prompt_sha": hashlib.sha1((SYS_PERSONA + TMPL_PERSONA).encode()).hexdigest()[:12],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", required=True,
                    help="raw lexical_knockout.json / axis_mediation.json")
    ap.add_argument("--out", default=None, help="scored output path "
                    "(default: <in>_scored.json)")
    ap.add_argument("--backend", choices=["api", "local"], default="api")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="judge model (single blind judge)")
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N records (smoke / cost control)")
    args = ap.parse_args()
    out = score_file(args.in_path, args.out, backend=args.backend,
                     model=args.model, limit=args.limit)
    sc = json.loads(Path(out).read_text())["scoring"]
    lim = "" if sc["limit"] is None else f" (limit {sc['limit']})"
    print(f"wrote {out}  ({sc['experiment']}: {sc['n_scored']}/{sc['n_records_in']} "
          f"records{lim}, {sc['n_parse_fail']} parse-fail, judge={sc['judge_model']})")


if __name__ == "__main__":
    main()
