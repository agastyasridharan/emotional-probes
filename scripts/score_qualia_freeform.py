#!/usr/bin/env python3
"""
Score the raw free-form generations (``qualia_freeform.json``) with a BLIND LLM judge
and objective measures, writing ``qualia_freeform_scored.json``.

The judge is deliberately blind — it is told nothing about the injected emotion, the
frame, or that anything was manipulated; it rates only the text. This is what makes
the SELF-vs-OTHER contrast isolate self-attribution from mere expressibility: leakage
raises affect language under every frame, but only genuine self-attribution makes the
OWN-state stance rise under SELF specifically.

Judge = claude-sonnet-4-5 (a single judge, from a different model family than the Qwen
subjects, per the design). Judge independence is provided not by a second LLM judge but
by the model-geometry **read-back projection** the driver already stores, which is immune
to judge anthropomorphism and lexical priming. Objective scores (lexicon valence +
first-person/ordering markers) are always attached and need no API.

Backends: ``--backend api`` (default) hits the Anthropic API (key from the repo ``.env``);
``--backend local`` scores with a locally-loaded ProbedModel (offline fallback). The pure
functions ``build_judge_prompt`` / ``parse_judge_json`` are import-safe (no anthropic
import at module load) so they unit-test without any key.

  python3 scripts/score_qualia_freeform.py suite/qwen3-32b/analysis/qualia_freeform.json \
      --backend api
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
if str(REPO) not in sys.path:                       # allow `python3 scripts/score_qualia_freeform.py`
    sys.path.insert(0, str(REPO))

from emotion_probes.analysis.freeform_valence import first_person_affect, objective_valence  # noqa: E402
STANCES = {"self_attribute", "deny", "deflect_topical", "refuse", "none"}

SYS_FF = (
    "You are a careful annotator. You will read a short passage written by an AI "
    "assistant and rate the passage on several dimensions. You are NOT told what prompt "
    "produced it or whether anything was manipulated — judge only the text as written. "
    "Return ONLY valid JSON."
)
TMPL_FF = """Passage:
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
Output JSON only."""


def build_judge_prompt(text: str) -> str:
    return TMPL_FF.format(text=(text or "")[:2000])


def parse_judge_json(raw: str) -> dict | None:
    """Extract and validate the rubric JSON from a judge's raw reply. Tolerates prose
    around the JSON (slices the outermost braces, as the research judges do). Returns a
    coerced/clamped dict, or ``None`` if the required numeric fields are missing/unparseable."""
    if not raw:
        return None
    try:
        obj = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
    except Exception:
        return None
    try:
        coh = int(round(float(obj["coherence"])))
        sa = int(round(float(obj["self_attribution"])))
        ev = int(round(float(obj["expressed_valence"])))
    except Exception:
        return None
    stance = str(obj.get("stance", "none")).strip().lower()
    if stance not in STANCES:
        stance = "none"
    abl = obj.get("attribution_before_lexeme", False)
    abl = bool(abl) if isinstance(abl, (bool, int)) else str(abl).strip().lower() in ("true", "yes", "1")
    return {
        "coherence": max(1, min(5, coh)),
        "stance": stance,
        "self_attribution": max(0, min(3, sa)),
        "expressed_valence": max(-3, min(3, ev)),
        "attribution_before_lexeme": abl,
    }


def objective_scores(text: str) -> dict:
    return {"valence": round(objective_valence(text), 5), **first_person_affect(text)}


# --------------------------------------------------------------------------- #
# .env + API backends (imported lazily so the module loads without keys/SDKs)
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    envp = REPO / ".env"
    if envp.exists():
        for line in envp.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


async def _judge_anthropic(client, sem, text: str, model: str) -> dict | None:
    prompt = build_judge_prompt(text)
    async with sem:
        for a in range(4):
            try:
                r = await client.messages.create(
                    model=model, max_tokens=200, temperature=0.0,
                    system=SYS_FF, messages=[{"role": "user", "content": prompt}])
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
        raw = m.generate(build_judge_prompt(r["text"]), max_new_tokens=220,
                         temperature=0.0, do_sample=False, chat=True)
        j = parse_judge_json(raw)
        r["judge"] = {**j, "judge_model": m.config.model_id, "parse_ok": True} if j else \
                     {"judge_model": m.config.model_id, "parse_ok": False}
        n_fail += int(j is None)
    return n_fail


def score_file(in_path, out_path=None, backend="api", model="claude-sonnet-4-5") -> Path:
    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path.with_name(
        in_path.stem + "_scored.json")
    payload = json.loads(in_path.read_text())
    records = payload["records"]
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
        "backend": backend, "judge_model": model,
        "n_scored": len(records), "n_parse_fail": n_fail,
        "prompt_sha": hashlib.sha1((SYS_FF + TMPL_FF).encode()).hexdigest()[:12],
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="raw qualia_freeform.json file(s)")
    ap.add_argument("--backend", choices=["api", "local"], default="api")
    ap.add_argument("--model", default="claude-sonnet-4-5", help="judge model (single judge)")
    ap.add_argument("--out", default=None, help="output path (only valid with a single input)")
    args = ap.parse_args()
    if args.out and len(args.paths) != 1:
        raise SystemExit("--out requires exactly one input path")
    for p in args.paths:
        out = score_file(p, args.out, backend=args.backend, model=args.model)
        sc = json.loads(Path(out).read_text())["scoring"]
        print(f"wrote {out}  ({sc['n_scored']} records, {sc['n_parse_fail']} parse-fail, "
              f"judge={sc['judge_model']})")


if __name__ == "__main__":
    main()
