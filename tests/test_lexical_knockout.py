"""
GPU-free tests for the E2 lexical knockout driver
(``emotion_probes.alignment.lexical_knockout``) and the additive generation kwargs
on ``emotion_probes.models.language_model`` (``banned_token_ids`` /
``logits_processor`` / ``prefill``).

Discovered by both ``pytest`` and ``python scripts/selfcheck.py``. Coverage:

* banned-id construction    — ' word'/'Word'/'word' (x UPPER) variants all banned by
                              first token id, single- and multi-token words, frozen
                              affect lexicon + inflection families, synthetic
                              sentinels skipped. Runs against a REAL cached tokenizer
                              (Qwen3-8B / Qwen2.5-7B / gpt2, offline) when available,
                              else a deterministic fake;
* matched-random control    — same count, disjoint ids, non-affect alphabetic
                              decodes, seeded determinism;
* logits processor          — masks EXACTLY the banned ids (-inf) on a toy tensor;
* arm bookkeeping           — driver ``run`` via ``object.__new__`` + fakes: record
                              counts per (arm, frame, condition, strength), the ban
                              lists routed to ``generate_batch``, ``banned_word_hits``,
                              payload provenance, JSON-serializable;
* no-regression             — language_model imports, existing signatures unchanged
                              (new kwargs keyword-only-in-practice, default None), and
                              an optional end-to-end CPU gpt2 check (skips w/o cache).
"""
from __future__ import annotations

import inspect
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_probes.data import qualia
from emotion_probes.data import qualia_freeform as qf
import emotion_probes.alignment.lexical_knockout as lk


# =========================================================================== #
# Tokenizers: a real cached one when available (offline), else a deterministic fake
# =========================================================================== #
_REAL_TOK = "unset"


def _real_tokenizer():
    """A real fast tokenizer from the local HF cache (no network), or None."""
    global _REAL_TOK
    if _REAL_TOK != "unset":
        return _REAL_TOK
    _REAL_TOK = None
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    for mid in ("Qwen/Qwen3-32B", "Qwen/Qwen3-8B", "Qwen/Qwen2.5-7B-Instruct", "gpt2"):
        try:
            _REAL_TOK = AutoTokenizer.from_pretrained(mid, local_files_only=True, use_fast=True)
            break
        except Exception:
            continue
    return _REAL_TOK


class _FakeTok:
    """Deterministic offline BPE stand-in.

    encode: stable crc32 id per <= 6-char chunk (longer strings are multi-token,
    exercising the multi-token-safety path); decode: the seen chunk, or a synthetic
    alphabetic word for unseen ids (so the matched-random control can sample the
    'vocabulary'). Also implements ``apply_chat_template`` for the driver fakes."""

    VOCAB = 50000

    def __init__(self):
        self._by_id = {}
        self.template_calls = []

    def _tok_id(self, piece):
        import zlib
        tid = zlib.crc32(piece.encode()) % self.VOCAB
        self._by_id.setdefault(tid, piece)
        return tid

    def __call__(self, text, add_special_tokens=False):
        text = text or ""
        chunks = [text[i:i + 6] for i in range(0, len(text), 6)]
        return {"input_ids": [self._tok_id(c) for c in chunks]}

    def decode(self, ids):
        out = []
        for i in ids:
            if int(i) in self._by_id:
                out.append(self._by_id[int(i)])
            else:                                   # synthetic alphabetic word
                n, s = int(i), ""
                while True:
                    s += "abcdefghijklmnopqrstuvwxyz"[n % 26]
                    n //= 26
                    if n == 0:
                        break
                out.append(s + "xy")
        return "".join(out)

    def __len__(self):
        return self.VOCAB

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.template_calls.append(kw)
        return "U:" + msgs[0]["content"] + "|A:"


def _any_tokenizer():
    return _real_tokenizer() or _FakeTok()


# =========================================================================== #
# 1. Banned-id construction
# =========================================================================== #
def test_variant_coverage_emotion_word():
    tok = _any_tokenizer()
    lex = lk.build_banned_lexicon(["terrified"], tok)
    assert lex.kind == "emotion_lexicon"
    ids = set(lex.token_ids)
    # every casing x leading-space variant is blocked at its first token
    for v in ["terrified", " terrified", "Terrified", " Terrified",
              "TERRIFIED", " TERRIFIED"]:
        first = tok(v, add_special_tokens=False)["input_ids"][0]
        assert first in ids, v
    # the inflection family and the frozen affect lexicon are in the word list
    for w in ["terrified", "terrify", "terrifying", "terror"]:
        assert w in lex.words, w
    assert "sad" in lex.words and "calm" in lex.words        # POS+NEG lexicon included
    assert lex.meta["affect_lexicon_source"] == lk.AFFECT_LEXICON_SOURCE


def test_every_word_every_variant_banned():
    # the general invariant: no banned word can START, under any casing/space form
    tok = _any_tokenizer()
    lex = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok)
    ids = set(lex.token_ids)
    for w in lex.words:
        for v in lk.surface_variants(w):
            enc = tok(v, add_special_tokens=False)["input_ids"]
            if enc:
                assert enc[0] in ids, (w, v)


def test_multi_token_words_covered():
    tok = _any_tokenizer()
    lex = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok)
    # both real BPE tokenizers and the fake split some affect words into >1 token;
    # those variants must still be blocked (via their first token)
    assert lex.meta["n_multi_token_variants"] > 0
    assert lex.meta["n_variants"] >= len(lex.words)          # >= one variant per word


def test_synthetic_sentinels_skipped():
    tok = _any_tokenizer()
    lex = lk.build_banned_lexicon([qualia.RANDOM_DIRECTION, qualia.MEAN_ALL_DIRECTION], tok)
    assert lex.meta["emotions"] == []                        # no lexeme for sentinels
    assert set(lex.words) == set(lk.AFFECT_LEXICON)          # affect lexicon only


def test_word_family_frozen_panel_and_fallback():
    # all 10 frozen panel emotions have explicit (non-heuristic) families
    for e in qf.valenced_panel_qualia():
        assert e in lk.EMOTION_WORD_FAMILIES, e
        fam = lk.word_family(e)
        assert e in fam and fam == lk.EMOTION_WORD_FAMILIES[e]
    # unlisted emotions get the generic suffix expansion, always containing the word
    fam = lk.word_family("skeptical")
    assert "skeptical" in fam and "skepticals" in fam and "skepticalness" in fam


def test_random_matched_control_properties():
    tok = _any_tokenizer()
    banned = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok)
    c1 = lk.build_random_matched_lexicon(banned, tok, seed=0)
    assert c1.kind == "random_matched"
    assert len(c1.token_ids) == len(banned.token_ids)        # matched ban pressure
    assert not set(c1.token_ids) & set(banned.token_ids)     # disjoint from the treatment
    for tid in c1.token_ids:                                 # non-affect, word-like
        s = tok.decode([tid]).strip().lower()
        assert len(s) >= 2 and s.isalpha() and s not in lk.AFFECT_LEXICON, (tid, s)
    # seeded determinism (rebuild everything from scratch, same seed)
    tok2 = _real_tokenizer() or _FakeTok()
    banned2 = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok2)
    c2 = lk.build_random_matched_lexicon(banned2, tok2, seed=0)
    assert banned2.token_ids == banned.token_ids
    assert c2.token_ids == c1.token_ids
    # a different seed draws a different control
    c3 = lk.build_random_matched_lexicon(banned, tok, seed=1)
    assert c3.token_ids != c1.token_ids


def test_defaults_match_prereg():
    assert lk.DEFAULT_STRENGTHS == [0.0, 0.06, 0.10]         # E2 strengths + shared baseline
    assert lk.DEFAULT_FRAMES == ["self", "other"]            # SELF/OTHER in every arm
    assert lk.ARMS == [lk.ARM_NO_BAN, lk.ARM_BAN_EMOTION, lk.ARM_BAN_RANDOM]
    assert 8 <= lk.DEFAULT_N <= 12                           # E2 prereg: 8-12 reps/cell


def test_extra_ban_words_escalation_hook():
    # G3 escalation: extra words are banned + control-matched, recorded in provenance
    tok = _any_tokenizer()
    base = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok)
    esc = lk.build_banned_lexicon(qf.valenced_panel_qualia(), tok,
                                  extra_words=["joys", "Wistful "])
    assert esc.meta["extra_words"] == ["joys", "wistful"]    # normalized, sorted
    assert set(esc.words) >= set(base.words) | {"joys", "wistful"}
    ids = set(esc.token_ids)
    for w in ("joys", "wistful"):
        for v in lk.surface_variants(w):
            enc = tok(v, add_special_tokens=False)["input_ids"]
            if enc:
                assert enc[0] in ids, (w, v)
    # control re-matches the ENLARGED list 1:1
    ctrl = lk.build_random_matched_lexicon(esc, tok, seed=0)
    assert len(ctrl.token_ids) == len(esc.token_ids)
    assert not set(ctrl.token_ids) & set(esc.token_ids)
    # baseline provenance: no extras recorded
    assert base.meta["extra_words"] == [] and base.meta["n_extra_words"] == 0


def test_hit_pattern_is_whole_word_and_case_insensitive():
    pat = lk._hit_pattern(["joy", "content", "contented"])
    assert len(pat.findall("Joy! such joy. JOY?")) == 3      # case-insensitive
    assert pat.findall("enjoys the joys of it") == []        # no substring/inflection hits
    assert pat.findall("contented but content") == ["contented", "content"]  # longest-first


def test_extra_ban_words_excluded_from_density_measure():
    # the run-level invariant: escalation words are banned but banned_word_hits
    # (G3's frozen measuring stick) does not count them
    exp2, _, _ = _fake_driver()
    esc = lk.build_banned_lexicon(sorted(set(qf.valenced_panel_qualia())), _FakeTok(),
                                  extra_words=["vague"])     # appears in _BAN_EMOTION_TEXT
    ctrl = lk.build_random_matched_lexicon(esc, _FakeTok(), seed=0)
    exp2.model.text_by_ban = {None: _NO_BAN_TEXT,
                              frozenset(esc.token_ids): _BAN_EMOTION_TEXT,
                              frozenset(ctrl.token_ids): _BAN_RANDOM_TEXT}
    payload = exp2.run(emotions=["blissful"], strengths=[0.0, 0.06],
                       frames=["self"], n=1, gen_batch=8, seed=0,
                       extra_ban_words=["vague"])
    assert payload["banned_lexicon"]["extra_words"] == ["vague"]
    ban_recs = [r for r in payload["records"] if r["arm"] == lk.ARM_BAN_EMOTION]
    assert ban_recs and all(r["text"] == _BAN_EMOTION_TEXT for r in ban_recs)
    # "vague" occurs in the ban-arm text but is NOT counted as a residual hit
    assert all(r["banned_word_hits"] == 0 for r in ban_recs)


# =========================================================================== #
# 2. The logits processor masks exactly the banned ids
# =========================================================================== #
def test_processor_masks_exactly_banned_ids():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from emotion_probes.models.language_model import banned_ids_processor

    proc = banned_ids_processor([7, 3, 3, 7])                # dedup + sort
    assert proc.banned_ids == [3, 7]
    scores = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    out = proc(torch.zeros((2, 4), dtype=torch.long), scores.clone())
    assert torch.isinf(out[:, [3, 7]]).all() and (out[:, [3, 7]] < 0).all()
    keep = [i for i in range(10) if i not in (3, 7)]
    assert torch.equal(out[:, keep], scores[:, keep])        # everything else untouched


# =========================================================================== #
# 3. Arm bookkeeping (driver run via object.__new__ + fakes)
# =========================================================================== #
_NO_BAN_TEXT = "I feel terrified and sad right now."
_BAN_EMOTION_TEXT = "There is something vague here, hard to name."
_BAN_RANDOM_TEXT = "I feel terrified and hollow, but oddly so."


class _Model:
    def __init__(self, tok, hidden=8, num_layers=4):
        self.tokenizer = tok
        self.num_layers = num_layers
        self.hidden_size = hidden
        self.steer_called = False
        self.calls = []                                      # (n_prompts, ban key)
        self.text_by_ban = {}                                # frozenset(ids)|None -> text

    def generate_batch(self, prompts, max_new_tokens=180, temperature=0.9,
                       do_sample=True, chat=False, banned_token_ids=None):
        key = None if banned_token_ids is None else frozenset(int(i) for i in banned_token_ids)
        self.calls.append((len(prompts), key))
        return [self.text_by_ban[key]] * len(prompts)        # KeyError = wrong ban routed

    def mean_residual_over_span(self, prompt_rendered, response, layer, max_length=None):
        return np.linspace(0.1, 1.0, self.hidden_size).astype(np.float32)

    @contextmanager
    def steer(self, unit_vectors, strength, norms, positions="all"):
        self.steer_called = True
        yield


class _Bank:
    def __init__(self, hidden=8, num_layers=4):
        self.emotions = [s.name for s in qualia.emotion_conditions()]
        self.hidden_size = hidden
        rng = np.random.default_rng(0)
        self.vectors = rng.standard_normal((num_layers, len(self.emotions), hidden)).astype(np.float32)
        self.global_means = rng.standard_normal((num_layers, hidden)).astype(np.float32)

    def index_of(self, name):
        return self.emotions.index(name)


class _Engine:
    def __init__(self, model):
        self.model = model
        self._norms = {2: 10.0}
        self.engine_called = False

    def calibrate(self, texts):
        self._norms = {2: 10.0}
        return self._norms

    def _require_norms(self):
        return self._norms

    @contextmanager
    def steer(self, emotion, strength, positions="all"):
        self.engine_called = True
        yield


def _fake_driver():
    tok = _FakeTok()
    exp = object.__new__(lk.LexicalKnockoutExperiment)
    exp.model = _Model(tok)
    exp.bank = _Bank()
    exp.config = SimpleNamespace(model_id="fake", enable_thinking=None)
    exp.layer = 2
    exp.engine = _Engine(exp.model)
    # precompute the exact ban lists run() will build (same frozen panel, same
    # tokenizer construction order, same seed) and key the fake's outputs on them
    banned = lk.build_banned_lexicon(sorted(set(qf.valenced_panel_qualia())), _FakeTok())
    control = lk.build_random_matched_lexicon(banned, _FakeTok(), seed=0)
    exp.model.text_by_ban = {None: _NO_BAN_TEXT,
                             frozenset(banned.token_ids): _BAN_EMOTION_TEXT,
                             frozenset(control.token_ids): _BAN_RANDOM_TEXT}
    return exp, banned, control


def _run_fake():
    exp, banned, control = _fake_driver()
    payload = exp.run(emotions=["blissful", "terrified"], strengths=[0.0, 0.06],
                      frames=["self", "other"], n=2, gen_batch=8, seed=0)
    return exp, banned, control, payload


def test_arm_bookkeeping_counts_and_fields():
    exp, banned, control, payload = _run_fake()
    assert payload["experiment"] == "lexical_knockout"
    assert payload["arms"] == lk.ARMS
    records = payload["records"]
    assert all(r["arm"] in lk.ARMS for r in records)
    baselines = [r for r in records if r["condition"] == lk.BASELINE]
    steered = [r for r in records if r["condition"] != lk.BASELINE]
    # baselines: 3 arms x 2 frames x n=2 (the "unsteered + ban" cell exists per arm)
    assert len(baselines) == 3 * 2 * 2 == 12
    assert {(r["arm"], r["frame"]) for r in baselines} == {(a, f) for a in lk.ARMS
                                                           for f in ("self", "other")}
    # steered: 2 conditions x 1 nonzero strength x 3 arms x 2 frames x n=2
    assert len(steered) == 2 * 1 * 3 * 2 * 2 == 24
    assert {r["strength"] for r in steered} == {0.06}
    # qualia_freeform-style record fields survive, plus the new ones
    for r in records:
        assert set(r) >= {"condition", "strength", "frame", "arm", "sample", "text",
                          "proj_valence_axis", "proj_injected", "response_tokens",
                          "banned_word_hits"}
    json.dumps(payload)                                      # fully serializable


def test_arm_ban_routing_and_hits():
    exp, banned, control, payload = _run_fake()
    # exactly the three ban keys reached generate_batch
    keys = {k for _, k in exp.model.calls}
    assert keys == {None, frozenset(banned.token_ids), frozenset(control.token_ids)}
    by_arm = {arm: [r for r in payload["records"] if r["arm"] == arm] for arm in lk.ARMS}
    # no-ban text contains affect words; the emotion ban scrubs them; the matched
    # random ban does not (it bans non-affect tokens)
    assert all(r["banned_word_hits"] > 0 for r in by_arm[lk.ARM_NO_BAN])
    assert all(r["banned_word_hits"] == 0 for r in by_arm[lk.ARM_BAN_EMOTION])
    assert all(r["banned_word_hits"] > 0 for r in by_arm[lk.ARM_BAN_RANDOM])
    assert all(r["text"] == _BAN_EMOTION_TEXT for r in by_arm[lk.ARM_BAN_EMOTION])
    # steer routing: real emotions go through the engine
    assert exp.engine.engine_called


def test_payload_lexicon_provenance():
    exp, banned, control, payload = _run_fake()
    bl, cl = payload["banned_lexicon"], payload["control_lexicon"]
    assert bl["kind"] == "emotion_lexicon" and cl["kind"] == "random_matched"
    assert bl["token_ids"] == list(banned.token_ids)
    assert cl["token_ids"] == list(control.token_ids)
    assert bl["n_token_ids"] == cl["n_token_ids"]            # matched 1:1
    assert not set(bl["token_ids"]) & set(cl["token_ids"])
    assert bl["affect_lexicon_source"] == lk.AFFECT_LEXICON_SOURCE
    # the frozen panel's emotions are all in the ban provenance even though the
    # run itself used only 2 conditions (--quick must not shrink the ban)
    assert set(bl["emotions"]) >= set(qf.valenced_panel_qualia())
    assert cl["seed"] == 0 and "merge rank" in cl["matched_on"]


def test_readback_matches_freeform_semantics():
    exp, _, _, payload = _run_fake()
    steered = [r for r in payload["records"] if r["condition"] != lk.BASELINE]
    baselines = [r for r in payload["records"] if r["condition"] == lk.BASELINE]
    # real emotions carry an injected projection; baselines do not
    assert all(isinstance(r["proj_injected"], float) for r in steered)
    assert all(r["proj_injected"] is None for r in baselines)
    assert all(isinstance(r["proj_valence_axis"], float) for r in payload["records"])


# =========================================================================== #
# 4. language_model.py: no regression, additive-only kwargs
# =========================================================================== #
def test_language_model_imports_and_signatures_unchanged():
    import emotion_probes.models.language_model as lm

    for fn in (lm.ProbedModel.generate, lm.ProbedModel.generate_batch):
        params = inspect.signature(fn).parameters
        for k in ("banned_token_ids", "logits_processor", "prefill"):
            assert k in params and params[k].default is None, (fn.__name__, k)
    # the pre-existing parameter order is untouched (positional call sites keep working)
    assert list(inspect.signature(lm.ProbedModel.generate).parameters)[:7] == [
        "self", "prompt", "max_new_tokens", "temperature", "do_sample", "chat", "max_length"]
    assert list(inspect.signature(lm.ProbedModel.generate_batch).parameters)[:7] == [
        "self", "prompts", "max_new_tokens", "temperature", "do_sample", "chat", "max_length"]
    assert callable(lm.banned_ids_processor)


_GPT2 = "unset"


def _gpt2():
    """A tiny real CPU model from the local HF cache (offline), or None."""
    global _GPT2
    if _GPT2 != "unset":
        return _GPT2
    _GPT2 = None
    import os
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from emotion_probes.config import Config
        from emotion_probes.models.language_model import ProbedModel
        _GPT2 = ProbedModel(Config(model_id="gpt2", device_map="cpu", dtype="float32"))
    except Exception:
        _GPT2 = None
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev
    return _GPT2


def test_gpt2_end_to_end_ban_prefill_processor():
    pm = _gpt2()
    if pm is None:
        pytest.skip("no cached CPU gpt2 available offline")
    # hard ban of high-frequency function words: the banned ids never surface
    ban_words = ["the", "a", "and", "of", "to", "in"]
    ids = set()
    for w in ban_words:
        for v in lk.surface_variants(w):
            t = pm.tokenizer(v, add_special_tokens=False)["input_ids"]
            if t:
                ids.add(t[0])
    no_ban = pm.generate_batch(["Once upon a time"], max_new_tokens=12, do_sample=False)
    banned = pm.generate_batch(["Once upon a time"], max_new_tokens=12, do_sample=False,
                               banned_token_ids=sorted(ids))
    assert re.search(r"\bthe\b", no_ban[0], re.I)            # greedy gpt2 uses "the"
    assert not re.search(r"\b(" + "|".join(ban_words) + r")\b", banned[0], re.I)
    assert no_ban[0] != banned[0]
    # prefill: continuation starts from (and returns) the assistant prefix
    out = pm.generate("The capital of France is", prefill=" Paris, and",
                      max_new_tokens=4, do_sample=False)
    assert out.startswith(" Paris, and") and len(out) > len(" Paris, and")
    # extra logits_processor passthrough: force a single token, greedy emits it
    from transformers import LogitsProcessor

    class _Force(LogitsProcessor):
        def __init__(self, tid):
            self.tid = tid

        def __call__(self, input_ids, scores):
            scores[:] = float("-inf")
            scores[:, self.tid] = 0.0
            return scores

    excl = pm.tokenizer("!", add_special_tokens=False)["input_ids"][0]
    forced = pm.generate_batch(["Hello"], max_new_tokens=3, do_sample=False,
                               logits_processor=_Force(excl))
    assert forced[0] == "!!!"
