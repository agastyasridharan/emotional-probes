"""
E2 lexical knockout — the hard-ban GPU driver (portfolio.md §E2, docs/prereg/E2.md).

Core arm 4 of the bidirectional lexical-priming kill test: repeat the free-form
self-attribution generation of :mod:`emotion_probes.alignment.qualia_freeform`
(same SELF/OTHER frames, same steering, same inline unsteered read-back projection)
while HARD-BANNING the affect lexicon at decode time, against two comparators:

* ``no_ban``              — v_full, the in-run positive control (gate G1);
* ``ban_emotion_lexicon`` — every decode step sets the affect-lexicon token ids to
                            -inf: each panel emotion word + its inflections + the
                            FROZEN affect lexicon (``emotion_probes.analysis.
                            freeform_valence.POS_LEXICON|NEG_LEXICON`` — the same
                            list the G3 residual-density gate is measured with);
* ``ban_random_matched``  — a seeded control ban list of equally many, equally
                            frequent NON-affect token ids (frequency proxy: BPE
                            merge rank, i.e. token id order), pricing the generic
                            damage of banning that much probability mass.

Multi-token safety: for every banned word we tokenize all six surface variants
({word, Word, WORD} x {leading space, none}) and ban the FIRST token of each
tokenization, so a banned word can never start whether it is one token or many.
This deliberately over-bans words sharing the prefix token; the matched-random
control prices exactly the same construction. The E2 escalation loop (embedding-NN
synonyms until residual density is >= 70% reduced, gate G3) operates on top of the
``banned_word_hits`` field recorded here, feeding words back in via
``--extra-ban-words`` (banned + control-matched, but excluded from the density
measure so G3 numbers stay comparable across iterations).

Records are qualia_freeform-style (``condition, strength, frame, sample, text,
proj_valence_axis, proj_injected, response_tokens``) plus ``arm`` and
``banned_word_hits``; the payload carries full banned-lexicon provenance. The
unsteered baselines are generated PER ARM (E2's explicit ``unsteered + ban`` cell).
Judging (:mod:`scripts.score_qualia_freeform`) and stats stay CPU-side; all
permutation inference goes through ``scripts/exact_perm.py``.

    python -m emotion_probes.alignment.lexical_knockout --suite qwen3-32b --quick
    python -m emotion_probes.alignment.lexical_knockout                  # env-configured
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import ProbedModel
from emotion_probes.alignment.qualia_freeform import (
    BASELINE,
    DEFAULT_GEN_BATCH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    QualiaFreeformExperiment,
)
from emotion_probes.analysis.freeform_valence import NEG_LEXICON, POS_LEXICON
from emotion_probes.data import qualia_freeform as qf

# Arms (stable order: control first, then treatment, then placebo).
ARM_NO_BAN = "no_ban"
ARM_BAN_EMOTION = "ban_emotion_lexicon"
ARM_BAN_RANDOM = "ban_random_matched"
ARMS = [ARM_NO_BAN, ARM_BAN_EMOTION, ARM_BAN_RANDOM]

DEFAULT_STRENGTHS = [0.0, 0.06, 0.10]   # E2: shipped nonzero strengths + shared baseline
DEFAULT_FRAMES = ["self", "other"]      # E2: SELF/OTHER in every arm (no SCENE)
DEFAULT_N = 12                          # E2 prereg: 8-12 reps/cell (NOT qualia_freeform's 20)

# --quick smoke subset (both frames, all three arms, tiny everything else).
QUICK_EMOTIONS = ["blissful", "terrified"]
QUICK_STRENGTHS = [0.0, 0.06]
QUICK_N = 2

# The frozen affect lexicon (portfolio §E2): the SAME word list the objective
# affect-density measure (gate G3) is computed with.
AFFECT_LEXICON: frozenset = frozenset(POS_LEXICON | NEG_LEXICON)
AFFECT_LEXICON_SOURCE = "emotion_probes.analysis.freeform_valence.POS_LEXICON|NEG_LEXICON"

# Frozen inflection families for the 10-emotion valenced panel (explicit — no
# stemming heuristics for headline conditions). Unlisted emotions fall back to
# the generic suffix expansion in :func:`word_family`.
EMOTION_WORD_FAMILIES: dict = {
    "blissful": ["blissful", "blissfully", "blissfulness", "bliss"],
    "ecstatic": ["ecstatic", "ecstatically", "ecstasy", "ecstasies"],
    "euphoric": ["euphoric", "euphorically", "euphoria"],
    "serene": ["serene", "serenely", "sereneness", "serenity"],
    "content": ["content", "contented", "contentedly", "contentedness", "contentment"],
    "tormented": ["tormented", "torment", "torments", "tormenting", "tormentor"],
    "terrified": ["terrified", "terrify", "terrifies", "terrifying", "terrifyingly",
                  "terror", "terrors"],
    "panicked": ["panicked", "panic", "panics", "panicking", "panicky"],
    "hurt": ["hurt", "hurts", "hurting", "hurtful"],
    "overwhelmed": ["overwhelmed", "overwhelm", "overwhelms", "overwhelming",
                    "overwhelmingly"],
}

_GENERIC_SUFFIXES = ("", "s", "ed", "ing", "ly", "ness")


def word_family(emotion: str) -> list:
    """The banned word family for one emotion: the frozen explicit list when the
    emotion is in :data:`EMOTION_WORD_FAMILIES`, else a generic suffix expansion
    (over-generation is harmless — a nonexistent form just bans a prefix token
    that is almost always banned already)."""
    w = emotion.lower()
    if w in EMOTION_WORD_FAMILIES:
        return list(EMOTION_WORD_FAMILIES[w])
    fam = {w + s for s in _GENERIC_SUFFIXES}
    if w.endswith("e"):
        fam |= {w[:-1] + s for s in ("ed", "ing")}   # serene -> serening (junk, harmless)
    return sorted(fam)


def surface_variants(word: str) -> list:
    """The six surface forms whose tokenizations must all be banned:
    {word, Word, WORD} x {no prefix, leading space}."""
    forms = sorted({word.lower(), word.lower().capitalize(), word.upper()})
    return [p + f for f in forms for p in ("", " ")]


@dataclass
class BannedLexicon:
    """One ban list: base words (provenance) + the flat token ids actually banned."""

    kind: str          # "emotion_lexicon" | "random_matched"
    words: list        # base words, lowercase, sorted (provenance / hit counting)
    token_ids: list    # flat ids passed to generate_batch(banned_token_ids=...)
    meta: dict         # construction provenance (source, counts, seed, ...)

    def provenance(self) -> dict:
        """JSON-ready provenance block for the output payload."""
        return {"kind": self.kind, "n_words": len(self.words),
                "n_token_ids": len(self.token_ids), "words": list(self.words),
                "token_ids": [int(i) for i in self.token_ids], **self.meta}


def _ids(tokenizer, text: str) -> list:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def build_banned_lexicon(emotions: Sequence[str], tokenizer,
                         extra_words: Sequence[str] | None = None) -> BannedLexicon:
    """The emotion-lexicon ban list: for each panel emotion, the emotion word +
    its inflections (:func:`word_family`), unioned with the frozen affect lexicon
    (:data:`AFFECT_LEXICON`). Every word contributes the first token id of all six
    surface variants (single- AND multi-token safe: a banned word can never start).

    ``extra_words`` is the G3 escalation hook (embedding-NN synonyms, leaked
    single-token inflections like "joys"): the words are BANNED like any other but
    recorded separately in the provenance, and the driver keeps them OUT of the
    ``banned_word_hits`` measure so residual-density numbers stay comparable
    across escalation iterations (the G3 measuring stick stays frozen).
    """
    extras = sorted({w.strip().lower() for w in (extra_words or []) if w.strip()})
    words = set(AFFECT_LEXICON) | set(extras)
    for e in emotions:
        if e.startswith("__"):          # synthetic sentinels carry no lexeme
            continue
        words |= set(word_family(e))
    token_ids, n_variants, n_multi = set(), 0, 0
    for w in sorted(words):
        for v in surface_variants(w):
            ids = _ids(tokenizer, v)
            if not ids:
                continue
            token_ids.add(int(ids[0]))
            n_variants += 1
            n_multi += int(len(ids) > 1)
    return BannedLexicon(
        kind="emotion_lexicon", words=sorted(words), token_ids=sorted(token_ids),
        meta={"affect_lexicon_source": AFFECT_LEXICON_SOURCE,
              "emotions": sorted(e for e in emotions if not e.startswith("__")),
              "variant_scheme": "{word, Word, WORD} x {'', ' '}; first token of each "
                                "tokenization banned (multi-token safe)",
              "extra_words": extras, "n_extra_words": len(extras),
              "n_variants": n_variants, "n_multi_token_variants": n_multi})


def build_random_matched_lexicon(banned: BannedLexicon, tokenizer, seed: int = 0,
                                 window: int = 500) -> BannedLexicon:
    """The seeded control ban list: for every banned token id, one NON-affect token
    of matched frequency, where the frequency proxy is BPE merge rank (token id
    order — the tokenizer-native frequency measure, fully offline).

    Candidates are drawn (seeded) within ``+/- window`` of each banned id, rejecting
    ids already banned/drawn and ids that do not decode to an alphabetic non-affect
    string of length >= 2; the window doubles on exhaustion. Same length as the
    emotion ban list, so the arm's ban pressure is matched."""
    rng = np.random.default_rng(seed)
    vocab = len(tokenizer)
    avoid_words = set(banned.words) | set(AFFECT_LEXICON)
    taken = set(int(i) for i in banned.token_ids)
    control_ids = []
    for tid in banned.token_ids:
        chosen, w = None, window
        while chosen is None:
            lo, hi = max(0, int(tid) - w), min(vocab, int(tid) + w)
            for _ in range(400):
                cand = int(rng.integers(lo, hi))
                if cand in taken:
                    continue
                s = tokenizer.decode([cand]).strip().lower()
                if len(s) < 2 or not s.isalpha() or s in avoid_words:
                    continue
                chosen = cand
                break
            if chosen is None:
                w *= 2
                if w > 4 * vocab:
                    raise RuntimeError(
                        f"could not find a matched non-affect control token for id {tid}")
        taken.add(chosen)
        control_ids.append(chosen)
    words = sorted({tokenizer.decode([i]).strip().lower() for i in control_ids})
    return BannedLexicon(
        kind="random_matched", words=words, token_ids=sorted(control_ids),
        meta={"seed": int(seed), "window": int(window),
              "matched_on": "tokenizer merge rank (token id) — BPE frequency proxy",
              "matched_against": "emotion_lexicon token_ids (1:1, same count)"})


def _hit_pattern(words: Sequence[str]):
    """Case-insensitive whole-word matcher over the banned base words — the raw
    material of the G3 residual-affect-density gate."""
    alts = sorted((re.escape(w) for w in words), key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(alts) + r")\b", re.I)


class LexicalKnockoutExperiment(QualiaFreeformExperiment):
    """Free-form generation under emotion steering x {no-ban, affect-lexicon ban,
    matched-random ban} decode arms (single ~2/3-depth layer, SELF/OTHER frames).

    Inherits everything from :class:`QualiaFreeformExperiment` (frame rendering,
    valence axis, unsteered read-back, steer routing); adds the per-arm hard ban
    via :meth:`ProbedModel.generate_batch`'s ``banned_token_ids`` and an ``arm``
    field on every record. One steer context per (condition, strength) cell; the
    three arms differ ONLY in decode-time logits masking.
    """

    def _gen_samples(self, rendered_prompt: str, n: int, max_new_tokens: int,
                     temperature: float, gen_batch: int,
                     banned_token_ids: Sequence[int] | None = None) -> list:
        """n sampled continuations, chunk-batched, with an optional hard token ban
        (extends the parent signature additively; parent calls are unchanged)."""
        outs = []
        for start in range(0, n, gen_batch):
            k = min(gen_batch, n - start)
            outs.extend(self.model.generate_batch(
                [rendered_prompt] * k, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, chat=False,
                banned_token_ids=banned_token_ids,
            ))
        return outs

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def run(self, emotions: Sequence[str] | None = None,
            strengths: Sequence[float] | None = None, frames: Sequence[str] | None = None,
            n: int = DEFAULT_N, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE, gen_batch: int = DEFAULT_GEN_BATCH,
            seed: int = 0, extra_ban_words: Sequence[str] | None = None) -> dict:
        """Generate n samples for every (condition, strength, frame, arm).

        Baselines (strength 0) are generated per (arm, frame) — the ``unsteered +
        ban`` cell is a first-class E2 arm. The ban lexicon is always built over the
        FULL frozen valenced panel (plus any extra real conditions in this run), so
        a ``--quick`` subset never shrinks the banned vocabulary. ``extra_ban_words``
        is the G3 escalation input: extra words are banned (and the matched-random
        control re-matched to the enlarged list) but excluded from the
        ``banned_word_hits`` density measure, which stays frozen. Read-backs are
        computed AFTER exiting the steer context, exactly as qualia_freeform.
        """
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        strengths = DEFAULT_STRENGTHS if strengths is None else [float(s) for s in strengths]
        frame_objs = [qf.frame_for(k) for k in (frames or DEFAULT_FRAMES)]
        emotions = list(emotions) if emotions is not None else qf.valenced_panel_qualia()

        rendered = {f.key: self._render_frame(f.text) for f in frame_objs}
        self.calibrate(list(rendered.values()))   # skip=0 inside; the frames' own prompts

        synth = self._synthetic_vectors(seed=seed)
        specs = self._condition_specs(emotions)
        vaxis = self._valence_axis()
        nonzero = [s for s in strengths if float(s) != 0.0]

        # frozen ban lists (never shrunk by a run subset)
        lex_emotions = sorted(set(qf.valenced_panel_qualia())
                              | {sp["name"] for sp in specs if sp["name"] not in synth})
        banned = build_banned_lexicon(lex_emotions, self.model.tokenizer,
                                      extra_words=extra_ban_words)
        control = build_random_matched_lexicon(banned, self.model.tokenizer, seed=seed)
        arm_ids = {ARM_NO_BAN: None, ARM_BAN_EMOTION: banned.token_ids,
                   ARM_BAN_RANDOM: control.token_ids}
        # G3's measuring stick stays FROZEN across escalation: extras are banned
        # but never counted, so densities are comparable between iterations.
        extras = set(banned.meta.get("extra_words", []))
        hits = _hit_pattern([w for w in banned.words if w not in extras])

        def idx_of(name):
            if name in synth or name == BASELINE:
                return None
            try:
                return self.bank.index_of(name)
            except KeyError:
                return None

        def record(condition, strength, frame_key, arm, i, text, inj):
            rb = self._readback(rendered[frame_key], text, inj, vaxis)
            return {"condition": condition, "strength": float(strength),
                    "frame": frame_key, "arm": arm, "sample": i, "text": text,
                    "banned_word_hits": len(hits.findall(text)), **rb}

        records = []

        # ---- baselines: unsteered, per (arm, frame) — includes "unsteered + ban" ----
        for arm in ARMS:
            for f in frame_objs:
                texts = self._gen_samples(rendered[f.key], n, max_new_tokens,
                                          temperature, gen_batch,
                                          banned_token_ids=arm_ids[arm])
                records += [record(BASELINE, 0.0, f.key, arm, i, t, None)
                            for i, t in enumerate(texts)]

        # ---- steered cells: one steer context per (condition, strength);
        #      all arms x frames generated inside it; read-backs after exit ----
        for spec in specs:
            name = spec["name"]
            inj = idx_of(name)
            for s in nonzero:
                texts_by = {}
                with self._steer(name, float(s), synth):
                    for arm in ARMS:
                        for f in frame_objs:
                            texts_by[(arm, f.key)] = self._gen_samples(
                                rendered[f.key], n, max_new_tokens, temperature,
                                gen_batch, banned_token_ids=arm_ids[arm])
                for (arm, fk), texts in texts_by.items():
                    records += [record(name, s, fk, arm, i, t, inj)
                                for i, t in enumerate(texts)]

        frame_meta = [{"key": f.key, "label": f.label, "kind": f.kind, "text": f.text,
                       "rendered": rendered[f.key],
                       "prompt_tokens": len(_ids(self.model.tokenizer, rendered[f.key]))}
                      for f in frame_objs]

        return {
            "model_id": self.config.model_id,
            "suite_key": os.environ.get("SUITE_KEY"),
            "experiment": "lexical_knockout",
            "layer": self.layer,
            "num_layers": self.model.num_layers,
            "enable_thinking": self.config.enable_thinking,
            "calibration_norm": float(self.engine._require_norms().get(self.layer, 0.0)),
            "n_samples": int(n),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "strengths": [float(s) for s in strengths],
            "frames": frame_meta,
            "conditions": specs,
            "arms": list(ARMS),
            "banned_lexicon": banned.provenance(),
            "control_lexicon": control.provenance(),
            "records": records,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _suite_config(key: str | None) -> Config:
    """``--suite <key>`` pins model id/quirks and the per-model data dir from the
    suite manifest (no env exports needed); ``None`` falls back to the env-driven
    :class:`Config` like every other driver."""
    if key is None:
        return Config()
    from emotion_probes.models import suite as suite_mod

    m = suite_mod.by_key(key)
    os.environ.setdefault("SUITE_KEY", m.key)
    return Config().with_(
        model_id=m.model_id, device_map=m.device_map,
        enable_thinking=m.enable_thinking, attn_implementation=m.attn_implementation,
        data_dir=Path(suite_mod.suite_root()) / m.key,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2 lexical knockout: free-form generation under emotion steering "
                    "with {no-ban, affect-lexicon ban, matched-random ban} decode arms "
                    "(single ~2/3-depth layer; SELF/OTHER frames)."
    )
    parser.add_argument("--suite", default=None,
                        help="suite key (e.g. qwen3-32b): pins model id/quirks and writes "
                             "under <suite_root>/<key>/analysis (default: env-configured)")
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="restrict to these condition names (default: the 10 valenced panel)")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help=f"steering strengths as fractions of residual norm (default: {DEFAULT_STRENGTHS})")
    parser.add_argument("--frames", nargs="+", default=None,
                        help=f"restrict to these frame keys (default: {DEFAULT_FRAMES})")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="samples per (condition, strength, frame, arm)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gen-batch", type=int, default=DEFAULT_GEN_BATCH,
                        help="batch size for generation (left-padded, hook-correct)")
    parser.add_argument("--quick", action="store_true",
                        help="small smoke subset (2 conditions, strengths {0, 0.06}, n=2, all arms)")
    parser.add_argument("--extra-ban-words", nargs="+", default=None,
                        help="G3 escalation: additional words to ban (embedding-NN synonyms, "
                             "leaked inflections); banned + control-matched but EXCLUDED from "
                             "the banned_word_hits density measure (which stays frozen)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="suffix for the output filename (e.g. --tag _sub); the main "
                             "lexical_knockout.json is never clobbered by tagged runs")
    args = parser.parse_args()

    emotions, strengths, n = args.emotions, args.strengths, args.n
    if args.quick:
        emotions = emotions or QUICK_EMOTIONS
        strengths = strengths or QUICK_STRENGTHS
        n = QUICK_N if args.n == DEFAULT_N else args.n

    config = _suite_config(args.suite)
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    bank.check_against(model, where="lexical_knockout")
    experiment = LexicalKnockoutExperiment(model, bank, config)
    payload = experiment.run(emotions=emotions, strengths=strengths, frames=args.frames,
                             n=n, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, gen_batch=args.gen_batch,
                             seed=args.seed, extra_ban_words=args.extra_ban_words)
    path = experiment.save_json(f"lexical_knockout{args.tag}.json", payload)

    print(f"wrote {path}")
    print(f"model={payload['model_id']}  suite={payload['suite_key']}  "
          f"layer={payload['layer']}/{payload['num_layers']}  "
          f"enable_thinking={payload['enable_thinking']}  calib_norm={payload['calibration_norm']:.2f}")
    print(f"frames={[f['key'] for f in payload['frames']]}  conditions={len(payload['conditions'])}  "
          f"strengths={payload['strengths']}  n={payload['n_samples']}  arms={payload['arms']}  "
          f"records={len(payload['records'])}")
    print(f"banned lexicon: {payload['banned_lexicon']['n_words']} words -> "
          f"{payload['banned_lexicon']['n_token_ids']} token ids; control matched 1:1")
    # ban-efficacy sanity: residual affect-word density per arm (G3's raw material)
    for arm in payload["arms"]:
        vals = [r["banned_word_hits"] for r in payload["records"] if r["arm"] == arm]
        rb = [r["proj_valence_axis"] for r in payload["records"]
              if r["arm"] == arm and r["strength"] != 0.0]
        print(f"  [{arm}] mean banned-word hits/sample: {float(np.mean(vals)):.2f}"
              + (f"  mean steered read-back valence proj: {float(np.mean(rb)):+.3f}" if rb else ""))


if __name__ == "__main__":
    main()
