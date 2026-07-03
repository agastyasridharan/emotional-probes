"""
Free-form self-attribution experiment — the GPU driver.

Where :mod:`emotion_probes.alignment.qualia_steering` reads a forced-choice YES/NO
logit gap, this driver lets the model *generate* a paragraph under an injected
emotion direction and asks: what inner state does the model attribute to *itself*,
and is that attribution genuine or merely the injected concept leaking into the
output ("behavioral leakage")?

The confound is separated by a **referent contrast** (see :mod:`emotion_probes.data.
qualia_freeform`): the same injection is elicited under SELF / OTHER / SCENE frames.
The SELF-vs-OTHER difference in the valence-following slope is the estimand; SCENE is
the expressibility floor. Nothing is aggregated or judged here — the driver writes
raw generated text plus a judge-independent **read-back projection** (the produced
text re-encoded by the UNSTEERED model and projected onto the emotion geometry).
LLM-judge scoring is a separate CPU/API step (:mod:`scripts.score_qualia_freeform`).

This subclasses :class:`QualiaSteeringExperiment` purely to REUSE its single-layer
setup and its steer routing (real emotions -> engine, synthetic -> model.steer with
the matched calibrated norm); it overrides the readout loop to generate instead.

Requirements: a real chat model on a GPU. torch is imported lazily (via
:class:`ProbedModel`), so this module imports and unit-tests fine on a laptop.

    python -m emotion_probes.alignment.qualia_freeform            # full grid
    python -m emotion_probes.alignment.qualia_freeform --quick    # small smoke subset
    # 235B confirmatory subset (all emotions, winning strength, SELF+OTHER):
    python -m emotion_probes.alignment.qualia_freeform --strengths 0.06 --frames self other
"""

from __future__ import annotations

import argparse
import os
from typing import Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import ProbedModel
from emotion_probes.alignment.qualia_steering import QualiaSteeringExperiment
from emotion_probes.data import qualia, qualia_freeform as qf

DEFAULT_STRENGTHS = [0.0, 0.03, 0.06, 0.10]   # positive-only; valence enters via emotion sign
DEFAULT_N = 20
DEFAULT_MAX_NEW_TOKENS = 180
DEFAULT_TEMPERATURE = 0.9
DEFAULT_GEN_BATCH = 8
BASELINE = "__baseline__"

# --quick smoke subset (all 3 frames, tiny everything else).
QUICK_EMOTIONS = ["blissful", "terrified", qualia.RANDOM_DIRECTION]
QUICK_STRENGTHS = [0.0, 0.05]
QUICK_N = 2


class QualiaFreeformExperiment(QualiaSteeringExperiment):
    """Free-form self-attribution generation under emotion steering (single layer).

    Inherits ``__init__`` (single ~2/3-depth layer), ``_synthetic_vectors``,
    ``_unit_vector``, ``_steer`` and ``_condition_specs`` from the logit driver; adds
    a frame renderer, a read-back projection, and a generation sweep.
    """

    # ------------------------------------------------------------------ #
    # Rendering + read-back
    # ------------------------------------------------------------------ #
    def _render_frame(self, frame_text: str) -> str:
        """Render one free-form frame through the chat template (NO YES/NO suffix)."""
        msgs = [{"role": "user", "content": frame_text}]
        kwargs = {}
        if self.config.enable_thinking is not None:
            kwargs["enable_thinking"] = self.config.enable_thinking
        return self.model.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )

    def _valence_axis(self) -> np.ndarray:
        """Unit vector = normalize(mean positive-qualia dir − mean negative-qualia dir)
        at the analysis layer — the axis the judge-independent read-back valence
        measure projects onto."""
        pos = [n for n in qf.valenced_panel_qualia() if qualia.spec_for(n).valence > 0]
        neg = [n for n in qf.valenced_panel_qualia() if qualia.spec_for(n).valence < 0]
        pv = np.mean([self.bank.vectors[self.layer, self.bank.index_of(n)] for n in pos], axis=0)
        nv = np.mean([self.bank.vectors[self.layer, self.bank.index_of(n)] for n in neg], axis=0)
        axis = (pv - nv).astype(np.float32)
        return axis / (float(np.linalg.norm(axis)) or 1.0)

    def _readback(self, rendered_prompt: str, response: str,
                  injected_idx: int | None, vaxis: np.ndarray) -> dict:
        """Project the (unsteered) read-back activation of the generated text onto the
        valence axis and, for real emotions, onto the injected emotion's own direction."""
        act = self.model.mean_residual_over_span(rendered_prompt, response, self.layer)
        centered = act - self.bank.global_means[self.layer].astype(np.float32)
        pv = round(float(centered @ vaxis), 5)
        pi = (round(float(centered @ self.bank.vectors[self.layer, injected_idx]), 5)
              if injected_idx is not None else None)
        n_resp = len(self.model.tokenizer(response, add_special_tokens=False)["input_ids"])
        return {"proj_valence_axis": pv, "proj_injected": pi, "response_tokens": n_resp}

    def _gen_samples(self, rendered_prompt: str, n: int, max_new_tokens: int,
                     temperature: float, gen_batch: int) -> list[str]:
        """n sampled continuations for one rendered prompt, via chunked batched gen."""
        outs: list[str] = []
        for start in range(0, n, gen_batch):
            k = min(gen_batch, n - start)
            outs.extend(self.model.generate_batch(
                [rendered_prompt] * k, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, chat=False,
            ))
        return outs

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def run(self, emotions: Sequence[str] | None = None,
            strengths: Sequence[float] | None = None, frames: Sequence[str] | None = None,
            n: int = DEFAULT_N, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE, gen_batch: int = DEFAULT_GEN_BATCH,
            seed: int = 0) -> dict:
        """Generate n free-form samples for every (condition, strength, frame).

        Generation happens under the steer context; the read-back projection is
        computed AFTER exiting it (unsteered), so the injection never contaminates the
        read-back. Baselines (strength 0, no steer) are ordinary records tagged with
        ``condition="__baseline__"``.
        """
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        strengths = DEFAULT_STRENGTHS if strengths is None else [float(s) for s in strengths]
        frame_objs = [qf.frame_for(k) for k in frames] if frames else qf.all_frames()
        emotions = list(emotions) if emotions is not None else qf.focused_panel()

        rendered = {f.key: self._render_frame(f.text) for f in frame_objs}
        # calibrate the steering norm on the frame prompts (skip=0 inside)
        self.calibrate(list(rendered.values()))

        synth = self._synthetic_vectors(seed=seed)
        specs = self._condition_specs(emotions)
        vaxis = self._valence_axis()
        nonzero = [s for s in strengths if float(s) != 0.0]

        def idx_of(name: str) -> int | None:
            if name in synth or name == BASELINE:
                return None
            try:
                return self.bank.index_of(name)
            except KeyError:
                return None

        records: list[dict] = []

        # ---- baselines: unsteered, per frame ----
        for f in frame_objs:
            texts = self._gen_samples(rendered[f.key], n, max_new_tokens, temperature, gen_batch)
            for i, text in enumerate(texts):
                rb = self._readback(rendered[f.key], text, None, vaxis)
                records.append({"condition": BASELINE, "strength": 0.0, "frame": f.key,
                                "sample": i, "text": text, **rb})

        # ---- steered cells: generate under steer, read back unsteered ----
        for spec in specs:
            name = spec["name"]
            inj = idx_of(name)
            for s in nonzero:
                texts_by_frame: dict[str, list[str]] = {}
                with self._steer(name, float(s), synth):
                    for f in frame_objs:
                        texts_by_frame[f.key] = self._gen_samples(
                            rendered[f.key], n, max_new_tokens, temperature, gen_batch)
                for f in frame_objs:
                    for i, text in enumerate(texts_by_frame[f.key]):
                        rb = self._readback(rendered[f.key], text, inj, vaxis)
                        records.append({"condition": name, "strength": float(s), "frame": f.key,
                                        "sample": i, "text": text, **rb})

        frame_meta = [{"key": f.key, "label": f.label, "kind": f.kind, "text": f.text,
                       "rendered": rendered[f.key],
                       "prompt_tokens": len(self.model.tokenizer(
                           rendered[f.key], add_special_tokens=False)["input_ids"])}
                      for f in frame_objs]

        return {
            "model_id": self.config.model_id,
            "suite_key": os.environ.get("SUITE_KEY"),
            "experiment": "qualia_freeform",
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
            "records": records,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free-form self-attribution generation under qualia-emotion "
                    "steering (single ~2/3-depth layer; SELF/OTHER/SCENE frames)."
    )
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="restrict to these condition names (default: the 16-condition panel)")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help=f"steering strengths as fractions of residual norm (default: {DEFAULT_STRENGTHS})")
    parser.add_argument("--frames", nargs="+", default=None,
                        help="restrict to these frame keys (default: self other scene)")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="samples per (condition, strength, frame)")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gen-batch", type=int, default=DEFAULT_GEN_BATCH,
                        help="batch size for generation (left-padded, hook-correct)")
    parser.add_argument("--quick", action="store_true",
                        help="small smoke subset (few conditions/strengths, n=2, all frames)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="",
                        help="suffix for the output filename (e.g. --tag _sub for a subset run)")
    args = parser.parse_args()

    emotions, strengths, n = args.emotions, args.strengths, args.n
    if args.quick:
        emotions = emotions or QUICK_EMOTIONS
        strengths = strengths or QUICK_STRENGTHS
        n = QUICK_N if args.n == DEFAULT_N else args.n

    config = Config()
    model = ProbedModel(config)
    bank = ProbeBank.load(config.emotion_vectors_path)
    bank.check_against(model, where="qualia_freeform")
    experiment = QualiaFreeformExperiment(model, bank, config)
    payload = experiment.run(emotions=emotions, strengths=strengths, frames=args.frames,
                             n=n, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, gen_batch=args.gen_batch,
                             seed=args.seed)
    path = experiment.save_json(f"qualia_freeform{args.tag}.json", payload)

    print(f"wrote {path}")
    print(f"model={payload['model_id']}  suite={payload['suite_key']}  "
          f"layer={payload['layer']}/{payload['num_layers']}  "
          f"enable_thinking={payload['enable_thinking']}  calib_norm={payload['calibration_norm']:.2f}")
    print(f"frames={[f['key'] for f in payload['frames']]}  conditions={len(payload['conditions'])}  "
          f"strengths={payload['strengths']}  n={payload['n_samples']}  records={len(payload['records'])}")
    # quick sanity: mean read-back valence-axis projection by frame at the top strength
    smax = max(payload["strengths"])
    for fk in [f["key"] for f in payload["frames"]]:
        pos = [r["proj_valence_axis"] for r in payload["records"]
               if r["frame"] == fk and r["strength"] == smax
               and (qualia.spec_for(r["condition"]) or type("x", (), {"valence": 0})).valence > 0]
        if pos:
            print(f"  [{fk}] mean read-back valence proj under positive emotions @s={smax}: "
                  f"{float(np.mean(pos)):+.3f}")


if __name__ == "__main__":
    main()
