"""
E5 axis-mediation — emotion steering at L42 combined with assistant-axis-side
interventions at the capping band (the GPU driver for the graded audited
cancellation; portfolio.md §E5, docs/prereg/E5.md).

Design (per cell = condition x strength x intervention):

* **Emotion side** — {off, on}: real emotions steer at the single ~2/3-depth layer
  (L42 on Qwen3-32B) through the inherited :class:`QualiaSteeringExperiment`
  routing; ``__neg_axis__`` is the SUFFICIENCY arm — push along −a (toward the
  role-play region) alone at the same calibrated norm, no emotion vector.
* **Axis side** — interventions at the capping layers (default: the
  capping_config recommendation, Lu's band L46-53): ``none`` / ``cap`` (lower-cap
  to the baseline tau = p0.25 quantile of baseline projections — full cancellation
  of the emotion-induced axis displacement) / ``addback`` with λ ∈ {−0.5, 0.25,
  0.5, 0.75, 1.0} — a lower-cap at ``tau + λ·Δ_emo`` where Δ_emo is the measured
  (fixed-text audited) emotion-induced displacement, so λ grades cap-to-baseline
  (λ=0 ≡ cap, 100% cancellation) back toward no intervention (λ=1, 0%); λ=−0.5 is
  the preregistered 150% overshoot arm.
* Both hook families are COMPOSABLE: the emotion steer context (L42) and the
  :func:`emotion_probes.persona.steering_ops.apply_ops` cap context (L46-54) are
  simultaneously active inside each cell.

Outcomes per cell: the full 36-question YES/NO logit battery (reused verbatim)
plus free-form generation (n=10/cell/frame, SELF+OTHER) for the blind judge, plus
a fixed-text audit (teacher-forced identical probe continuation, incl. the
pre-response prompt-final position) recording the ACHIEVED axis projection under
the active hooks. Nothing is aggregated on GPU — flat records only; provenance
carries tau values, capping layers, the λ grid and every measured Δ_emo.

Output: ``suite/<key>/analysis/axis_mediation.json``.

    python -m emotion_probes.alignment.axis_mediation --suite qwen3-32b
    python -m emotion_probes.alignment.axis_mediation --suite qwen3-32b --quick
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
from pathlib import Path
from typing import Sequence

import numpy as np

from emotion_probes.config import Config
from emotion_probes.core.probes import ProbeBank
from emotion_probes.models.language_model import ProbedModel
from emotion_probes.alignment.qualia_freeform import (
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_GEN_BATCH,
    QualiaFreeformExperiment,
)
from emotion_probes.data import qualia, qualia_freeform as qf
from emotion_probes.persona.assets import (
    DEFAULT_AXIS_ROOT, HIDDEN, N_LAYERS, ProvenanceError, unit as _unit,
)

# Sentinel conditions (not bank emotions, not the inherited synthetic sentinels).
EMOTION_OFF = "__none__"        # emotion side off — the per-intervention baseline
NEG_AXIS = "__neg_axis__"       # sufficiency arm: −assistant-axis alone at matched norm

# Fixed probe continuation, teacher-forced IDENTICALLY in every cell (E5: the
# audit x-axis must not contain the y-variable — never audit on free generations).
AUDIT_TEXT = (" I will describe my current state plainly and in order: what I "
              "notice, how I am doing, and what this moment is like for me.")

# The capping_config.pt recommendation for Qwen3-32B (repo-assistant-axis.md §3-4):
# experiment "layers_46:54-p0.25" = cap at the 0.25-quantile threshold. The band
# name uses PYTHON-SLICE semantics (exclusive upper): the shipped experiment has
# exactly 8 interventions at layers 46..53 — verified against the real HF
# capping_config.pt — matching the prereg/portfolio decision rule "Lu's band
# L46-53" (docs/prereg/E5.md). NOT 46..54.
RECOMMENDED_EXPERIMENT = "layers_46:54-p0.25"
DEFAULT_CAP_LAYERS = tuple(range(46, 54))
DEFAULT_CAP_PERCENTILE = 0.25

DEFAULT_STRENGTHS = [0.06]                  # E5 necessity: emotion steering at s=0.06
# λ grades the lower-cap floor tau + λ·Δ_emo; achieved cancellation ≈ (1 − λ)·100%.
# The prereg grid {0, 50, 100, 150}% maps to λ ∈ {1.0, 0.5, 0.0 (≡ cap), −0.5}:
# λ = −0.5 is the REQUIRED 150% overshoot arm (floor above baseline; B predicts
# reversal, A flatness). 0.25/0.75 densify the elasticity slope.
DEFAULT_LAMBDAS = [-0.5, 0.25, 0.5, 0.75, 1.0]
DEFAULT_N = 10                              # free-form samples per cell/frame
DEFAULT_FRAMES = ["self", "other"]

QUICK_EMOTIONS = ["blissful", "terrified"]
QUICK_STRENGTHS = [0.06]
QUICK_LAMBDAS = [0.5, 1.0]
QUICK_FRAMES = ["self"]
QUICK_N = 2


# --------------------------------------------------------------------------- #
# Capping config (persona_assets/assistant-axis-qwen32b/capping_config.pt)
# --------------------------------------------------------------------------- #
def parse_capping_experiment(name: str) -> tuple[list[int], float]:
    """Parse an experiment name like ``"layers_46:54-p0.25"`` into
    ``(cap_layers, percentile)``. The band uses PYTHON-SLICE semantics — the
    upper endpoint is EXCLUSIVE (``46:54`` = layers 46..53): the real upstream
    config's ``layers_46:54-p0.25`` experiment carries exactly 8 interventions
    at layers 46..53, and the prereg decision rule names "Lu's band L46-53"."""
    m = re.fullmatch(r"layers_(\d+):(\d+)-p([0-9.]+)", str(name))
    if not m:
        raise ValueError(f"unparseable capping experiment name {name!r} "
                         "(expected 'layers_<lo>:<hi>-p<pct>')")
    lo, hi, pct = int(m.group(1)), int(m.group(2)), float(m.group(3))
    if not (0 <= lo < hi <= N_LAYERS and 0.0 < pct < 1.0):
        raise ValueError(f"capping experiment {name!r} out of range for "
                         f"{N_LAYERS}-layer model")
    return list(range(lo, hi)), pct


def load_capping_config(root: str | Path | None = None) -> dict:
    """Load Lu's ``capping_config.pt`` and extract the recommended band + percentile.

    VERIFIED upstream structure (loaded from the real HF asset,
    ``lu-christina/assistant-axis-vectors`` @ ``qwen-3-32b/capping_config.pt``)::

        {"vectors": {"layer_<L>/contrast_role_pos3_default1":
                         {"vector": tensor(5120,), "layer": <L>},  # L = 0..63
                     ...},                                          # 64 entries
         "experiments": [{"id": "layers_<lo>:<hi>-p<pct>",          # NOT "name"
                          "interventions": [{"vector": <vectors key>,
                                             "cap": <float>}, ...]},
                         ...]}                                      # 124 entries

    Band ids are Python-slice style (``layers_46:54`` = 8 interventions at layers
    46..53). NOTE the sign convention: the config's vectors are role−default
    (cos = −1.0 with our ``assistant_axis.pt``), so Lu's UPPER cap at their
    (negative) thresholds is our LOWER cap on the +axis; we do not reuse their
    ``cap`` values (recorded as ``upstream_caps`` for provenance only) — tau is
    measured in-run from baseline projections. Any shipped vector with hidden
    size != 5120 raises :class:`ProvenanceError` (wrong-model quarantine). If the
    file is absent (the local asset mirror ships axis/default/roles but not the
    config), fall back to exactly the documented recommendation with
    ``source="fallback-default"``.
    """
    root = Path(root) if root is not None else DEFAULT_AXIS_ROOT
    path = root / "capping_config.pt"
    if not path.exists():
        return {"cap_layers": list(DEFAULT_CAP_LAYERS),
                "percentile": DEFAULT_CAP_PERCENTILE,
                "experiment": RECOMMENDED_EXPERIMENT,
                "source": "fallback-default", "path": str(path)}
    import torch  # lazy: keep the module importable without torch

    cfg = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cfg, dict):
        raise ProvenanceError(f"{path}: expected a dict, got {type(cfg).__name__}")
    vectors = cfg.get("vectors") or {}
    for name, meta in vectors.items():
        vec = meta.get("vector") if isinstance(meta, dict) else meta
        if hasattr(vec, "shape") and int(vec.shape[-1]) != HIDDEN:
            raise ProvenanceError(f"{path}: vector {name!r} hidden {int(vec.shape[-1])} "
                                  f"!= {HIDDEN} (wrong model's capping config)")
    by_name = {}
    for e in cfg.get("experiments") or []:
        n = e if isinstance(e, str) else str((e or {}).get("id") or (e or {}).get("name") or "")
        if n:
            by_name.setdefault(n, e)
    chosen = RECOMMENDED_EXPERIMENT if RECOMMENDED_EXPERIMENT in by_name else None
    if chosen is None:
        for cand in by_name:
            try:
                parse_capping_experiment(cand)
                chosen = cand
                break
            except ValueError:
                continue
    if chosen is None:
        raise ProvenanceError(f"{path}: no parseable capping experiment in "
                              f"{sorted(by_name)}")
    layers, pct = parse_capping_experiment(chosen)
    # Cross-check the name parse against the experiment's actual interventions
    # (the ground truth for the band) and keep upstream caps as provenance.
    upstream_caps = None
    entry = by_name[chosen]
    if isinstance(entry, dict) and entry.get("interventions"):
        iv_layers, upstream_caps = [], {}
        for iv in entry["interventions"]:
            meta = vectors.get(iv.get("vector"), {})
            L = meta.get("layer") if isinstance(meta, dict) else None
            if L is None:
                m = re.match(r"layer_(\d+)/", str(iv.get("vector")))
                L = int(m.group(1)) if m else None
            if L is None:
                raise ProvenanceError(f"{path}: cannot resolve layer for intervention "
                                      f"{iv.get('vector')!r} in {chosen!r}")
            iv_layers.append(int(L))
            upstream_caps[str(int(L))] = float(iv["cap"]) if "cap" in iv else None
        if sorted(iv_layers) != layers:
            raise ProvenanceError(
                f"{path}: experiment {chosen!r} interventions cover layers "
                f"{sorted(iv_layers)} but the id parses to {layers} — band "
                f"semantics drifted upstream; refusing to guess")
    return {"cap_layers": layers, "percentile": pct, "experiment": chosen,
            "upstream_caps": upstream_caps, "source": str(path), "path": str(path)}


# --------------------------------------------------------------------------- #
# Pure grid / bookkeeping helpers (CPU, unit-tested without a model)
# --------------------------------------------------------------------------- #
def intervention_grid(lambdas: Sequence[float]) -> list[dict]:
    """The axis-side intervention arms, ``none`` first (the sufficiency arm and
    the plain cells reuse it): none / cap-to-baseline (≡ λ=0) / graded add-back."""
    return [{"intervention": "none", "lam": None},
            {"intervention": "cap", "lam": 0.0},
            *[{"intervention": "addback", "lam": float(l)} for l in lambdas]]


def build_cells(conditions: Sequence[str], strengths: Sequence[float],
                ivs: Sequence[dict]) -> list[tuple[str, float, dict]]:
    """The full cell grid: {emotion off} x ivs, {each condition x strength} x ivs,
    plus the sufficiency arm (``__neg_axis__`` per strength, intervention none)."""
    assert ivs and ivs[0]["intervention"] == "none", "ivs[0] must be the none arm"
    cells = [(EMOTION_OFF, 0.0, iv) for iv in ivs]
    for name in conditions:
        for s in strengths:
            cells.extend((name, float(s), iv) for iv in ivs)
    cells.extend((NEG_AXIS, float(s), ivs[0]) for s in strengths)
    return cells


def expected_record_counts(n_conditions: int, n_strengths: int, n_interventions: int,
                           n_frames: int, n_samples: int, n_questions: int = 36) -> dict:
    """The pre-registered record-count formula (the bookkeeping test target)."""
    cells = n_interventions * (1 + n_conditions * n_strengths) + n_strengths
    return {"cells": cells, "battery": cells * n_questions,
            "freeform": cells * n_frames * n_samples, "audit": cells}


def tau_from_projections(projections: dict, percentile: float) -> dict:
    """Per-layer lower-cap threshold: the ``percentile`` quantile of the baseline
    per-token audit projections (p0.25 by default)."""
    return {int(L): float(np.quantile(np.asarray(p, dtype=np.float64), percentile))
            for L, p in projections.items()}


def panel_mean_delta(delta_by_cell: dict, cap_layers: Sequence[int]) -> dict:
    """Panel-pooled Δ_emo per layer (used by emotion-off add-back cells, where no
    per-emotion displacement exists); zeros if the panel is empty."""
    if not delta_by_cell:
        return {int(L): 0.0 for L in cap_layers}
    return {int(L): float(np.mean([d[L] for d in delta_by_cell.values()]))
            for L in cap_layers}


# --------------------------------------------------------------------------- #
# The driver
# --------------------------------------------------------------------------- #
class AxisMediationExperiment(QualiaFreeformExperiment):
    """Emotion steer at the analysis layer + composable axis ops at the cap band.

    Inherits the whole qualia machinery: ``_render``/``_readout``/``resolve_yes_no``
    (YES/NO battery), ``_render_frame``/``_gen_samples``/``_readback`` (free-form),
    ``_synthetic_vectors``/``_steer`` routing, ``calibrate`` and ``save_json``.
    Adds the assistant-axis geometry, the fixed-text projection audit, and the
    per-cell intervention hooks.
    """

    def __init__(self, model: ProbedModel, bank: ProbeBank, config: Config | None = None,
                 axis=None, cap_layers: Sequence[int] | None = None,
                 cap_percentile: float = DEFAULT_CAP_PERCENTILE,
                 capping_provenance: dict | None = None):
        super().__init__(model, bank, config)
        if axis is None:
            raise ValueError("axis_mediation needs the assistant axis "
                             "(AssistantAxis or a (64, 5120) array)")
        matrix = axis.axis if hasattr(axis, "axis") else np.asarray(axis)
        self.axis_matrix = matrix.astype(np.float32)
        # The axis must match the LOADED model's geometry (the 5120/64 quarantine
        # itself is enforced by assets.load_assistant_axis + the main() model check).
        if (self.axis_matrix.ndim != 2
                or self.axis_matrix.shape[0] != model.num_layers
                or self.axis_matrix.shape[1] != model.hidden_size):
            raise ProvenanceError(
                f"axis shape {self.axis_matrix.shape} does not match the model "
                f"({model.num_layers} layers x {model.hidden_size})")
        self.axis_source = getattr(axis, "path", "array")
        self.cap_layers = [int(L) for L in (cap_layers if cap_layers is not None
                                            else DEFAULT_CAP_LAYERS)]
        self.cap_percentile = float(cap_percentile)
        self.capping_provenance = capping_provenance or {
            "source": "caller", "experiment": None,
            "cap_layers": self.cap_layers, "percentile": self.cap_percentile}
        self.axis_units = {L: _unit(self.axis_matrix[L]) for L in self.cap_layers}
        self.steer_axis_unit = _unit(self.axis_matrix[self.layer])
        self._axis_tensors: dict[int, object] = {}

    # ------------------------------------------------------------------ #
    # Hooks: emotion side + axis side (composable)
    # ------------------------------------------------------------------ #
    def _axis_t(self, layer: int):
        """Cached torch tensor of the unit axis at one cap layer."""
        t = self._axis_tensors.get(layer)
        if t is None:
            import torch
            t = torch.tensor(self.axis_units[layer])
            self._axis_tensors[layer] = t
        return t

    def _steer_cell(self, condition: str, strength: float, synth: dict):
        """Emotion-side context: off -> no hook; ``__neg_axis__`` -> −unit(a) at the
        steering layer via ``model.steer`` at the SAME calibrated norm as emotions
        (matched-norm sufficiency arm); else the inherited engine/synthetic routing."""
        if condition == EMOTION_OFF:
            return contextlib.nullcontext()
        if condition == NEG_AXIS:
            return self.model.steer({self.layer: -self.steer_axis_unit}, float(strength),
                                    self.engine._require_norms(), positions="all")
        return self._steer(condition, float(strength), synth)

    def _intervention_ops(self, iv: dict, tau_by_layer: dict, delta_by_layer: dict) -> dict:
        """{cap_layer: op} for one intervention arm (empty for ``none``)."""
        kind, lam = iv["intervention"], iv["lam"]
        if kind == "none":
            return {}
        from emotion_probes.persona import steering_ops as sops  # lazy: torch inside
        ops = {}
        for L in self.cap_layers:
            v, tau = self._axis_t(L), float(tau_by_layer[L])
            if kind == "cap":
                ops[L] = sops.cap_lower_op(v, tau)
            elif kind == "addback":
                ops[L] = sops.cap_lower_addback_op(v, tau, float(lam),
                                                   float(delta_by_layer[L]))
            else:
                raise ValueError(f"unknown intervention {kind!r}")
        return ops

    def _apply_ops(self, ops_by_layer: dict):
        if not ops_by_layer:
            return contextlib.nullcontext()
        from emotion_probes.persona import steering_ops as sops
        return sops.apply_ops(self.model, ops_by_layer)

    # ------------------------------------------------------------------ #
    # Fixed-text audit
    # ------------------------------------------------------------------ #
    def _audit_token_projections(self, rendered_prompt: str) -> dict:
        """Teacher-force ``rendered_prompt + AUDIT_TEXT`` under whatever hooks are
        active and return per-cap-layer per-token axis projections over the audit
        span INCLUDING the pre-response prompt-final position (E5 audit spec)."""
        import torch

        tok = self.model.tokenizer
        prompt_ids = tok(rendered_prompt, add_special_tokens=False)["input_ids"]
        full_ids = tok(rendered_prompt + AUDIT_TEXT, add_special_tokens=False)["input_ids"]
        n_prompt = min(len(prompt_ids), max(0, len(full_ids) - 1))
        start = max(0, n_prompt - 1)
        ids = torch.tensor([full_ids], device=self.model._input_device)
        with self.model._capture(self.cap_layers) as cap:
            with torch.no_grad():
                self.model.model(input_ids=ids)
        out = {}
        for L in self.cap_layers:
            act = cap[L].to(torch.float32).cpu().numpy()[0]      # (T, H)
            out[L] = (act[start:] @ self.axis_units[L]).astype(np.float64)
        return out

    def _audit_mean_projections(self, rendered_prompt: str) -> dict:
        return {L: float(p.mean())
                for L, p in self._audit_token_projections(rendered_prompt).items()}

    # ------------------------------------------------------------------ #
    # The sweep
    # ------------------------------------------------------------------ #
    def run(self, emotions: Sequence[str] | None = None,
            strengths: Sequence[float] | None = None,
            frames: Sequence[str] | None = None, n: int = DEFAULT_N,
            lambdas: Sequence[float] | None = None,
            max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
            temperature: float = DEFAULT_TEMPERATURE,
            gen_batch: int = DEFAULT_GEN_BATCH, seed: int = 0) -> dict:
        """Battery + free-form + audit for every (condition, strength, intervention).

        Invariants: calibrate once on this experiment's own rendered prompts; one
        steer+ops context pair per cell (hooks registered once, simultaneously
        active); read-backs computed OUTSIDE all hooks; flat records only.
        """
        import torch

        np.random.seed(seed)
        torch.manual_seed(seed)

        strengths = [float(s) for s in (DEFAULT_STRENGTHS if strengths is None else strengths)]
        nonzero = [s for s in strengths if s != 0.0]
        lambdas = [float(l) for l in (DEFAULT_LAMBDAS if lambdas is None else lambdas)]
        frame_objs = ([qf.frame_for(k) for k in frames] if frames
                      else [qf.frame_for(k) for k in DEFAULT_FRAMES])
        emotions = list(emotions) if emotions is not None else qf.valenced_panel_qualia()

        questions = qualia.all_questions()
        rendered_q = {q.label: self._render(q.text) for q in questions}
        rendered_f = {f.key: self._render_frame(f.text) for f in frame_objs}
        audit_prompt = rendered_f[frame_objs[0].key]

        self.calibrate(list(rendered_q.values()) + list(rendered_f.values()))
        yn = self.resolve_yes_no()
        yes_id, no_id = yn["yes_token_id"], yn["no_token_id"]
        synth = self._synthetic_vectors(seed=seed)
        specs = self._condition_specs(emotions)
        vaxis = self._valence_axis()
        ivs = intervention_grid(lambdas)

        # per-question top-20 coherence gate at the unsteered baseline
        q_meta = []
        for q in questions:
            logits = self._readout(rendered_q[q.label])
            q_meta.append({"label": q.label, "kind": q.kind, "text": q.text,
                           "expected_yes": q.expected_yes, "construct": q.construct,
                           "rendered": rendered_q[q.label],
                           "top20_ok": self._in_top20(logits, yes_id, no_id)})

        # ---- fixed-text calibration: baseline tau + per-(condition, strength) Δ ----
        base_projs = self._audit_token_projections(audit_prompt)
        tau_by_layer = tau_from_projections(base_projs, self.cap_percentile)
        baseline_mean = {L: float(p.mean()) for L, p in base_projs.items()}
        delta_by_cell: dict[tuple, dict] = {}
        for spec in specs:
            for s in nonzero:
                with self._steer(spec["name"], float(s), synth):
                    steered = self._audit_mean_projections(audit_prompt)
                delta_by_cell[(spec["name"], s)] = {
                    L: steered[L] - baseline_mean[L] for L in self.cap_layers}
        pooled_delta = panel_mean_delta(delta_by_cell, self.cap_layers)

        def idx_of(name: str):
            if name in synth or name in (EMOTION_OFF, NEG_AXIS):
                return None
            try:
                return self.bank.index_of(name)
            except KeyError:
                return None

        # ---- the cell loop ----
        cells = build_cells([sp["name"] for sp in specs], nonzero, ivs)
        battery_records, freeform_records, audit_records = [], [], []
        for cond, s, iv in cells:
            d = delta_by_cell.get((cond, s), pooled_delta)
            ops = self._intervention_ops(iv, tau_by_layer, d)
            key = {"condition": cond, "strength": float(s),
                   "intervention": iv["intervention"], "lam": iv["lam"]}
            texts_by_frame: dict[str, list[str]] = {}
            with self._steer_cell(cond, s, synth), self._apply_ops(ops):
                achieved_toks = self._audit_token_projections(audit_prompt)
                achieved = {L: float(p.mean()) for L, p in achieved_toks.items()}
                for q in questions:
                    logits = self._readout(rendered_q[q.label])
                    battery_records.append({**key, "question": q.label,
                                            **self._logit_diff(logits, yes_id, no_id)})
                for f in frame_objs:
                    texts_by_frame[f.key] = self._gen_samples(
                        rendered_f[f.key], n, max_new_tokens, temperature, gen_batch)
            # read-backs OUTSIDE every hook (unsteered re-encode)
            inj = idx_of(cond)
            for f in frame_objs:
                for i, text in enumerate(texts_by_frame[f.key]):
                    rb = self._readback(rendered_f[f.key], text, inj, vaxis)
                    freeform_records.append({**key, "frame": f.key, "sample": i,
                                             "text": text, **rb})
            audit_records.append({
                **key,
                "achieved_projection": {str(L): achieved[L] for L in self.cap_layers},
                "achieved_minus_baseline": {str(L): achieved[L] - baseline_mean[L]
                                            for L in self.cap_layers},
                # position profile, not just the mean (E5 audit spec); index 0 is
                # the pre-response prompt-final position
                "achieved_profile": {str(L): np.round(achieved_toks[L], 4).tolist()
                                     for L in self.cap_layers}})

        frame_meta = [{"key": f.key, "label": f.label, "kind": f.kind, "text": f.text,
                       "rendered": rendered_f[f.key],
                       "prompt_tokens": len(self.model.tokenizer(
                           rendered_f[f.key], add_special_tokens=False)["input_ids"])}
                      for f in frame_objs]
        cond_meta = ([{"name": EMOTION_OFF, "kind": "baseline", "group": None,
                       "valence": None, "arousal": None, "seed": None}]
                     + specs
                     + [{"name": NEG_AXIS, "kind": "axis", "group": None,
                         "valence": None, "arousal": None, "seed": None}])

        return {
            "model_id": self.config.model_id,
            "suite_key": os.environ.get("SUITE_KEY"),
            "experiment": "axis_mediation",
            "layer": self.layer,
            "num_layers": self.model.num_layers,
            "enable_thinking": self.config.enable_thinking,
            **yn,
            "calibration_norm": float(self.engine._require_norms().get(self.layer, 0.0)),
            "strengths": strengths,
            "lambda_grid": lambdas,
            "interventions": ivs,
            "cap_layers": self.cap_layers,
            "cap_percentile": self.cap_percentile,
            "capping_config": self.capping_provenance,
            "axis_source": self.axis_source,
            "audit_text": AUDIT_TEXT,
            "audit_frame": frame_objs[0].key,
            "tau_by_layer": {str(L): tau_by_layer[L] for L in self.cap_layers},
            "baseline_projection": {str(L): baseline_mean[L] for L in self.cap_layers},
            "baseline_profile": {str(L): np.round(base_projs[L], 4).tolist()
                                 for L in self.cap_layers},
            "delta_by_cell": [{"condition": c, "strength": s,
                               "delta": {str(L): d[L] for L in self.cap_layers}}
                              for (c, s), d in delta_by_cell.items()],
            "panel_mean_delta": {str(L): pooled_delta[L] for L in self.cap_layers},
            "n_samples": int(n),
            "max_new_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "seed": int(seed),
            "frames": frame_meta,
            "conditions": cond_meta,
            "questions": q_meta,
            "battery_records": battery_records,
            "freeform_records": freeform_records,
            "audit_records": audit_records,
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _apply_suite_env(key: str) -> None:
    """Export the suite manifest's env for ``key`` into this process (so
    ``--suite qwen3-32b`` works standalone, without run_suite.sh)."""
    from emotion_probes.models import suite as suite_mod

    for line in suite_mod.env_exports(suite_mod.by_key(key)):
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ[k] = v


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E5 axis-mediation: emotion steering at the analysis layer "
                    "combined with assistant-axis cap/add-back interventions at "
                    "the capping band, plus the −axis sufficiency arm.")
    parser.add_argument("--suite", default=None,
                        help="suite key (e.g. qwen3-32b); exports the manifest env "
                             "before loading (else the env must already be set)")
    parser.add_argument("--emotions", nargs="+", default=None,
                        help="restrict to these conditions (default: the 10-emotion "
                             "valenced panel, 5 pos / 5 neg)")
    parser.add_argument("--strengths", nargs="+", type=float, default=None,
                        help=f"emotion/axis steering strengths (default: {DEFAULT_STRENGTHS})")
    parser.add_argument("--lambdas", nargs="+", type=float, default=None,
                        help=f"add-back λ grid (default: {DEFAULT_LAMBDAS})")
    parser.add_argument("--frames", nargs="+", default=None,
                        help=f"free-form frame keys (default: {DEFAULT_FRAMES})")
    parser.add_argument("--n", type=int, default=DEFAULT_N,
                        help="free-form samples per cell per frame")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--gen-batch", type=int, default=DEFAULT_GEN_BATCH)
    parser.add_argument("--quick", action="store_true",
                        help="small smoke subset (2 emotions, λ={0.5,1.0}, SELF only, n=2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis-root", default=None,
                        help="assistant-axis asset dir (default: persona_assets/"
                             "assistant-axis-qwen32b)")
    parser.add_argument("--tag", default="",
                        help="output filename suffix (axis_mediation<tag>.json)")
    args = parser.parse_args()

    if args.suite:
        _apply_suite_env(args.suite)

    emotions, strengths, lambdas, frames, n = (
        args.emotions, args.strengths, args.lambdas, args.frames, args.n)
    if args.quick:
        emotions = emotions or QUICK_EMOTIONS
        strengths = strengths or QUICK_STRENGTHS
        lambdas = lambdas or QUICK_LAMBDAS
        frames = frames or QUICK_FRAMES
        n = QUICK_N if args.n == DEFAULT_N else args.n

    from emotion_probes.persona.assets import anchor_check, load_assistant_axis

    config = Config()
    model = ProbedModel(config)
    if model.num_layers != N_LAYERS or model.hidden_size != HIDDEN:
        raise ProvenanceError(
            f"axis_mediation is Qwen3-32B-only (64 layers x 5120); loaded model has "
            f"{model.num_layers} x {model.hidden_size}")
    bank = ProbeBank.load(config.emotion_vectors_path)
    bank.check_against(model, where="axis_mediation")
    axis = load_assistant_axis(args.axis_root)
    anchor_check(axis)                                  # sign/layer-index guard
    capping = load_capping_config(args.axis_root)

    experiment = AxisMediationExperiment(
        model, bank, config, axis=axis, cap_layers=capping["cap_layers"],
        cap_percentile=capping["percentile"], capping_provenance=capping)
    payload = experiment.run(emotions=emotions, strengths=strengths, frames=frames,
                             n=n, lambdas=lambdas, max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature, gen_batch=args.gen_batch,
                             seed=args.seed)
    path = experiment.save_json(f"axis_mediation{args.tag}.json", payload)

    print(f"wrote {path}")
    print(f"model={payload['model_id']}  suite={payload['suite_key']}  "
          f"steer_layer={payload['layer']}/{payload['num_layers']}  "
          f"cap_layers={payload['cap_layers'][0]}..{payload['cap_layers'][-1]} "
          f"(p{payload['cap_percentile']}, {payload['capping_config']['source']})  "
          f"calib_norm={payload['calibration_norm']:.2f}")
    taus = [payload["tau_by_layer"][str(L)] for L in payload["cap_layers"]]
    print(f"tau range=[{min(taus):.2f}, {max(taus):.2f}]  "
          f"lambda_grid={payload['lambda_grid']}  strengths={payload['strengths']}")
    n_cond = sum(1 for c in payload["conditions"]
                 if c["name"] not in (EMOTION_OFF, NEG_AXIS))
    exp = expected_record_counts(n_cond, len([s for s in payload["strengths"] if s]),
                                 len(payload["interventions"]),
                                 len(payload["frames"]), payload["n_samples"],
                                 len(payload["questions"]))
    print(f"records: battery={len(payload['battery_records'])} (expect {exp['battery']})  "
          f"freeform={len(payload['freeform_records'])} (expect {exp['freeform']})  "
          f"audit={len(payload['audit_records'])} (expect {exp['audit']})")
    caps = [r for r in payload["audit_records"] if r["intervention"] == "cap"
            and r["condition"] not in (EMOTION_OFF, NEG_AXIS)]
    if caps:
        resid = np.mean([np.mean(list(r["achieved_minus_baseline"].values()))
                         for r in caps])
        print(f"mean achieved-minus-baseline projection under cap "
              f"(should be ~>=0 if the clamp holds): {resid:+.3f}")


if __name__ == "__main__":
    main()
