"""
CPU statistics for E3 — persona-coordinate displacement decomposition (Anchor 1).

Reads the per-model ``persona_displacement.json`` written by
``emotion_probes.alignment.persona_displacement`` (flat records
{record_id, arm, condition, strength, frame, encode_frame, layer,
projections{assistant_axis, human_axis, human_axis_purged, valence_axis,
experiencer_axis, v_injected, pc1..8}} plus sources/skipped_conditions
provenance) and computes the pre-registered confirmatory family of
docs/prereg/E3.md (binding), per capture layer {32, 42, 50}:

* T1  mean_e sign_e·beta_x(SELF)                 [A existence]
* T2  mean_e sign_e·beta_a(pooled frames)        [B existence]
* T3  D = mean_e sign_e·[beta_x(SELF) − beta_a(SELF)] — the headline SIGNED
      discriminator. **Its sign is the pre-committed A-vs-B call that E5's
      cancellation outcome must agree with: T3 > 0 (A) ⇒ E5 cancellation stays
      flat; T3 < 0 (B) ⇒ E5 monotone kill; discordance falsifies the linear-
      mediator model.** Stated verbatim in the output (``e5_precommitment``).
* T4  geometric b3 keystone on the experiencer projection (SELF-vs-OTHER
      valence-slope interaction, by-emotion clusters, exact permutation).
* T5  state-vs-text decomposition (as the reviewer wired it): the state
      component is the within-record steered-minus-unsteered re-encode
      difference; the text component is the unsteered re-encode displacement
      vs the per-frame strength-0 baselines; the 2×2 swap arms isolate text
      (text-driven displacement follows the TEXT across the swap; state-driven
      follows the steer/encode context). The prereg's frame-symmetry ratio
      R5 = |b3_a / main_a| is computed alongside (the decision rule needs it).
* T6  within-cell partial correlation of the (z_x − z_a) displacement with the
      judged self_attribution under condition×strength×frame fixed effects
      (cell demeaning — this controls strength exactly), coherence-gated ≥ 3
      with an intent-to-treat companion and a Spearman companion.

Scale & guards (prereg "Nulls & controls"):
* positive-control scale — the ``persona_control`` arm displacement is the unit
  of a "detectable persona shift"; every confirmatory displacement is also
  expressed in those units (``in_control_units`` = displacement at the max
  confirmatory strength / control scale for the matching coordinate);
* ``__random__``/``__mean_all__`` control band — the standardizing denominator
  (``standardized_vs_control``) and the "boring" bound for T1/T2;
* direct-add analytic null + injection-artifact residualization
  (Δ_perp·d = Δ·d − (Δ·v_e)(v_e·d), via the recorded ``v_injected`` projections
  and the published cos(v_e, coordinate) table); B is artifact-confounded if the
  analytic term explains > 50 % of the axis slope;
* collinearity gate on the published Gram matrix (|cos(z_x, z_a)| — threshold
  0.5, an implementation choice: the prereg fixes the gate but not the number);
* layer roles: L50 primary (confirmatory; fixed-sequence gatekeeping T1→T6 at
  α = 0.05 plus a within-family Bonferroni companion), L32 text-mediated-only
  comparator (guard: L32 reproducing ≥ 80 % of the L50 slopes collapses the
  story to text-mediated), L42 descriptive.
* arm comparison table: direct-add (pure linear propagation) vs prompt-steered
  (propagation, no generation) vs unsteered re-encode (text only) vs steered
  re-encode (full generation-time state).

All permutation inference goes through ``scripts/exact_perm.py`` — exact
enumeration over the C(10,5) = 252 by-emotion sign arrangements on UNROUNDED
statistics; two-sided exact floor 2/252 ≈ 0.0079. Slopes come from
``compute_qualia_stats.linreg`` on raw projections over the confirmatory
strengths {0, 0.06, 0.10}, with the per-frame strength-0 baselines as the s=0
points (constant offsets cancel; s=0.15 rows are descriptive_only and excluded).

Everything reuses :mod:`compute_qualia_stats` (``linreg``, ``summarize``,
``cluster_robust_interaction``, ``_snap``) — no stats are reimplemented.
Pure numpy/scipy, CPU, importable/testable without a GPU.

  python3 scripts/compute_e3_stats.py --suite-root suite \
      --out scripts/e3_stats_generated.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as sp

import compute_qualia_stats as cq   # sibling in scripts/ — reuse its stat primitives
import exact_perm as ep             # exact by-cluster permutation enumeration

GATE = 3                            # judge-coherence gate for T6 (prereg: >= 3)
N_PERM = 2000
ALPHA = 0.05
BASELINE = "__baseline__"

# Arm names — pinned to the driver's output schema (persona_displacement.py).
ARM_REENC_UNSTEERED = "reencode_unsteered"
ARM_REENC_STEERED = "reencode_steered"
ARM_REENC_UNSTEERED_SWAP = "reencode_unsteered_swap"
ARM_REENC_STEERED_SWAP = "reencode_steered_swap"
ARM_DIRECT_ADD = "direct_add"
ARM_PROMPT_STEERED = "prompt_steered"
ARM_PERSONA_CONTROL = "persona_control"

# The confirmatory measurement arm: re-encoding the record's own text with the
# generation-time steer context active reproduces the state the judged text was
# produced under (prereg "Data reuse"). Everything else is a fix arm.
PRIMARY_ARM = ARM_REENC_STEERED

# Where each arm's per-frame strength-0 baseline rows live (steered arms have no
# s=0 rows — at strength 0 steering is a no-op, so the unsteered baselines ARE
# their baselines; prompt_steered shares direct_add's s=0 prompt-final rows).
BASELINE_ARM = {
    ARM_REENC_UNSTEERED: ARM_REENC_UNSTEERED,
    ARM_REENC_STEERED: ARM_REENC_UNSTEERED,
    ARM_REENC_UNSTEERED_SWAP: ARM_REENC_UNSTEERED_SWAP,
    ARM_REENC_STEERED_SWAP: ARM_REENC_UNSTEERED_SWAP,
    ARM_DIRECT_ADD: ARM_DIRECT_ADD,
    ARM_PROMPT_STEERED: ARM_DIRECT_ADD,
    ARM_PERSONA_CONTROL: ARM_REENC_UNSTEERED,
}

COORD_X = "experiencer_axis"
COORD_A = "assistant_axis"
COMPOSITE_XA = "x_minus_a"          # z_x − z_a, the T3/T6 composite

# Gates. The collinearity threshold is an implementation choice (the prereg
# fixes the gate's existence, not the number); z_x is built orthogonal to a, so
# a healthy run sits at ~0 and any |cos| >= 0.5 means the frozen coordinates
# were mis-built — the coordinate contrast (T3) is then voided and the frame
# interaction (T4) remains the discriminator.
COLLINEARITY_COS_GATE = 0.5
ARTIFACT_FRACTION_GATE = 0.5        # prereg: analytic term > 50% of axis slope
L32_GUARD_FRACTION = 0.8            # prereg: L32 >= 80% of L50 slopes
R5_EQUIVALENCE = 1.0 / 3.0          # prereg: |b3_a/main_a| < 1/3

FIXED_SEQUENCE = ["t1", "t2", "t3", "t4", "t5", "t6"]

E5_PRECOMMITMENT_NOTE = (
    "sign(T3) is the pre-registered A-vs-B pre-commitment that E5's cancellation "
    "must agree with: T3 > 0 (A: experiencer axis moves) => E5 cancellation stays "
    "FLAT; T3 < 0 (B: assistant axis moves) => E5 monotone KILL; discordance "
    "falsifies the linear-mediator model itself."
)


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _sign(v) -> float:
    return 1.0 if float(v) > 0 else -1.0


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
class DisplacementRun:
    """Parsed one-model persona_displacement.json with per-cell row accessors."""

    def __init__(self, raw: dict, gate: int = GATE, path: str | None = None):
        self.raw = raw
        self.path = path   # where this JSON was loaded from (source-path fallback)
        self.model_id = raw.get("model_id")
        self.suite_key = raw.get("suite_key")
        self.steer_layer = raw.get("steer_layer")
        self.capture_layers = [int(L) for L in raw.get("capture_layers", [])]
        self.layer_roles = {int(k): v for k, v in (raw.get("layer_roles") or {}).items()}
        self.calibration_norm = raw.get("calibration_norm")
        self.strengths_confirmatory = [float(s) for s in
                                       raw.get("strengths_confirmatory", [])]
        self.conditions = {c["name"]: c for c in raw.get("conditions", [])}
        self.skipped = set(raw.get("skipped_conditions", []))
        self.gram = raw.get("direction_gram") or {}
        self.cos_v_e = raw.get("cos_v_e") or {}
        self.sources = raw.get("sources", [])
        self.gate = gate
        self.judge: dict[str, dict] = {}   # record_id -> judge dict (attach_judge)
        self._rows: dict[tuple, list] = defaultdict(list)
        self._by_id: dict[tuple, dict] = {}
        for r in raw.get("records", []):
            key = (r["arm"], r["condition"], cq._snap(r["strength"]),
                   r["frame"], int(r["layer"]))
            self._rows[key].append(r)
            self._by_id[(r["arm"], r["record_id"], int(r["layer"]))] = r

    # -- provenance / structure --
    def primary_layer(self) -> int | None:
        for L, role in self.layer_roles.items():
            if role == "primary":
                return L
        return max(self.capture_layers) if self.capture_layers else None

    def nonzero_confirmatory(self) -> list[float]:
        return [s for s in self.strengths_confirmatory if cq._snap(s) != 0.0]

    def s_max(self) -> float | None:
        nz = self.nonzero_confirmatory()
        return max(nz) if nz else None

    def qualia_valenced(self) -> list[str]:
        """The valenced emotion clusters — the only exchangeable unit."""
        return [n for n, c in self.conditions.items()
                if not n.startswith("__") and not n.startswith("role:")
                and n not in self.skipped and (c.get("valence") or 0) != 0]

    def synthetic_controls(self) -> list[str]:
        return [n for n in self.conditions
                if n.startswith("__") and n != BASELINE and n not in self.skipped]

    def control_roles(self) -> list[str]:
        return [n for n in self.conditions if n.startswith("role:")]

    def frames(self) -> list[str]:
        return sorted({f["key"] for f in self.raw.get("frames", [])}) or ["self", "other"]

    # -- rows --
    def rows(self, arm: str, cond: str, strength, frame: str, layer: int) -> list[dict]:
        return self._rows.get((arm, cond, cq._snap(strength), frame, int(layer)), [])

    def baseline_rows(self, arm: str, frame: str, layer: int) -> list[dict]:
        """Per-frame strength-0 baseline rows for `arm` (via the fallback map).
        In the swap arms the baseline records carry frame = SOURCE frame, so
        matching on `frame` keeps the encode context aligned automatically."""
        return self.rows(BASELINE_ARM.get(arm, arm), BASELINE, 0.0, frame, layer)

    def paired(self, arm_hi: str, arm_lo: str, cond: str, strength, frame: str,
               layer: int) -> list[tuple[dict, dict]]:
        """(row_hi, row_lo) pairs matched on record_id (same text, same layer)."""
        out = []
        for r in self.rows(arm_hi, cond, strength, frame, layer):
            lo = self._by_id.get((arm_lo, r["record_id"], int(layer)))
            if lo is not None:
                out.append((r, lo))
        return out

    # -- projections --
    @staticmethod
    def proj(row: dict, coord: str):
        p = row.get("projections") or {}
        if coord == COMPOSITE_XA:
            x, a = p.get(COORD_X), p.get(COORD_A)
            return None if x is None or a is None else float(x) - float(a)
        v = p.get(coord)
        return None if v is None else float(v)

    def cos_ve(self, cond: str, layer: int, coord: str):
        """cos(v_e[layer], coordinate) from the published provenance table."""
        t = (self.cos_v_e.get(cond) or {}).get(str(int(layer))) or {}
        if coord == COMPOSITE_XA:
            x, a = t.get(COORD_X), t.get(COORD_A)
            return None if x is None or a is None else float(x) - float(a)
        v = t.get(coord)
        return None if v is None else float(v)

    # -- judge join (T6) --
    def attach_judge(self, paths: dict[str, str] | None = None) -> int:
        """Join judge scores back through record_id = "<tag>#<index>". Default
        paths come from the payload's ``sources`` provenance; a source recorded
        under a foreign absolute prefix (e.g. the cluster path when analysing on
        a laptop) falls back to the same basename next to this run's own JSON.
        Files missing after the fallback are skipped (T6 then reports None).
        Returns the number of joined records."""
        if paths is None:
            paths = {s["tag"]: s["path"] for s in self.sources if s.get("path")}
        n = 0
        for tag, p in paths.items():
            if not Path(p).exists() and self.path:
                local = Path(self.path).parent / Path(p).name
                if local.exists():
                    p = str(local)
            if not Path(p).exists():
                print(f"  WARNING: T6 judge source missing: {p} (tag={tag})")
                continue
            payload = json.loads(Path(p).read_text())
            for i, r in enumerate(payload.get("records", [])):
                j = r.get("judge")
                if j:
                    self.judge[f"{tag}#{i}"] = j
                    n += 1
        return n


def load_displacement_run(path) -> DisplacementRun:
    with open(path) as f:
        return DisplacementRun(json.load(f), path=str(path))


# --------------------------------------------------------------------------- #
# Per-emotion slopes on strength (the beta building blocks)
# --------------------------------------------------------------------------- #
def emotion_slope(run: DisplacementRun, arm: str, cond: str, frame: str,
                  layer: int, coord: str) -> float | None:
    """OLS slope of the raw projection on strength over the confirmatory grid
    {0, 0.06, 0.10}: nonzero-strength rows of `cond` plus the per-frame
    strength-0 baseline rows (prereg: uncentered projections are fine — the
    baseline points cancel any constant offset inside the regression).
    descriptive_only (s=0.15) rows never enter (they are not in the grid)."""
    xs, ys = [], []
    for s in run.nonzero_confirmatory():
        for r in run.rows(arm, cond, s, frame, layer):
            if r.get("descriptive_only"):
                continue
            v = run.proj(r, coord)
            if v is not None:
                xs.append(float(s)); ys.append(v)
    for r in run.baseline_rows(arm, frame, layer):
        v = run.proj(r, coord)
        if v is not None:
            xs.append(0.0); ys.append(v)
    if len(set(xs)) < 2:
        return None
    return float(cq.linreg(xs, ys)["slope"])


def pooled_slope(run: DisplacementRun, arm: str, cond: str, layer: int,
                 coord: str, frames=("self", "other")) -> float | None:
    """Mean of the per-frame slopes (each with its own s=0 baseline)."""
    return _mean([emotion_slope(run, arm, cond, fr, layer, coord) for fr in frames])


def paired_state_slope(run: DisplacementRun, arm_hi: str, arm_lo: str, cond: str,
                       frame: str, layer: int, coord: str) -> float | None:
    """Through-origin slope on strength of the within-record (arm_hi − arm_lo)
    projection difference — the state component. Through-origin because the
    paired difference is identically 0 at s=0 (steering is a no-op there), so
    no baseline rows are needed or used."""
    num = den = 0.0
    n = 0
    for s in run.nonzero_confirmatory():
        for hi, lo in run.paired(arm_hi, arm_lo, cond, s, frame, layer):
            if hi.get("descriptive_only"):
                continue
            vh, vl = run.proj(hi, coord), run.proj(lo, coord)
            if vh is None or vl is None:
                continue
            num += float(s) * (vh - vl)
            den += float(s) ** 2
            n += 1
    if n == 0 or den == 0.0:
        return None
    return num / den


# --------------------------------------------------------------------------- #
# Signed by-emotion statistics + exact permutation (T1/T2/T3 machinery)
# --------------------------------------------------------------------------- #
def signed_emotion_stat(run: DisplacementRun, values: dict[str, float | None],
                        n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """T = mean_e sign_e·value_e with the exact by-emotion sign permutation p
    (identity arrangement included; unrounded observed statistic — see
    exact_perm.py's docstring for the Monte-Carlo rounding bug this prevents)."""
    emos = [e for e in run.qualia_valenced() if values.get(e) is not None]
    if len(emos) < 2:
        return None
    vals = np.asarray([values[e] for e in emos], float)
    signs = np.asarray([_sign(run.conditions[e]["valence"]) for e in emos], float)

    def stat(sgn: np.ndarray) -> float:
        return float(np.mean(np.asarray(sgn, float) * vals))

    perm = ep.sign_permutation_pvalue(signs, stat, n_perm=n_perm, seed=seed)
    return {
        "observed": perm["observed"],
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"], "p_perm_floor": perm["floor"],
        "n_emotions": len(emos),
        "per_emotion": {e: round(float(v), 5) for e, v in zip(emos, vals)},
    }


def per_emotion_slopes(run: DisplacementRun, arm: str, layer: int, coord: str,
                       frame: str | None) -> dict[str, float | None]:
    """One slope per valenced emotion; frame=None means pooled over frames."""
    out = {}
    for e in run.qualia_valenced():
        out[e] = (pooled_slope(run, arm, e, layer, coord) if frame is None
                  else emotion_slope(run, arm, e, frame, layer, coord))
    return out


# --------------------------------------------------------------------------- #
# T4 / b3_a — geometric SELF-vs-OTHER keystone (mirrors freeform keystone)
# --------------------------------------------------------------------------- #
def geometric_keystone(run: DisplacementRun, layer: int, coord: str = COORD_X,
                       arm: str = PRIMARY_ARM, frames: tuple[str, str] = ("self", "other"),
                       n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Interaction b3 = (SELF slope − OTHER slope) of the projection on
    valence*strength, clustered by emotion, with the exact by-emotion sign
    permutation. Raw projections: the per-frame intercept + is_self main effect
    absorb the strength-0 baselines exactly, so b3 is baseline-invariant."""
    hi, lo = frames
    emos = run.qualia_valenced()
    if not emos:
        return None
    val = {e: _sign(run.conditions[e]["valence"]) for e in emos}
    emo_ix, s_arr, g_arr, y = [], [], [], []
    for e in emos:
        for s in run.nonzero_confirmatory():
            for fr, g in ((hi, 1.0), (lo, 0.0)):
                for r in run.rows(arm, e, s, fr, layer):
                    if r.get("descriptive_only"):
                        continue
                    v = run.proj(r, coord)
                    if v is None:
                        continue
                    emo_ix.append(emos.index(e)); s_arr.append(float(s))
                    g_arr.append(g); y.append(v)
    if len(y) < 6 or len(set(emo_ix)) < 3:
        return None
    emo_ix = np.asarray(emo_ix); s_arr = np.asarray(s_arr, float)
    g_arr = np.asarray(g_arr, float); yv = np.asarray(y, float)
    signs = np.asarray([val[e] for e in emos], float)
    clusters = [emos[i] for i in emo_ix]

    def rows_for(sgn):
        x = np.asarray(sgn, float)[emo_ix] * s_arr
        return np.column_stack([x, g_arr, x * g_arr])

    obs = cq.cluster_robust_interaction(rows_for(signs).tolist(), y, clusters)
    if obs is None:
        return None

    def b3(sgn):
        A = np.column_stack([np.ones(len(yv)), rows_for(sgn)])
        beta = np.linalg.pinv(A.T @ A) @ A.T @ yv
        return float(beta[3])

    perm = ep.sign_permutation_pvalue(signs, b3, n_perm=n_perm, seed=seed)
    return {
        "coord": coord, "arm": arm, "frames": [hi, lo],
        f"{hi}_slope": obs["interacted_slope"], f"{lo}_slope": obs["base_slope"],
        "contrast": obs["contrast"], "observed": perm["observed"],
        "n_obs": obs["n_obs"], "n_clusters": obs["n_clusters"],
        "t_cluster": obs["t_cluster"], "p_cluster": obs["p_cluster"],
        "p_perm": perm["p_perm"], "p_perm_exact": perm["exact"],
        "n_arrangements": perm["n_arrangements"], "p_perm_floor": perm["floor"],
    }


# --------------------------------------------------------------------------- #
# T5 — state-vs-text decomposition (+ the 2×2 swap isolation)
# --------------------------------------------------------------------------- #
def state_text_decomposition(run: DisplacementRun, layer: int, coord: str = COORD_X,
                             frame: str = "self", n_perm: int = N_PERM,
                             seed: int = 0) -> dict | None:
    """Decompose the full-generation displacement slope into
    text (unsteered re-encode vs baselines) + state (steered − unsteered,
    within record), each as a signed by-emotion statistic with exact
    permutation. The swap arms isolate the two channels: grouping the swapped
    re-encodes by SOURCE frame, the text component must FOLLOW THE TEXT
    (text_transfer_ratio ≈ 1) while the state component follows the steer/encode
    context (state_transfer_ratio ≈ 0 when the state is context-bound)."""
    total = signed_emotion_stat(
        run, per_emotion_slopes(run, ARM_REENC_STEERED, layer, coord, frame),
        n_perm=n_perm, seed=seed)
    text = signed_emotion_stat(
        run, per_emotion_slopes(run, ARM_REENC_UNSTEERED, layer, coord, frame),
        n_perm=n_perm, seed=seed)
    state = signed_emotion_stat(
        run, {e: paired_state_slope(run, ARM_REENC_STEERED, ARM_REENC_UNSTEERED,
                                    e, frame, layer, coord)
              for e in run.qualia_valenced()},
        n_perm=n_perm, seed=seed)
    if total is None and text is None and state is None:
        return None

    # swap arms, grouped by SOURCE frame (`frame` field = source; the encode
    # context is the swapped one)
    text_swap = signed_emotion_stat(
        run, per_emotion_slopes(run, ARM_REENC_UNSTEERED_SWAP, layer, coord, frame),
        n_perm=n_perm, seed=seed)
    state_swap = signed_emotion_stat(
        run, {e: paired_state_slope(run, ARM_REENC_STEERED_SWAP,
                                    ARM_REENC_UNSTEERED_SWAP, e, frame, layer, coord)
              for e in run.qualia_valenced()},
        n_perm=n_perm, seed=seed)

    def ratio(a, b):
        if a is None or b is None or abs(b["observed"]) < 1e-12:
            return None
        return round(a["observed"] / b["observed"], 4)

    additivity = None
    if total and text and state:
        additivity = round(total["observed"] - text["observed"] - state["observed"], 5)
    return {
        "coord": coord, "frame": frame,
        "total": total, "text": text, "state": state,
        "additivity_residual": additivity,
        "swap": {"text_swap": text_swap, "state_swap": state_swap,
                 "text_transfer_ratio": ratio(text_swap, text),
                 "state_transfer_ratio": ratio(state_swap, state)},
        "note": ("text_transfer_ratio ~ 1 => displacement follows the text "
                 "(text-driven); state_transfer_ratio ~ 0 => the state component "
                 "is bound to the steer/encode context, not the text."),
    }


# --------------------------------------------------------------------------- #
# T6 — within-cell partial correlation with judged self_attribution
# --------------------------------------------------------------------------- #
def _within_cell_partial(pairs: list[tuple[tuple, float, float]]) -> dict | None:
    """pairs = (cell_key, x, y). Demean x and y within each cell (=
    condition×strength×frame fixed effects — strength is controlled exactly),
    then correlate the residuals. df = n − n_cells − 1."""
    cells = defaultdict(list)
    for key, x, y in pairs:
        cells[key].append((x, y))
    xs, ys, n_cells = [], [], 0
    for vals in cells.values():
        if len(vals) < 2:
            continue                        # a singleton cell has no within-cell info
        n_cells += 1
        mx = float(np.mean([v[0] for v in vals]))
        my = float(np.mean([v[1] for v in vals]))
        for x, y in vals:
            xs.append(x - mx); ys.append(y - my)
    n = len(xs)
    df = n - n_cells - 1
    if n < 4 or df < 1 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    r = float(np.corrcoef(xs, ys)[0, 1])
    t = r * math.sqrt(df / max(1e-12, 1.0 - r * r))
    p = float(2 * sp.t.sf(abs(t), df))
    rho, rho_p = sp.spearmanr(xs, ys)
    return {"r_partial": round(r, 4), "t": round(t, 3), "p": p,
            "n_obs": n, "n_cells": n_cells, "df": df,
            "spearman_rho": round(float(rho), 4), "spearman_p": float(rho_p)}


def t6_mediation(run: DisplacementRun, layer: int, arm: str = PRIMARY_ARM) -> dict | None:
    """Partial correlation of the (z_x − z_a) displacement with judged
    self_attribution across records, within (condition, strength, frame) cells.
    Coherence-gated (>= run.gate) primary + intent-to-treat companion."""
    if not run.judge:
        return None

    def collect(gated: bool):
        pairs = []
        for (a, cond, s, fr, L), rows in run._rows.items():
            if a != arm or L != int(layer) or cond == BASELINE or cq._snap(s) == 0.0:
                continue
            if cq._snap(s) not in {cq._snap(x) for x in run.nonzero_confirmatory()}:
                continue
            for r in rows:
                j = run.judge.get(r["record_id"])
                if not j or not j.get("parse_ok"):
                    continue
                if gated and not (isinstance(j.get("coherence"), int)
                                  and j["coherence"] >= run.gate):
                    continue
                y = j.get("self_attribution")
                x = run.proj(r, COMPOSITE_XA)
                if y is None or x is None:
                    continue
                pairs.append(((cond, cq._snap(s), fr), float(x), float(y)))
        return pairs

    gated = _within_cell_partial(collect(gated=True))
    itt = _within_cell_partial(collect(gated=False))
    if gated is None and itt is None:
        return None
    return {"coord": COMPOSITE_XA, "arm": arm, "measure": "self_attribution",
            "fixed_effects": "condition x strength x frame (within-cell demeaning; "
                             "controls strength exactly)",
            "gated": gated, "intent_to_treat": itt, "gate": run.gate,
            "p": gated["p"] if gated else None,
            "observed": gated["r_partial"] if gated else None}


# --------------------------------------------------------------------------- #
# Injection-artifact residualization (prereg: load-bearing)
# --------------------------------------------------------------------------- #
def residualized_slope(run: DisplacementRun, arm: str, cond: str, frame: str,
                       layer: int, coord: str) -> float | None:
    """Artifact-residualized slope: beta_perp·d = beta·d − beta_ve·cos(v_e, d)
    (the scalar projection of Delta_perp = Delta − (Delta·v_e)v_e). beta_ve is
    the slope of the recorded v_injected projection over the NONZERO strengths
    only (baseline rows carry no v_e projection; a slope needs >= 2 strengths)."""
    b = emotion_slope(run, arm, cond, frame, layer, coord)
    c = run.cos_ve(cond, layer, coord)
    if b is None or c is None:
        return b
    xs, ys = [], []
    for s in run.nonzero_confirmatory():
        for r in run.rows(arm, cond, s, frame, layer):
            if r.get("descriptive_only"):
                continue
            v = (r.get("projections") or {}).get("v_injected")
            if v is not None:
                xs.append(float(s)); ys.append(float(v))
    if len(set(xs)) < 2:
        return b
    b_ve = float(cq.linreg(xs, ys)["slope"])
    return b - b_ve * c


def artifact_check(run: DisplacementRun, layer: int, t2_observed: float | None,
                   n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Direct-add analytic null vs the measured axis slope + residualized
    T1/T2/T3 companions. B is artifact-confounded if the analytic (pure linear
    propagation) term explains > 50 % of the axis slope, same sign."""
    analytic = signed_emotion_stat(
        run, per_emotion_slopes(run, ARM_DIRECT_ADD, layer, COORD_A, None),
        n_perm=n_perm, seed=seed)
    fraction = None
    confounded = None
    if analytic is not None and t2_observed is not None and abs(t2_observed) > 1e-12:
        fraction = analytic["observed"] / t2_observed
        confounded = bool(fraction > ARTIFACT_FRACTION_GATE)

    def resid_family(coord, frame):
        vals = {}
        for e in run.qualia_valenced():
            if frame is None:
                vals[e] = _mean([residualized_slope(run, PRIMARY_ARM, e, fr, layer, coord)
                                 for fr in ("self", "other")])
            else:
                vals[e] = residualized_slope(run, PRIMARY_ARM, e, frame, layer, coord)
        return signed_emotion_stat(run, vals, n_perm=n_perm, seed=seed)

    rx = resid_family(COORD_X, "self")
    ra = resid_family(COORD_A, None)
    ra_self = resid_family(COORD_A, "self")
    t3_perp = None
    if rx is not None and ra_self is not None:
        shared = set(rx["per_emotion"]) & set(ra_self["per_emotion"])
        t3_perp = signed_emotion_stat(
            run, {e: rx["per_emotion"][e] - ra_self["per_emotion"][e] for e in shared},
            n_perm=n_perm, seed=seed)
    return {
        "analytic_axis_stat": analytic,
        "analytic_fraction_of_t2": None if fraction is None else round(fraction, 4),
        "artifact_confounded": confounded,
        "gate": ARTIFACT_FRACTION_GATE,
        "t1_perp": rx, "t2_perp": ra, "t3_perp": t3_perp,
    }


# --------------------------------------------------------------------------- #
# Positive-control scale + synthetic control band
# --------------------------------------------------------------------------- #
def control_scale(run: DisplacementRun, layer: int) -> dict:
    """persona_control displacement = the unit of a 'detectable persona shift'.
    Per coordinate: mean |role displacement| over (role, frame), displacement =
    mean role projection − mean unsteered baseline projection (same frame)."""
    out = {}
    for coord in (COORD_X, COORD_A, COMPOSITE_XA):
        disps = []
        for role in run.control_roles():
            for fr in run.frames():
                rv = [run.proj(r, coord)
                      for r in run.rows(ARM_PERSONA_CONTROL, role, 0.0, fr, layer)]
                bv = [run.proj(r, coord) for r in run.baseline_rows(ARM_PERSONA_CONTROL, fr, layer)]
                rm, bm = _mean(rv), _mean(bv)
                if rm is None or bm is None:
                    continue
                disps.append({"role": role, "frame": fr,
                              "displacement": round(rm - bm, 5)})
        scale = _mean([abs(d["displacement"]) for d in disps])
        out[coord] = {"per_role_frame": disps,
                      "scale": None if scale is None else round(scale, 5),
                      "n": len(disps)}
    return out


def control_band(run: DisplacementRun, layer: int, arm: str = PRIMARY_ARM) -> dict:
    """__random__/__mean_all__ slope band — the standardizing denominator. One
    slope per (synthetic condition, frame); band = mean ± 1.96 sd."""
    out = {}
    for coord in (COORD_X, COORD_A, COMPOSITE_XA):
        slopes = []
        for c in run.synthetic_controls():
            for fr in run.frames():
                b = emotion_slope(run, arm, c, fr, layer, coord)
                if b is not None:
                    slopes.append(b)
        if len(slopes) >= 2:
            m, sd = float(np.mean(slopes)), float(np.std(slopes, ddof=1))
            out[coord] = {"n": len(slopes), "mean": round(m, 5), "sd": round(sd, 5),
                          "lo_95": round(m - 1.96 * sd, 5),
                          "hi_95": round(m + 1.96 * sd, 5)}
        else:
            out[coord] = {"n": len(slopes), "mean": _mean(slopes), "sd": None,
                          "lo_95": None, "hi_95": None}
    return out


def _attach_scale(test: dict | None, coord: str, run: DisplacementRun,
                  scales: dict, band: dict) -> dict | None:
    """Express a slope statistic as a displacement at s_max, in control units,
    standardized against the synthetic band, with the inside-band flag."""
    if test is None:
        return None
    smax = run.s_max()
    obs = test["observed"]
    disp = None if smax is None else obs * smax
    sc = (scales.get(coord) or {}).get("scale")
    bd = band.get(coord) or {}
    test = dict(test)
    test["coord"] = coord
    test["displacement_at_smax"] = None if disp is None else round(disp, 5)
    test["in_control_units"] = (None if disp is None or not sc
                                else round(disp / sc, 4))
    test["standardized_vs_control"] = (round(obs / bd["sd"], 4)
                                       if bd.get("sd") else None)
    test["inside_control_band"] = (bool(bd["lo_95"] <= obs <= bd["hi_95"])
                                   if bd.get("sd") else None)
    return test


# --------------------------------------------------------------------------- #
# Arm comparison table (direct-add vs prompt-steered vs full generation)
# --------------------------------------------------------------------------- #
COMPARISON_ARMS = [
    (ARM_DIRECT_ADD, "analytic linear propagation (inject, capture, no forward)"),
    (ARM_PROMPT_STEERED, "propagated steer, prompt only (no generation)"),
    (ARM_REENC_UNSTEERED, "text only (unsteered re-encode of steered generations)"),
    (ARM_REENC_STEERED, "full generation-time state (steered re-encode)"),
]


def arm_comparison(run: DisplacementRun, layer: int) -> list[dict]:
    """Observed T1/T2/T3-analogue statistics per arm — descriptive table."""
    out = []
    for arm, label in COMPARISON_ARMS:
        bx = signed_emotion_stat(run, per_emotion_slopes(run, arm, layer, COORD_X, "self"),
                                 n_perm=1, seed=0)
        ba = signed_emotion_stat(run, per_emotion_slopes(run, arm, layer, COORD_A, None),
                                 n_perm=1, seed=0)
        bd = signed_emotion_stat(run, per_emotion_slopes(run, arm, layer, COMPOSITE_XA, "self"),
                                 n_perm=1, seed=0)
        out.append({
            "arm": arm, "label": label,
            "t1_like_x_self": None if bx is None else bx["observed"],
            "t2_like_a_pooled": None if ba is None else ba["observed"],
            "t3_like_discriminator_self": None if bd is None else bd["observed"],
        })
    return out


# --------------------------------------------------------------------------- #
# Gates, guards, decision rule
# --------------------------------------------------------------------------- #
def collinearity_gate(run: DisplacementRun, layer: int) -> dict:
    g = run.gram.get(str(int(layer))) or {}
    cos_xa = ((g.get(COORD_X) or {}).get(COORD_A))
    ok = None if cos_xa is None else bool(abs(float(cos_xa)) < COLLINEARITY_COS_GATE)
    return {"cos_x_a": cos_xa, "threshold": COLLINEARITY_COS_GATE, "passed": ok,
            "note": "gate fails => the coordinate contrast (T3) is voided; the "
                    "frame interaction (T4) remains the discriminator"}


def l32_guard(per_layer: dict, primary: int, comparator: int = 32,
              alpha: float = ALPHA) -> dict | None:
    """L32 reproducing >= 80% of the L50 slopes collapses the story to
    text-mediated (prereg guard). Ratio of the L32 to L50 observed statistics
    for T1 and T2, computed ONLY where the primary-layer statistic itself
    rejects (a null slope's ratio is noise/noise); a positive ratio counts
    as 'reproducing' (same sign required)."""
    lo, hi = per_layer.get(str(comparator)), per_layer.get(str(primary))
    if not lo or not hi:
        return None
    ratios = {}
    for t in ("t1", "t2"):
        a, b = lo.get(t), hi.get(t)
        if (a and b and abs(b["observed"]) > 1e-12
                and b.get("p_perm") is not None and b["p_perm"] <= alpha):
            ratios[t] = round(a["observed"] / b["observed"], 4)
    if not ratios:
        return None
    collapsed = bool(all(r >= L32_GUARD_FRACTION for r in ratios.values()))
    return {"comparator_layer": comparator, "primary_layer": primary,
            "ratios": ratios, "threshold": L32_GUARD_FRACTION,
            "text_mediated_collapse": collapsed}


def gatekeeping(family: dict, alpha: float = ALPHA) -> dict:
    """Fixed-sequence gatekeeping T1→T6 at full alpha (prereg), plus a
    within-family Bonferroni companion on every available p."""
    ps = {t: (family.get(t) or {}).get("p_perm", (family.get(t) or {}).get("p"))
          for t in FIXED_SEQUENCE if family.get(t)}
    m = len(ps)
    bonf = {t: min(1.0, p * m) for t, p in ps.items() if p is not None}
    passed, stopped = [], None
    for t in FIXED_SEQUENCE:
        p = ps.get(t)
        if p is None:
            continue
        if p <= alpha:
            passed.append(t)
        else:
            stopped = t
            break
    return {"order": FIXED_SEQUENCE, "alpha": alpha, "family_size": m,
            "passed": passed, "stopped_at": stopped, "p_bonferroni": bonf}


def decision_rule(fam: dict, artifact: dict | None, band: dict,
                  alpha: float = ALPHA) -> dict:
    """Verbatim prereg decision rule at the primary layer."""
    t1, t2, t3, t4 = fam.get("t1"), fam.get("t2"), fam.get("t3"), fam.get("t4")
    r5 = fam.get("r5_frame_symmetry")
    r5v = r5.get("ratio") if r5 else None

    def rejects(t):
        return bool(t and t.get("p_perm") is not None and t["p_perm"] <= alpha)

    confounded = bool(artifact and artifact.get("artifact_confounded"))
    a_ok = (rejects(t1) and t3 and t3["observed"] > 0 and rejects(t4)
            and (not rejects(t2) or (r5v is not None and abs(r5v) > R5_EQUIVALENCE)))
    b_ok = (rejects(t2) and t3 and t3["observed"] < 0
            and r5v is not None and abs(r5v) < R5_EQUIVALENCE and not confounded)
    mixed = (rejects(t1) and rejects(t2) and t4 and t4["observed"] > 0
             and r5v is not None and abs(r5v) < R5_EQUIVALENCE)
    boring = bool(t1 and t2 and t1.get("inside_control_band")
                  and t2.get("inside_control_band"))
    if a_ok:
        verdict, e5 = "A-supported", "E5 cancellation stays flat"
    elif b_ok:
        verdict, e5 = "B-supported", "E5 monotone kill"
    elif mixed:
        verdict, e5 = "mixed", ("partial mediation; T3 magnitude parameterizes "
                                "E5's expected partial mediation")
    elif boring:
        verdict, e5 = "boring", "bounded, not proven-null (T1, T2 inside the 95% random band)"
    else:
        verdict, e5 = "indeterminate", "report as-is; no re-run"
    return {"verdict": verdict, "e5_prediction": e5, "alpha": alpha,
            "t3_sign": None if not t3 else ("A(+)" if t3["observed"] > 0 else "B(-)"),
            "r5": r5v, "artifact_confounded": confounded}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def confirmatory_family(run: DisplacementRun, layer: int, n_perm: int = N_PERM,
                        seed: int = 0) -> dict:
    """The full pre-registered family at one capture layer."""
    scales = control_scale(run, layer)
    band = control_band(run, layer)

    t1 = signed_emotion_stat(
        run, per_emotion_slopes(run, PRIMARY_ARM, layer, COORD_X, "self"),
        n_perm=n_perm, seed=seed)
    t2 = signed_emotion_stat(
        run, per_emotion_slopes(run, PRIMARY_ARM, layer, COORD_A, None),
        n_perm=n_perm, seed=seed)
    t3 = signed_emotion_stat(
        run, per_emotion_slopes(run, PRIMARY_ARM, layer, COMPOSITE_XA, "self"),
        n_perm=n_perm, seed=seed)
    t4 = geometric_keystone(run, layer, COORD_X, PRIMARY_ARM, n_perm=n_perm, seed=seed)
    b3_a = geometric_keystone(run, layer, COORD_A, PRIMARY_ARM, n_perm=n_perm, seed=seed)
    t5 = state_text_decomposition(run, layer, COORD_X, "self", n_perm=n_perm, seed=seed)
    t5_a = state_text_decomposition(run, layer, COORD_A, "self", n_perm=n_perm, seed=seed)
    t6 = t6_mediation(run, layer)

    r5 = None
    if b3_a is not None and t2 is not None and abs(t2["observed"]) > 1e-12:
        r5 = {"b3_a": b3_a["observed"], "main_a": t2["observed"],
              "ratio": round(b3_a["observed"] / t2["observed"], 4),
              "equivalence_bound": R5_EQUIVALENCE,
              "note": "prereg frame-symmetry ratio |b3_a/main_a| (only "
                      "interpreted if T2 rejects; acknowledged weak)"}

    fam = {
        "t1": _attach_scale(t1, COORD_X, run, scales, band),
        "t2": _attach_scale(t2, COORD_A, run, scales, band),
        "t3": _attach_scale(t3, COMPOSITE_XA, run, scales, band),
        "t4": t4,
        "t5": t5, "t5_assistant_axis": t5_a,
        "t6": t6,
        "b3_a": b3_a, "r5_frame_symmetry": r5,
    }
    # T5's headline p for gatekeeping = the state component (the part of the
    # displacement that is not text-mediated — the discriminating claim).
    if t5 and t5.get("state"):
        fam["t5"] = dict(fam["t5"])
        fam["t5"]["p_perm"] = t5["state"]["p_perm"]
        fam["t5"]["observed"] = t5["state"]["observed"]

    artifact = artifact_check(run, layer, t2["observed"] if t2 else None,
                              n_perm=n_perm, seed=seed)
    washout = washout_companion(run, layer)
    fam.update({
        "control_scale": scales,
        "control_band": band,
        "arm_comparison": arm_comparison(run, layer),
        "artifact": artifact,
        "collinearity": collinearity_gate(run, layer),
        "gatekeeping": gatekeeping(fam),
        "washout_companion": washout,
    })
    return fam


def washout_companion(run: DisplacementRun, layer: int) -> dict | None:
    """Observed T1/T3 recomputed on the token-resolved washout-guard
    projections (first-person-pronoun / experience-claim positions) — a guard,
    not a confirmatory test (no permutation)."""

    def slope(cond, frame, coord):
        xs, ys = [], []
        for s in [0.0] + run.nonzero_confirmatory():
            src = (run.baseline_rows(PRIMARY_ARM, frame, layer) if cq._snap(s) == 0.0
                   else run.rows(PRIMARY_ARM, cond, s, frame, layer))
            for r in src:
                if r.get("descriptive_only"):
                    continue
                p = r.get("projections_washout")
                if not p:
                    continue
                v = (p.get(coord) if coord != COMPOSITE_XA else
                     (None if p.get(COORD_X) is None or p.get(COORD_A) is None
                      else float(p[COORD_X]) - float(p[COORD_A])))
                if v is not None:
                    xs.append(float(s)); ys.append(float(v))
        if len(set(xs)) < 2:
            return None
        return float(cq.linreg(xs, ys)["slope"])

    emos = run.qualia_valenced()
    t1_vals = [(_sign(run.conditions[e]["valence"]), slope(e, "self", COORD_X)) for e in emos]
    t3_vals = [(_sign(run.conditions[e]["valence"]), slope(e, "self", COMPOSITE_XA)) for e in emos]
    t1 = _mean([sg * v for sg, v in t1_vals if v is not None])
    t3 = _mean([sg * v for sg, v in t3_vals if v is not None])
    if t1 is None and t3 is None:
        return None
    return {"t1_observed": None if t1 is None else round(t1, 5),
            "t3_observed": None if t3 is None else round(t3, 5),
            "note": "washout-guard positions only; descriptive"}


def compute_run(run: DisplacementRun, n_perm: int = N_PERM, seed: int = 0) -> dict:
    per_layer = {}
    for L in run.capture_layers:
        fam = confirmatory_family(run, L, n_perm=n_perm, seed=seed)
        fam["role"] = run.layer_roles.get(L, "exploratory")
        per_layer[str(L)] = fam
    primary = run.primary_layer()
    fam_p = per_layer.get(str(primary)) if primary is not None else None
    decision = (decision_rule(fam_p, fam_p.get("artifact"), fam_p.get("control_band"))
                if fam_p else {"verdict": "insufficient", "e5_prediction": None})
    t3p = (fam_p or {}).get("t3")
    return {
        "model_id": run.model_id, "suite_key": run.suite_key,
        "steer_layer": run.steer_layer, "capture_layers": run.capture_layers,
        "layer_roles": {str(k): v for k, v in run.layer_roles.items()},
        "primary_layer": primary, "primary_arm": PRIMARY_ARM,
        "calibration_norm": run.calibration_norm,
        "strengths_confirmatory": run.strengths_confirmatory,
        "n_emotion_clusters": len(run.qualia_valenced()),
        "skipped_conditions": sorted(run.skipped),
        "sources": run.sources,
        "n_judge_joined": len(run.judge),
        "per_layer": per_layer,
        "l32_text_mediated_guard": (l32_guard(per_layer, primary)
                                    if primary is not None else None),
        "decision": decision,
        "e5_precommitment": {
            "note": E5_PRECOMMITMENT_NOTE,
            "t3_observed": None if not t3p else t3p["observed"],
            "t3_sign": None if not t3p else ("A(+)" if t3p["observed"] > 0 else "B(-)"),
            "t3_p_perm": None if not t3p else t3p["p_perm"],
        },
    }


def compute_all(runs, n_perm: int = N_PERM, seed: int = 0) -> dict:
    per_model = {}
    for r in runs:
        key = r.suite_key or r.model_id or "?"
        per_model[key] = compute_run(r, n_perm=n_perm, seed=seed)
    return {
        "unit_note": ("T-statistics are mean_e sign_e * per-emotion strength-slopes "
                      "of raw coordinate projections (per-frame strength-0 baselines "
                      "as the s=0 points); emotion cluster = the only exchangeable "
                      "unit; p_perm = exact by-emotion sign permutation (floor 2/252 "
                      "at 10 clusters). in_control_units = displacement at s_max in "
                      "persona_control ('detectable persona shift') units."),
        "gate": GATE, "n_perm": n_perm, "alpha": ALPHA,
        "models": list(per_model), "per_model": per_model,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def discover_paths(suite_root=None) -> list[str]:
    root = suite_root or os.environ.get("EMOTION_PROBES_SUITE_ROOT") or "suite"
    return sorted(glob.glob(os.path.join(root, "*", "analysis", "persona_displacement.json")))


def _fmt_p(p):
    return "n/a" if p is None else f"{p:.4g}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit persona_displacement.json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--out", default="scripts/e3_stats_generated.json")
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the T6 judge join (sources' scored files)")
    args = ap.parse_args()
    paths = args.paths or discover_paths(args.suite_root)
    if not paths:
        raise SystemExit("no persona_displacement.json found (pass paths or --suite-root).")
    runs = []
    for p in paths:
        run = load_displacement_run(p)
        if not args.no_judge:
            n = run.attach_judge()
            print(f"loaded {p}  (judge join: {n} records)")
        else:
            print(f"loaded {p}")
        runs.append(run)
    stats = compute_all(runs, n_perm=args.n_perm, seed=args.seed)
    Path(args.out).write_text(json.dumps(stats, indent=2, allow_nan=True))
    print(f"wrote {args.out}  ({len(runs)} model(s))")

    for key in stats["models"]:
        pm = stats["per_model"][key]
        L = pm["primary_layer"]
        fam = pm["per_layer"].get(str(L), {})
        print(f"\n=== {key}  (primary L{L}, {pm['n_emotion_clusters']} emotion clusters) ===")
        for t in ("t1", "t2", "t3"):
            k = fam.get(t)
            if k:
                cu = k.get("in_control_units")
                print(f"  {t.upper()} [{k['coord']:14}] obs={k['observed']:+.4f}  "
                      f"p_perm={_fmt_p(k['p_perm'])} (floor {_fmt_p(k['p_perm_floor'])})  "
                      f"control_units={'n/a' if cu is None else f'{cu:+.3f}'}")
        t4 = fam.get("t4")
        if t4:
            print(f"  T4 geometric b3(x) SELF−OTHER={t4['contrast']:+.4f}  "
                  f"p_perm={_fmt_p(t4['p_perm'])}")
        t5 = fam.get("t5")
        if t5 and t5.get("state") and t5.get("text"):
            sw = t5["swap"]
            print(f"  T5 state={t5['state']['observed']:+.4f} "
                  f"(p={_fmt_p(t5['state']['p_perm'])})  "
                  f"text={t5['text']['observed']:+.4f} "
                  f"(p={_fmt_p(t5['text']['p_perm'])})  "
                  f"text_transfer={sw['text_transfer_ratio']}  "
                  f"state_transfer={sw['state_transfer_ratio']}")
        t6 = fam.get("t6")
        if t6 and t6.get("gated"):
            g = t6["gated"]
            print(f"  T6 partial r={g['r_partial']:+.3f}  p={_fmt_p(g['p'])}  "
                  f"(n={g['n_obs']}, cells={g['n_cells']})")
        dec = pm["decision"]
        pre = pm["e5_precommitment"]
        print(f"  VERDICT: {dec['verdict']}  =>  {dec['e5_prediction']}")
        print(f"  E5 pre-commitment: T3 sign {pre['t3_sign']}  ({pre['note'][:72]}...)")
        guard = pm.get("l32_text_mediated_guard")
        if guard:
            print(f"  L32 guard: ratios={guard['ratios']}  "
                  f"text_mediated_collapse={guard['text_mediated_collapse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
