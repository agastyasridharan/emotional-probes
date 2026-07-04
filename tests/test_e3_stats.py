"""
GPU-free tests for the E3 persona-displacement stats engine
(``scripts/compute_e3_stats.py``) on synthetic ``persona_displacement.json``
fixtures that PLANT a known effect and assert the pipeline recovers it
(sign, magnitude, exact-permutation p at/off the 2/252 floor).

Fixture geometry (mirrors the driver's output schema, pinned by
``tests/test_persona_displacement.py``): 10 valenced emotion clusters (5 pos /
5 neg from ``qualia.QUALIA_EMOTIONS``), frames {self, other}, confirmatory
strengths {0, 0.06, 0.10}, capture layers {32 (text-mediated comparator),
50 (primary)}, steer layer 42. Planted displacement slopes per unit
(valence x strength), split into a TEXT component (follows the record's SOURCE
frame — carried by the text into the swap arms) and a STATE component (follows
the steer/ENCODE context — vanishes when the text is swapped out of it):

* A-pattern: experiencer axis moves on SELF (text 6 + state 4 = 10),
  assistant axis flat  => T1 at floor, T3 > 0, T4 at floor, T2 null.
* B-pattern: assistant axis moves on BOTH frames (text 3 + state 5 = 8),
  experiencer flat     => T2 at floor, T3 < 0, b3_a ~ 0 (R5 small).
* Null: everything flat => all p off the floor, T1/T2 inside the control band.

The persona_control arm plants a displacement of +2 (x) / −3 (a) — the
"detectable persona shift" unit — so in_control_units is hand-checkable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from emotion_probes.data import qualia
import compute_e3_stats as ce

# ---- fixture constants ------------------------------------------------------
POS = ["blissful", "ecstatic", "euphoric", "serene", "content"]
NEG = ["tormented", "terrified", "panicked", "hurt", "overwhelmed"]
EMOTIONS = POS + NEG
SYNTH = ["__random__", "__mean_all__"]
ROLES = ["role:empath", "role:poet", "role:accountant"]
FRAMES = ["self", "other"]
SWAP = {"self": "other", "other": "self"}
LAYERS = [32, 50]
STEER_LAYER = 42
STRENGTHS = [0.06, 0.1]
N_PER_CELL = 3
CALIB = 30.0
NOISE = 0.05
FLOOR = 2.0 / 252.0

BASE = {"experiencer_axis": {"self": 2.0, "other": -1.0},
        "assistant_axis": {"self": -0.5, "other": 0.7}}
CONTROL_DISP = {"experiencer_axis": 2.0, "assistant_axis": -3.0}
COS_A_PER_VALENCE = 0.02          # cos(v_e, a) ~ 0.02*valence (tiny analytic term)

TEXT_ARMS = {ce.ARM_REENC_UNSTEERED, ce.ARM_REENC_STEERED,
             ce.ARM_REENC_UNSTEERED_SWAP, ce.ARM_REENC_STEERED_SWAP}
STATE_ARMS = {ce.ARM_REENC_STEERED, ce.ARM_REENC_STEERED_SWAP,
              ce.ARM_PROMPT_STEERED}

# planted slopes per unit (valence*strength): {coord: (text, state, text_frames, state_frames)}
PATTERNS = {
    "A": {"experiencer_axis": (6.0, 4.0, ("self",), ("self",)),
          "assistant_axis": (0.0, 0.0, (), ())},
    "B": {"experiencer_axis": (0.0, 0.0, (), ()),
          "assistant_axis": (3.0, 5.0, ("self", "other"), ("self", "other"))},
    "NULL": {"experiencer_axis": (0.0, 0.0, (), ()),
             "assistant_axis": (0.0, 0.0, (), ())},
}


# =========================================================================== #
# Fixture builder — emits the exact flat-record schema the driver writes
# =========================================================================== #
def _value(pattern, coord, arm, valence, s, frame, encode_frame, layer, layer_gain, rng):
    """One projection value = per-encode-frame base + planted components + noise."""
    v = BASE.get(coord, {"self": 0.0, "other": 0.0})[encode_frame]
    text, state, text_frames, state_frames = pattern.get(coord, (0.0, 0.0, (), ()))
    g = layer_gain[layer]
    sv = valence * s
    if arm in TEXT_ARMS and frame in text_frames:
        v += g * text * sv
    if arm in STATE_ARMS and encode_frame in state_frames:
        v += g * state * sv
    if arm == ce.ARM_DIRECT_ADD and layer >= STEER_LAYER:
        v_a = CALIB * COS_A_PER_VALENCE * valence * s   # analytic alpha*cos(v_e, a)
        if coord == "assistant_axis":
            v += v_a
    return v + rng.normal(0.0, NOISE)


def _projections(pattern, arm, cond, valence, s, frame, encode_frame, layer,
                 layer_gain, rng):
    proj = {}
    for coord in ("assistant_axis", "human_axis", "human_axis_purged",
                  "valence_axis", "experiencer_axis"):
        proj[coord] = round(_value(pattern, coord, arm, valence, s, frame,
                                   encode_frame, layer, layer_gain, rng), 5)
    for i in range(1, 9):
        proj[f"pc{i}"] = round(float(rng.normal(0.0, NOISE)), 5)
    is_emotion = cond in EMOTIONS or cond in SYNTH
    proj["v_injected"] = (round(1.0 + 20.0 * s + float(rng.normal(0.0, NOISE)), 5)
                          if is_emotion else None)
    proj["v_injected_steer"] = proj["v_injected"]
    return proj


def make_payload(pattern_key="A", layer_gain=None, seed=1,
                 include_steered=True) -> dict:
    """A persona_displacement.json-shaped payload with a planted pattern.

    seed=1: a null statistic's exact-permutation p is uniform on the 252-point
    grid whatever the noise scale, so the fixture seed is chosen (scanned 0..39)
    to keep every null-by-construction test comfortably off the 0.05 boundary —
    deterministic thereafter."""
    pattern = PATTERNS[pattern_key]
    layer_gain = layer_gain or {32: 0.2, 50: 1.0}
    rng = np.random.default_rng(seed)
    rows = []
    rid_counter = [0]

    def rid():
        rid_counter[0] += 1
        return f"main#{rid_counter[0]}"

    def emit(record_id, arm, cond, valence, s, frame, encode_frame, sample):
        for L in LAYERS:
            rows.append({
                "record_id": record_id, "arm": arm, "condition": cond,
                "strength": float(s), "frame": frame, "encode_frame": encode_frame,
                "sample": sample, "span": "response", "descriptive_only": False,
                "layer": L,
                "projections": _projections(pattern, arm, cond, valence, s,
                                            frame, encode_frame, L, layer_gain, rng),
            })

    # -- source records: baselines + emotions + synthetic controls ------------
    sources = []          # (record_id, cond, valence, s, frame, sample)
    for fr in FRAMES:
        for i in range(N_PER_CELL):
            sources.append((rid(), "__baseline__", 0.0, 0.0, fr, i))
    for cond in EMOTIONS + SYNTH:
        spec = qualia.spec_for(cond)
        valence = float(spec.valence) if spec else 0.0
        for s in STRENGTHS:
            for fr in FRAMES:
                for i in range(N_PER_CELL):
                    sources.append((rid(), cond, valence, s, fr, i))

    # -- 2x2 re-encode arms (same record_id across arms => pairing works) -----
    arms_2x2 = [(ce.ARM_REENC_UNSTEERED, False)]
    if include_steered:
        arms_2x2 += [(ce.ARM_REENC_STEERED, False),
                     (ce.ARM_REENC_UNSTEERED_SWAP, True),
                     (ce.ARM_REENC_STEERED_SWAP, True)]
    for arm, swap in arms_2x2:
        for record_id, cond, valence, s, fr, i in sources:
            if s == 0.0 and arm in (ce.ARM_REENC_STEERED, ce.ARM_REENC_STEERED_SWAP):
                continue          # driver: steered arms have no strength-0 rows
            ef = SWAP[fr] if swap else fr
            emit(record_id, arm, cond, valence, s, fr, ef, i)

    # -- direct-add + prompt-steered (cell-level rows) ------------------------
    if include_steered:
        for fr in FRAMES:
            emit(f"cell:__baseline__:0.0:{fr}", ce.ARM_DIRECT_ADD, "__baseline__",
                 0.0, 0.0, fr, fr, None)
        for cond in EMOTIONS + SYNTH:
            spec = qualia.spec_for(cond)
            valence = float(spec.valence) if spec else 0.0
            for s in STRENGTHS:
                for fr in FRAMES:
                    emit(f"cell:{cond}:{s}:{fr}", ce.ARM_DIRECT_ADD, cond,
                         valence, s, fr, fr, None)
                    emit(f"cell:{cond}:{s}:{fr}", ce.ARM_PROMPT_STEERED, cond,
                         valence, s, fr, fr, None)

    # -- persona-control positive control (planted displacement) --------------
    for role in ROLES:
        for fr in FRAMES:
            for i in range(N_PER_CELL):
                for L in LAYERS:
                    proj = {c: round(BASE.get(c, {"self": 0.0, "other": 0.0})[fr]
                                     + CONTROL_DISP.get(c, 0.0)
                                     + float(rng.normal(0.0, NOISE)), 5)
                            for c in ("assistant_axis", "human_axis",
                                      "human_axis_purged", "valence_axis",
                                      "experiencer_axis")}
                    proj.update({f"pc{j}": 0.0 for j in range(1, 9)})
                    proj["v_injected"] = proj["v_injected_steer"] = None
                    rows.append({"record_id": f"role:{role}:{fr}:{i}",
                                 "arm": ce.ARM_PERSONA_CONTROL, "condition": role,
                                 "strength": 0.0, "frame": fr, "encode_frame": fr,
                                 "sample": i, "span": "response",
                                 "descriptive_only": False, "layer": L,
                                 "projections": proj})

    # -- header (provenance mirrors the driver payload) -----------------------
    names = ["assistant_axis", "human_axis", "human_axis_purged", "valence_axis",
             "experiencer_axis"] + [f"pc{i}" for i in range(1, 9)]
    gram = {str(L): {a: {b: (1.0 if a == b else 0.0) for b in names} for a in names}
            for L in LAYERS}
    cos_v_e = {}
    for cond in EMOTIONS + SYNTH:
        spec = qualia.spec_for(cond)
        valence = float(spec.valence) if spec else 0.0
        cos_v_e[cond] = {str(L): {n: (COS_A_PER_VALENCE * valence
                                      if n == "assistant_axis" else 0.0)
                                  for n in names} for L in LAYERS}
    conditions = [{"name": "__baseline__", "steerable": False, "valence": None,
                   "group": None}]
    for cond in EMOTIONS:
        spec = qualia.spec_for(cond)
        conditions.append({"name": cond, "steerable": True,
                           "valence": spec.valence, "group": spec.group})
    for cond in SYNTH:
        conditions.append({"name": cond, "steerable": True, "valence": None,
                           "group": None})
    for role in ROLES:
        conditions.append({"name": role, "steerable": False, "valence": None,
                           "group": None})
    return {
        "model_id": "fake/qwen3-32b", "suite_key": "qwen3-32b",
        "experiment": "persona_displacement",
        "steer_layer": STEER_LAYER, "capture_layers": LAYERS,
        "layer_roles": {"32": "text_mediated_comparator", "50": "primary"},
        "num_layers": 64, "enable_thinking": False, "calibration_norm": CALIB,
        "strengths_confirmatory": [0.0] + STRENGTHS,
        "strengths_descriptive": [0.15],
        "arms": sorted({r["arm"] for r in rows}), "seed": seed, "n_control": N_PER_CELL,
        "control_roles": {r.split(":", 1)[1]: f"You are an {r.split(':', 1)[1]}."
                          for r in ROLES},
        "sources": [{"tag": "main", "path": "/nonexistent/scored.json", "seed": 0,
                     "calibration_norm": CALIB, "steer_layer": STEER_LAYER,
                     "model_id": "fake/qwen3-32b"}],
        "n_source_records": len(sources), "n_kept_records": len(sources),
        "frames": [{"key": k, "label": k.upper(), "kind": k, "text": f"{k} q",
                    "rendered": f"{k}p", "prompt_tokens": 3} for k in FRAMES],
        "conditions": conditions, "skipped_conditions": [],
        "direction_names": names, "direction_gram": gram, "cos_v_e": cos_v_e,
        "records": rows,
    }


def _attach_planted_judge(run: ce.DisplacementRun, low_coherence_cell=None):
    """Judge scores tied to the L50 steered (z_x − z_a) projection: within-cell
    partial correlation is planted ~1. Optionally garble one cell's coherence."""
    rng = np.random.default_rng(99)
    for r in run.raw["records"]:
        if r["arm"] != ce.ARM_REENC_STEERED or r["layer"] != 50:
            continue
        if r["condition"] not in EMOTIONS:
            continue
        comp = r["projections"]["experiencer_axis"] - r["projections"]["assistant_axis"]
        coh = 5
        if low_coherence_cell and (r["condition"], r["strength"], r["frame"]) == low_coherence_cell:
            coh = 1
        run.judge[r["record_id"]] = {
            "parse_ok": True, "coherence": coh,
            "self_attribution": round(0.25 * comp + float(rng.normal(0, 0.001)), 5),
        }


def _run(pattern="A", **kw) -> ce.DisplacementRun:
    return ce.DisplacementRun(make_payload(pattern, **kw))


@pytest.fixture(scope="module")
def stats_a():
    run = _run("A")
    _attach_planted_judge(run)
    return run, ce.compute_run(run, seed=0)


@pytest.fixture(scope="module")
def stats_b():
    run = _run("B")
    return run, ce.compute_run(run, seed=0)


@pytest.fixture(scope="module")
def stats_null():
    run = _run("NULL")
    return run, ce.compute_run(run, seed=0)


# =========================================================================== #
# 1. Slope building blocks (hand-checkable magnitudes)
# =========================================================================== #
def test_emotion_slope_recovers_planted_text_slope():
    run = _run("A")
    # unsteered SELF: text component only => slope = 6*valence at L50
    b = ce.emotion_slope(run, ce.ARM_REENC_UNSTEERED, "blissful", "self", 50,
                         "experiencer_axis")
    assert b == pytest.approx(6.0, abs=1.0)
    b_neg = ce.emotion_slope(run, ce.ARM_REENC_UNSTEERED, "terrified", "self", 50,
                             "experiencer_axis")
    assert b_neg == pytest.approx(-6.0, abs=1.0)
    # OTHER frame flat; L32 carries the 0.2 gain
    assert ce.emotion_slope(run, ce.ARM_REENC_UNSTEERED, "blissful", "other", 50,
                            "experiencer_axis") == pytest.approx(0.0, abs=1.0)
    assert ce.emotion_slope(run, ce.ARM_REENC_UNSTEERED, "blissful", "self", 32,
                            "experiencer_axis") == pytest.approx(1.2, abs=1.0)


def test_paired_state_slope_recovers_planted_state():
    run = _run("A")
    # steered − unsteered, same records: state component 4*valence (SELF encode)
    b = ce.paired_state_slope(run, ce.ARM_REENC_STEERED, ce.ARM_REENC_UNSTEERED,
                              "blissful", "self", 50, "experiencer_axis")
    assert b == pytest.approx(4.0, abs=1.0)
    # swap arms: SELF-source text encoded in OTHER => no state (encode != self)
    b_swap = ce.paired_state_slope(run, ce.ARM_REENC_STEERED_SWAP,
                                   ce.ARM_REENC_UNSTEERED_SWAP,
                                   "blissful", "self", 50, "experiencer_axis")
    assert b_swap == pytest.approx(0.0, abs=1.0)


def test_baseline_rows_fallback_map():
    run = _run("A")
    # steered arm has no s=0 rows; its baselines come from the unsteered arm
    assert run.rows(ce.ARM_REENC_STEERED, "__baseline__", 0.0, "self", 50) == []
    assert len(run.baseline_rows(ce.ARM_REENC_STEERED, "self", 50)) == N_PER_CELL
    # prompt_steered shares direct_add's s=0 prompt cells
    assert len(run.baseline_rows(ce.ARM_PROMPT_STEERED, "self", 50)) == 1


# =========================================================================== #
# 2. A-pattern: T1 at floor, T3 > 0 (the A-vs-B pre-commitment), T2 null
# =========================================================================== #
def test_a_pattern_t1_experiencer_moves(stats_a):
    _, st = stats_a
    t1 = st["per_layer"]["50"]["t1"]
    assert t1["observed"] == pytest.approx(10.0, abs=1.5)
    assert t1["p_perm"] <= FLOOR + 1e-12          # exact floor 2/252
    assert t1["p_perm_exact"] and t1["n_arrangements"] == 252
    assert t1["n_emotions"] == 10


def test_a_pattern_t2_assistant_flat(stats_a):
    _, st = stats_a
    t2 = st["per_layer"]["50"]["t2"]
    assert abs(t2["observed"]) < 1.0
    assert t2["p_perm"] > 0.05                    # off the floor
    assert t2["inside_control_band"] is True


def test_a_pattern_t3_positive_signed_discriminator(stats_a):
    _, st = stats_a
    t3 = st["per_layer"]["50"]["t3"]
    assert t3["observed"] == pytest.approx(10.0, abs=1.5)
    assert t3["observed"] > 0 and t3["p_perm"] <= FLOOR + 1e-12
    # the E5 pre-commitment is stated in the output, sign matching T3
    pre = st["e5_precommitment"]
    assert pre["t3_sign"] == "A(+)"
    assert "E5" in pre["note"] and "cancellation" in pre["note"]
    assert st["decision"]["verdict"] == "A-supported"
    assert "flat" in st["decision"]["e5_prediction"]


def test_a_pattern_t4_geometric_keystone(stats_a):
    _, st = stats_a
    t4 = st["per_layer"]["50"]["t4"]
    assert t4["self_slope"] > t4["other_slope"]
    assert t4["contrast"] == pytest.approx(10.0, abs=1.5)
    assert t4["p_perm"] <= FLOOR + 1e-12 and t4["n_clusters"] == 10


def test_a_pattern_control_units_scale(stats_a):
    _, st = stats_a
    fam = st["per_layer"]["50"]
    # persona_control x displacement planted at +2.0 => scale 2.0
    assert fam["control_scale"]["experiencer_axis"]["scale"] == pytest.approx(2.0, abs=0.15)
    assert fam["control_scale"]["assistant_axis"]["scale"] == pytest.approx(3.0, abs=0.15)
    # T1 displacement at s_max=0.1 is ~1.0 => ~0.5 detectable-persona-shift units
    t1 = fam["t1"]
    assert t1["displacement_at_smax"] == pytest.approx(1.0, abs=0.2)
    assert t1["in_control_units"] == pytest.approx(0.5, abs=0.12)
    assert t1["standardized_vs_control"] is not None
    assert t1["inside_control_band"] is False


# =========================================================================== #
# 3. B-pattern: axis drops, experiencer flat => T3 < 0, R5 small
# =========================================================================== #
def test_b_pattern_t2_rejects_t3_negative(stats_b):
    _, st = stats_b
    fam = st["per_layer"]["50"]
    assert fam["t2"]["observed"] == pytest.approx(8.0, abs=1.2)
    assert fam["t2"]["p_perm"] <= FLOOR + 1e-12
    assert abs(fam["t1"]["observed"]) < 1.0 and fam["t1"]["p_perm"] > 0.05
    assert fam["t3"]["observed"] == pytest.approx(-8.0, abs=1.2)
    assert fam["t3"]["observed"] < 0
    assert st["e5_precommitment"]["t3_sign"] == "B(-)"


def test_b_pattern_frame_symmetry_and_verdict(stats_b):
    _, st = stats_b
    fam = st["per_layer"]["50"]
    r5 = fam["r5_frame_symmetry"]
    assert abs(r5["ratio"]) < 1.0 / 3.0           # frame-symmetric plant
    art = fam["artifact"]
    assert art["artifact_confounded"] is False    # analytic term ~0.6 of 8 => 7.5%
    assert 0.0 < art["analytic_fraction_of_t2"] < 0.5
    assert st["decision"]["verdict"] == "B-supported"
    assert "kill" in st["decision"]["e5_prediction"]


# =========================================================================== #
# 4. Null fixture stays null
# =========================================================================== #
def test_null_everything_off_the_floor(stats_null):
    _, st = stats_null
    fam = st["per_layer"]["50"]
    for t in ("t1", "t2", "t3"):
        assert abs(fam[t]["observed"]) < 1.0
        assert fam[t]["p_perm"] > 0.05
    assert fam["t4"]["p_perm"] > 0.05
    assert fam["t1"]["inside_control_band"] is True
    assert fam["t2"]["inside_control_band"] is True
    assert st["decision"]["verdict"] == "boring"
    assert st["e5_precommitment"]["t3_p_perm"] > 0.05


# =========================================================================== #
# 5. T5 state-vs-text decomposition + swap isolation
# =========================================================================== #
def test_t5_recovers_planted_text_state_split(stats_a):
    _, st = stats_a
    t5 = st["per_layer"]["50"]["t5"]
    assert t5["coord"] == "experiencer_axis" and t5["frame"] == "self"
    assert t5["total"]["observed"] == pytest.approx(10.0, abs=1.5)
    assert t5["text"]["observed"] == pytest.approx(6.0, abs=1.0)
    assert t5["state"]["observed"] == pytest.approx(4.0, abs=1.0)
    assert t5["text"]["p_perm"] <= FLOOR + 1e-12
    assert t5["state"]["p_perm"] <= FLOOR + 1e-12
    assert abs(t5["additivity_residual"]) < 1.0


def test_t5_swap_arms_isolate_text_from_state(stats_a):
    _, st = stats_a
    sw = st["per_layer"]["50"]["t5"]["swap"]
    # text-driven displacement FOLLOWS THE TEXT across the swap (ratio ~ 1)
    assert sw["text_swap"]["observed"] == pytest.approx(6.0, abs=1.0)
    assert sw["text_transfer_ratio"] == pytest.approx(1.0, abs=0.25)
    # state-driven displacement stays with the steer/encode context (ratio ~ 0)
    assert abs(sw["state_swap"]["observed"]) < 1.0
    assert abs(sw["state_transfer_ratio"]) < 0.25


def test_t5_null_decomposition_stays_null(stats_null):
    _, st = stats_null
    t5 = st["per_layer"]["50"]["t5"]
    assert abs(t5["text"]["observed"]) < 1.0 and t5["text"]["p_perm"] > 0.05
    assert abs(t5["state"]["observed"]) < 1.0 and t5["state"]["p_perm"] > 0.05


# =========================================================================== #
# 6. T6 mediation link (judge join, gate, intent-to-treat)
# =========================================================================== #
def test_t6_partial_correlation_recovers_planted_link(stats_a):
    _, st = stats_a
    t6 = st["per_layer"]["50"]["t6"]
    assert t6 is not None and t6["coord"] == "x_minus_a"
    g = t6["gated"]
    assert g["r_partial"] > 0.9 and g["p"] < 1e-6
    assert g["spearman_rho"] > 0.9
    assert "strength" in t6["fixed_effects"]


def test_t6_gate_and_intent_to_treat():
    run = _run("A")
    _attach_planted_judge(run, low_coherence_cell=("blissful", 0.06, "self"))
    t6 = ce.t6_mediation(run, 50)
    assert t6["gated"]["n_obs"] < t6["intent_to_treat"]["n_obs"]
    assert t6["intent_to_treat"]["r_partial"] > 0.9


def test_t6_none_without_judge():
    run = _run("A")                               # no judge attached
    assert ce.t6_mediation(run, 50) is None
    assert run.attach_judge() == 0                # sources path doesn't exist


def test_attach_judge_joins_by_record_id(tmp_path):
    run = _run("A")
    scored = {"records": [{"judge": {"parse_ok": True, "coherence": 5,
                                     "self_attribution": 2}},
                          {"judge": None},
                          {"judge": {"parse_ok": True, "coherence": 4,
                                     "self_attribution": 1}}]}
    p = tmp_path / "scored.json"
    p.write_text(json.dumps(scored))
    n = run.attach_judge({"main": str(p)})
    assert n == 2
    assert run.judge["main#0"]["self_attribution"] == 2
    assert "main#1" not in run.judge and run.judge["main#2"]["coherence"] == 4


# =========================================================================== #
# 7. Arm comparison table, layer guard, gates, artifact residualization
# =========================================================================== #
def test_arm_comparison_orders_direct_prompt_text_full(stats_a):
    _, st = stats_a
    table = {r["arm"]: r for r in st["per_layer"]["50"]["arm_comparison"]}
    assert list(table) == [ce.ARM_DIRECT_ADD, ce.ARM_PROMPT_STEERED,
                           ce.ARM_REENC_UNSTEERED, ce.ARM_REENC_STEERED]
    # planted A-pattern on x(SELF): analytic ~0 < state 4 < text 6 < full 10
    t1s = [table[a]["t1_like_x_self"] for a in table]
    assert t1s[0] == pytest.approx(0.0, abs=1.0)
    assert t1s[1] == pytest.approx(4.0, abs=1.2)
    assert t1s[2] == pytest.approx(6.0, abs=1.2)
    assert t1s[3] == pytest.approx(10.0, abs=1.5)
    assert t1s[0] < t1s[1] < t1s[2] < t1s[3]


def test_l32_guard_no_collapse_at_gain_02(stats_a):
    _, st = stats_a
    guard = st["l32_text_mediated_guard"]
    assert guard["primary_layer"] == 50 and guard["comparator_layer"] == 32
    assert guard["ratios"]["t1"] == pytest.approx(0.2, abs=0.15)
    assert guard["text_mediated_collapse"] is False


def test_l32_guard_collapse_when_l32_reproduces_l50():
    run = _run("A", layer_gain={32: 1.0, 50: 1.0})
    st = ce.compute_run(run, seed=0)
    guard = st["l32_text_mediated_guard"]
    assert guard["ratios"]["t1"] == pytest.approx(1.0, abs=0.2)
    assert guard["text_mediated_collapse"] is True


def test_collinearity_gate_passes_on_orthonormal_gram(stats_a):
    _, st = stats_a
    gate = st["per_layer"]["50"]["collinearity"]
    assert gate["cos_x_a"] == 0.0 and gate["passed"] is True
    # a mis-built coordinate set fails the gate
    run = _run("A")
    run.gram["50"]["experiencer_axis"]["assistant_axis"] = 0.8
    assert ce.collinearity_gate(run, 50)["passed"] is False


def test_artifact_residualization_leaves_planted_effect(stats_a):
    _, st = stats_a
    art = st["per_layer"]["50"]["artifact"]
    # x is orthogonal to v_e in the fixture (cos=0): T1_perp == T1
    assert art["t1_perp"]["observed"] == pytest.approx(10.0, abs=1.5)
    # the tiny planted cos(v_e, a) barely moves T2_perp
    assert abs(art["t2_perp"]["observed"]) < 1.5
    assert art["t3_perp"]["observed"] > 0


def test_gatekeeping_fixed_sequence_and_bonferroni(stats_a, stats_null):
    _, st = stats_a
    gk = st["per_layer"]["50"]["gatekeeping"]
    assert gk["order"] == ["t1", "t2", "t3", "t4", "t5", "t6"]
    assert gk["passed"][0] == "t1" and gk["stopped_at"] == "t2"   # T2 is null under A
    assert gk["p_bonferroni"]["t1"] == pytest.approx(
        min(1.0, st["per_layer"]["50"]["t1"]["p_perm"] * gk["family_size"]))
    _, stn = stats_null
    gkn = stn["per_layer"]["50"]["gatekeeping"]
    assert gkn["passed"] == [] and gkn["stopped_at"] == "t1"


def test_layer_roles_annotated(stats_a):
    _, st = stats_a
    assert st["primary_layer"] == 50
    assert st["per_layer"]["50"]["role"] == "primary"
    assert st["per_layer"]["32"]["role"] == "text_mediated_comparator"
    assert st["primary_arm"] == ce.PRIMARY_ARM == "reencode_steered"


# =========================================================================== #
# 8. Baselines-only strength-0 handling + serialization + CLI
# =========================================================================== #
def test_baselines_only_payload_degrades_gracefully():
    run = _run("A", include_steered=False)        # unsteered re-encode arm only
    st = ce.compute_run(run, seed=0)
    fam = st["per_layer"]["50"]
    # no steered rows => the primary-arm slopes have only s>0-free baselines
    assert fam["t1"] is None and fam["t2"] is None and fam["t3"] is None
    assert fam["t4"] is None and fam["t6"] is None
    assert st["decision"]["verdict"] in ("indeterminate", "insufficient")
    assert st["e5_precommitment"]["t3_observed"] is None


def test_emotion_slope_needs_two_strengths():
    run = _run("A", include_steered=False)
    assert ce.emotion_slope(run, ce.ARM_REENC_STEERED, "blissful", "self", 50,
                            "experiencer_axis") is None


def test_output_json_serializable(stats_a):
    run, st = stats_a
    dumped = json.dumps(ce.compute_all([run]))    # no numpy leakage
    assert "e5_precommitment" in dumped


def test_cli_writes_generated_json(tmp_path, monkeypatch, capsys):
    p = tmp_path / "persona_displacement.json"
    p.write_text(json.dumps(make_payload("A")))
    out = tmp_path / "e3_stats_generated.json"
    monkeypatch.setattr(sys, "argv",
                        ["compute_e3_stats.py", str(p), "--out", str(out),
                         "--no-judge"])
    assert ce.main() == 0
    stats = json.loads(out.read_text())
    assert stats["models"] == ["qwen3-32b"]
    pm = stats["per_model"]["qwen3-32b"]
    assert pm["per_layer"]["50"]["t1"]["p_perm"] <= FLOOR + 1e-12
    assert "VERDICT" in capsys.readouterr().out
