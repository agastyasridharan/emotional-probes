#!/usr/bin/env python3
"""
E1 geometry atlas — CPU driver (portfolio.md §E1, docs/prereg/E1.md).

Builds the per-layer direction atlas (assistant axis a, matched human sub-axis h,
emotion-purged h_purged, valence axis, 10 suite emotion directions), the 8-PC
persona basis under both centerings, the full |cos| Gram tables, exact 252-
arrangement valence-sign permutation tests, the isotropic floor, the trait
comparability gate (CPU half), and — when the per-story residual cache exists —
the two decision-grade story nulls. Writes:

* ``suite/qwen3-32b/analysis/e1_atlas.json`` (+ ``e1_atlas_directions.npz``
  sidecar with the raw direction/PC arrays for E3/E5), and
* ``docs/research/e1_atlas_summary.md`` — the human-readable headline table.

E1 is a geometry gate. It does NOT adjudicate mechanism A vs B.

Run on a laptop (assets are local, no GPU)::

    python3 scripts/run_e1_atlas.py --layers 32 42 50 16
    python3 scripts/run_e1_atlas.py --story-acts /data/.../story_residuals
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import exact_perm as ep
import emotion_probes.persona.assets as pa
import emotion_probes.persona.geometry as geo
from emotion_probes.alignment.persona_displacement import (
    experiencer_axis as pd_experiencer_axis,
)
from emotion_probes.persona import (
    anchor_check,
    load_assistant_axis,
    load_emotion_bank,
    load_role_subsets,
    unit,
)

DEFAULT_LAYERS = [32, 42, 50, 16]           # headline 42; 32/50 secondary; 16 exploratory
HEADLINE_ORDER = [42, 32, 50, 16]
# E3's frozen-coordinate loader (persona_displacement.load_directions_npz) reads
# "<name>@L<layer>" keys with these names; the sidecar carries both key styles.
# The sidecar additionally carries "experiencer_axis@L<layer>" (z_x, built by
# persona_displacement.experiencer_axis) so it satisfies E3's REQUIRED_DIRECTIONS.
E3_NAME_MAP = {"a": "assistant_axis", "h": "human_axis",
               "h_purged": "human_axis_purged", "valence": "valence_axis"}
COMPARABILITY_TRAITS = ("empathetic", "kind", "skeptical")   # portfolio §E1 gate
DEFAULT_OUT = REPO_ROOT / "suite" / "qwen3-32b" / "analysis" / "e1_atlas.json"
DEFAULT_SUMMARY = REPO_ROOT / "docs" / "research" / "e1_atlas_summary.md"
DEFAULT_TRAIT_DIR = REPO_ROOT / "persona_assets" / "assistant-axis-qwen32b" / "trait_vectors"


def _cos(u: np.ndarray, v: np.ndarray) -> float:
    """Signed cosine between two (possibly non-unit) directions."""
    return float(unit(u) @ unit(v))


# --------------------------------------------------------------------------- #
# Exact valence-sign permutation (252 arrangements at 5/5)
# --------------------------------------------------------------------------- #
def valence_sign_test(v_rows: np.ndarray, signs, target: np.ndarray,
                      degenerate_tol: float = 1e-5) -> dict:
    """Exact sign-permutation p for |cos(valence(signs), target)| where the
    valence direction is rebuilt from each of the C(10,5)=252 arrangements by the
    identical pipeline (mean of + rows minus mean of − rows, unit-normalized).
    Inference goes through exact_perm.sign_permutation_pvalue (identity counts;
    two-sided floor 2/252 ≈ 0.0079).

    DEGENERACY GUARD: every valence(signs) lies in span(v_rows). If the target is
    (numerically) orthogonal to that span — true BY CONSTRUCTION for h_purged,
    since purge is linear — the statistic is identically zero for all 252
    arrangements and any p-value would be a float-noise artifact. Such tests are
    returned with ``degenerate=True`` and ``p_perm=None``, never a number."""
    t = unit(target)
    rows = np.asarray(v_rows, dtype=np.float32)

    def stat(s: np.ndarray) -> float:
        v = rows[s > 0].mean(axis=0) - rows[s < 0].mean(axis=0)
        nrm = float(np.linalg.norm(v))
        return float(v @ t) / nrm if nrm > 0 else 0.0

    span_proj = float(np.linalg.norm(geo.span_basis(rows) @ t))
    if span_proj < degenerate_tol:
        return {"observed_cos": float(stat(np.asarray(signs, dtype=float))),
                "p_perm": None, "degenerate": True,
                "target_span_projection": span_proj,
                "note": ("target orthogonal to the emotion span by construction; "
                         "the statistic is identically zero for every arrangement "
                         "and the pre-registered sign prediction is unfalsifiable "
                         "under this construction")}

    signs = np.asarray(signs, dtype=float)
    res = ep.sign_permutation_pvalue(signs, stat)
    # Two-sided attainable floor: with a balanced sign multiset the mirrored
    # arrangement always ties |observed| (stat is exactly negated), so exact p can
    # never go below 2/n_arrangements — the prereg's "two-sided exact floor
    # 2/252 ≈ 0.0079" (docs/prereg/E1.md §Inference). exact_perm's generic floor
    # (1/n) counts only the identity and would under-report the attainable floor.
    floor = float(res["floor"])
    if res["exact"] and int((signs > 0).sum()) == int((signs < 0).sum()):
        floor = 2.0 / int(res["n_arrangements"])
    return {"observed_cos": float(res["observed"]),
            "p_perm": float(res["p_perm"]),
            "exact": bool(res["exact"]),
            "n_arrangements": int(res["n_arrangements"]),
            "floor": floor,
            "degenerate": False,
            "target_span_projection": span_proj}


# --------------------------------------------------------------------------- #
# Axis reconstruction + trait comparability (CPU half of the gate)
# --------------------------------------------------------------------------- #
def axis_reconstruction(ax, layers) -> dict:
    """cos(default − mean(roles), shipped axis) per layer — the E1 'verify by
    reconstructing' check (bf16 shipping precision => expect ≈ 1 − O(1e-3))."""
    role_mean = np.mean(list(ax.role_vectors.values()), axis=0)
    recon = ax.default_vector - role_mean
    return {str(int(l)): _cos(recon[l], ax.axis[l]) for l in layers}


def trait_comparability(bank, trait_dir, layers, traits=COMPARABILITY_TRAITS) -> dict:
    """CPU half of the comparability gate: cos(our bank direction, Lu's matching
    trait vector) per layer vs the matched null (|cos| of our direction against
    every OTHER Lu trait at the same layer). The register-direction partial-out
    half needs neutral-story acts and is gated on the story cache."""
    if trait_dir is None or not Path(trait_dir).is_dir():
        return {"status": f"skipped: trait dir {trait_dir} not found"}
    import torch  # local-only dependency of this block

    loaded = {}
    for f in sorted(Path(trait_dir).glob("*.pt")):
        try:
            t = torch.load(f, map_location="cpu")
            if isinstance(t, dict):
                t = t.get("vector", t.get("axis"))
            loaded[f.stem] = t.float().numpy().astype(np.float32)
        except Exception as exc:  # corrupted shipping file: record, keep going
            loaded[f.stem] = None
            print(f"  [trait_comparability] skipping {f.name}: {exc}")
    usable = {k: v for k, v in loaded.items() if v is not None}

    out = {"trait_dir": str(trait_dir), "n_trait_vectors": len(usable), "traits": {},
           "note": ("matched null = |cos| vs every other Lu trait at the same layer; "
                    "register-direction partial-out is gated on the story cache")}
    for trait in traits:
        if trait not in usable:
            out["traits"][trait] = {"status": "no matching Lu trait vector shipped"}
            continue
        try:
            per_layer = {}
            for layer in layers:
                ours = unit(bank.vector(trait, layer))
                lu = unit(usable[trait][layer])
                obs = abs(float(ours @ lu))
                null = np.array([abs(float(ours @ unit(v[layer])))
                                 for k, v in usable.items() if k != trait])
                per_layer[str(int(layer))] = {
                    "abs_cos": obs,
                    "null_p95": float(np.percentile(null, 95)),
                    "null_max": float(null.max()),
                    "beats_p95": bool(obs > np.percentile(null, 95)),
                }
            out["traits"][trait] = per_layer
        except KeyError:
            out["traits"][trait] = {"status": "not in the 171-emotion bank"}
    return out


# --------------------------------------------------------------------------- #
# AI-role jackknife (the human sub-axis rests on very few ai_artificial roles)
# --------------------------------------------------------------------------- #
def _jackknife_se(thetas: np.ndarray) -> float:
    n = len(thetas)
    return float(np.sqrt((n - 1) / n * float(((thetas - thetas.mean()) ** 2).sum())))


def ai_role_jackknife(ax, subsets, directions_by_layer: dict) -> dict:
    """Leave-one-AI-role-out SE for the h statistics, per layer.

    The frozen atlas has 232 human vs only 7 ai_artificial roles; h's variance is
    dominated by that n = 7 side. Reported per layer: jackknife SE of
    cos(valence, h) and cos(h, a), plus the direction stability
    min/mean cos(h_loo, h_full). These SEs are honesty bounds on every raw-h
    number in the atlas (h_purged inherits them)."""
    h_loo = pa.human_axis_jackknife(ax, subsets)          # {dropped: (64, H)}
    h_full = pa.human_axis(ax, subsets)
    out = {"n_ai_roles": len(h_loo),                      # one replicate per role
           "dropped_roles": sorted(h_loo), "layers": {}}
    for layer, dirs in sorted(directions_by_layer.items()):
        hf = unit(h_full[layer])
        cos_vh, cos_ha, stab = [], [], []
        for name in sorted(h_loo):
            hj = unit(h_loo[name][layer])
            cos_vh.append(_cos(dirs["valence"], hj))
            cos_ha.append(_cos(hj, dirs["a"]))
            stab.append(float(hj @ hf))
        out["layers"][str(int(layer))] = {
            "se_cos_valence_h": _jackknife_se(np.asarray(cos_vh)),
            "se_cos_h_a": _jackknife_se(np.asarray(cos_ha)),
            "loo_cos_valence_h": [float(v) for v in cos_vh],
            "loo_cos_h_a": [float(v) for v in cos_ha],
            "min_cos_h_loo_h_full": float(min(stab)),
            "mean_cos_h_loo_h_full": float(np.mean(stab)),
        }
    return out


# --------------------------------------------------------------------------- #
# Atlas build
# --------------------------------------------------------------------------- #
def build_atlas(bank, ax, subsets, layers, story_acts_dir=None, n_perm=10000,
                seed=0, floor_n=2000, trait_dir=None) -> tuple[dict, dict]:
    """Assemble the full atlas. Returns (json_payload, sidecar_arrays) where
    sidecar_arrays maps npz keys to the raw direction / PC-basis arrays that
    E3/E5 consume as frozen coordinates."""
    layers = [int(l) for l in layers]
    pos, neg = geo.suite_valence_panel()
    names = [*pos, *neg]
    signs = [1.0] * len(pos) + [-1.0] * len(neg)

    payload = {
        "experiment": "e1_geometry_atlas",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": pa.SOURCE_MODEL_ID,
        "n_layers": int(ax.axis.shape[0]),
        "hidden": int(ax.axis.shape[1]),
        "n_roles": len(ax.role_vectors),
        "layers_requested": layers,
        "panel": {"positive": pos, "negative": neg},
        "seed": int(seed),
        "anchor_check": anchor_check(ax, layers=tuple(layers)),
        "axis_reconstruction": axis_reconstruction(ax, layers),
        "isotropic_floor": geo.isotropic_floor(dim=int(ax.axis.shape[1]),
                                               n=floor_n, seed=seed),
        "layers": {},
        "notes": [
            "E1 is a geometry gate; it does NOT adjudicate mechanism A vs B.",
            "Only cos(valence, h_purged) is B-relevant; raw cos(valence, h) is descriptive.",
            "Pre-registered signs: B predicts cos(valence, a) < 0 and cos(valence, h_purged) > 0.",
            "The isotropic floor is a floor, never a decision criterion.",
            "DEGENERACY (flagged for prereg amendment): valence lies in span(v_1..v_10) "
            "and purge is linear, so h_purged is orthogonal to that span BY CONSTRUCTION "
            "— cos(valence, h_purged) is identically 0 and its pre-registered sign "
            "prediction is unfalsifiable as written. The informative purged quantities "
            "are cos(h_purged, a), cos(h, h_purged) (contamination measurement), and "
            "the story-cache occupancy/null tests.",
        ],
    }

    directions_by_layer, arrays = {}, {}
    for layer in layers:
        dirs = geo.build_directions(bank, ax, subsets, layer, seed=seed)
        directions_by_layer[layer] = dirs
        for dname, v in dirs.items():
            arrays[f"L{layer}__{dname}"] = v
            # E3-consumable duplicate: persona_displacement.load_directions_npz
            # reads frozen coordinates as "<name>@L<layer>" with its long names.
            arrays[f"{E3_NAME_MAP.get(dname, dname)}@L{layer}"] = v
        # z_x — E3's run() REFUSES direction sets missing any REQUIRED_DIRECTIONS
        # (which includes experiencer_axis), so the sidecar must carry it for the
        # ``--directions`` path to work. Built by the canonical prereg-E3 function
        # (single source of the definition; identical to frozen_directions()'s z_x).
        arrays[f"experiencer_axis@L{layer}"] = pd_experiencer_axis(ax, subsets, layer)

        per_emotion = {e: {t: _cos(dirs[f"v_{e}"], dirs[t])
                           for t in geo.DIRECTION_TARGETS}
                       for e in names}
        abs_a = [abs(per_emotion[e]["a"]) for e in names]
        abs_h = [abs(per_emotion[e]["h"]) for e in names]

        pca = {}
        for center in ("default", "mean"):
            block = geo.pca_atlas(ax, layer, center=center)
            comps = block.pop("components")
            arrays[f"L{layer}__pca_{center}_components"] = comps
            if center == "default":              # E3 uses default-centered PCs
                for i in range(comps.shape[0]):
                    arrays[f"pc{i + 1}@L{layer}"] = comps[i]
            pca[center] = block

        payload["layers"][str(layer)] = {
            "signed": {
                "cos_valence_a": _cos(dirs["valence"], dirs["a"]),
                "cos_valence_h": _cos(dirs["valence"], dirs["h"]),
                "cos_valence_h_purged": _cos(dirs["valence"], dirs["h_purged"]),
                "cos_h_a": _cos(dirs["h"], dirs["a"]),
                "cos_h_purged_a": _cos(dirs["h_purged"], dirs["a"]),
                "cos_h_h_purged": _cos(dirs["h"], dirs["h_purged"]),
            },
            "per_emotion": per_emotion,
            "emotion_summary": {
                "mean_abs_cos_v_a": float(np.mean(abs_a)),
                "max_abs_cos_v_a": float(np.max(abs_a)),
                "mean_abs_cos_v_h": float(np.mean(abs_h)),
                "max_abs_cos_v_h": float(np.max(abs_h)),
            },
            "cosine_table": geo.cosine_table(dirs),
            "pca": pca,
            "valence_sign_perm": {
                t: valence_sign_test(np.stack([dirs[f"v_{e}"] for e in names]),
                                     signs, dirs[t])
                for t in geo.DIRECTION_TARGETS
            },
        }

    payload["human_axis_jackknife"] = ai_role_jackknife(ax, subsets,
                                                        directions_by_layer)

    story = geo.story_nulls(story_acts_dir, directions_by_layer,
                            n_perm=n_perm, seed=seed)
    payload["story_nulls"] = story if story is not None else {
        "status": "gated on story cache",
        "detail": ("per-story residual cache (scripts/cache_story_residuals.py) "
                   "not found — the two decision-grade nulls and the B-substrate "
                   "verdict cannot be issued until it runs"),
        "requested_dir": str(story_acts_dir) if story_acts_dir else None,
    }
    payload["trait_comparability"] = trait_comparability(bank, trait_dir, layers)
    return payload, arrays


# --------------------------------------------------------------------------- #
# Summary rendering
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def _fmt_p(perm: dict) -> str:
    if perm.get("degenerate"):
        return "— (degenerate)"
    return f"{perm['p_perm']:.4f}"


def render_summary(payload: dict) -> str:
    """Human-readable headline table (markdown) against the isotropic floor."""
    layers = [l for l in HEADLINE_ORDER if str(l) in payload["layers"]]
    layers += [int(l) for l in payload["layers"] if int(l) not in layers]
    iso = payload["isotropic_floor"]

    rows = [
        ("cos(valence, a) [signed]", lambda b: _fmt(b["signed"]["cos_valence_a"])),
        ("cos(valence, h) [signed, descriptive]",
         lambda b: _fmt(b["signed"]["cos_valence_h"])),
        ("cos(valence, h_purged) [signed, B-relevant]",
         lambda b: _fmt(b["signed"]["cos_valence_h_purged"])),
        (r"\|cos(h, a)\|", lambda b: f"{abs(b['signed']['cos_h_a']):.4f}"),
        (r"\|cos(h_purged, a)\|", lambda b: f"{abs(b['signed']['cos_h_purged_a']):.4f}"),
        (r"mean \|cos(v_e, a)\| (10 emotions)",
         lambda b: f"{b['emotion_summary']['mean_abs_cos_v_a']:.4f}"),
        (r"max \|cos(v_e, a)\|",
         lambda b: f"{b['emotion_summary']['max_abs_cos_v_a']:.4f}"),
        (r"mean \|cos(v_e, h)\|",
         lambda b: f"{b['emotion_summary']['mean_abs_cos_v_h']:.4f}"),
        (r"\|cos(PC1, a)\| (default-centered)",
         lambda b: f"{b['pca']['default']['cos_axis']['pc1']:.4f}"),
        (r"\|cos(PC1, a)\| (mean-centered)",
         lambda b: f"{b['pca']['mean']['cos_axis']['pc1']:.4f}"),
        ("PC1 explained variance (mean-centered)",
         lambda b: f"{b['pca']['mean']['explained_variance_ratio'][0]:.4f}"),
        (r"exact p: \|cos(valence, a)\| (252 arr.)",
         lambda b: _fmt_p(b["valence_sign_perm"]["a"])),
        (r"exact p: \|cos(valence, h)\|",
         lambda b: _fmt_p(b["valence_sign_perm"]["h"])),
        (r"exact p: \|cos(valence, h_purged)\|",
         lambda b: _fmt_p(b["valence_sign_perm"]["h_purged"])),
    ]

    lines = [
        "# E1 geometry atlas — headline summary",
        "",
        f"Generated {payload['generated']} · model {payload['model_id']} · "
        f"{payload['n_roles']} roles · hidden {payload['hidden']} · "
        f"seed {payload['seed']}",
        "",
        "**E1 is a geometry gate. It does NOT adjudicate mechanism A vs B.** "
        "Only cos(valence, h_purged) is B-relevant; raw h cosines are descriptive. "
        "Pre-registered signs: B predicts cos(valence, a) < 0 and "
        "cos(valence, h_purged) > 0.",
        "",
        "**Degeneracy flag (prereg amendment needed):** valence lies in "
        "span(v_1..v_10) and the purge is linear, so h_purged is orthogonal to "
        "that span *by construction* — cos(valence, h_purged) is identically 0 "
        "and the pre-registered sign prediction on it is unfalsifiable as "
        "written. No p-value is reported for a degenerate statistic. The "
        "informative purged quantities are |cos(h_purged, a)| and "
        "cos(h, h_purged) (the contamination measurement), plus the story-cache "
        "nulls once built.",
        "",
        f"**Isotropic floor** (floor only, never a criterion): mean |cos| = "
        f"{iso['mean_abs_cos']:.4f} (analytic {iso['analytic_mean']:.4f}), "
        f"p95 = {iso['p95']:.4f}, p99 = {iso['p99']:.4f} "
        f"(dim {iso['dim']}, n = {iso['n']}).",
        "",
        "| statistic | " + " | ".join(f"L{l}" for l in layers) + " |",
        "|---|" + "---|" * len(layers),
    ]
    for label, fn in rows:
        cells = [fn(payload["layers"][str(l)]) for l in layers]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["",
              "Layer roles: L42 headline; L32/L50 secondary (Bonferroni ×3); "
              "L16 exploratory. Exact valence-sign floor 2/252 ≈ 0.0079."]

    recon = payload["axis_reconstruction"]
    lines += ["",
              "**Axis reconstruction** cos(default − mean(roles), shipped axis): "
              + ", ".join(f"L{l} = {recon[str(l)]:.4f}" for l in layers) + "."]

    jk = payload.get("human_axis_jackknife")
    if jk:
        lines += ["",
                  f"**Human sub-axis n = {jk['n_ai_roles']} AI-role jackknife** "
                  "(honesty bound on every raw-h statistic; h rests on "
                  f"{jk['n_ai_roles']} ai_artificial roles):"]
        for l in layers:
            b = jk["layers"].get(str(l))
            if b is None:
                continue
            lines.append(
                f"- L{l}: SE[cos(valence, h)] = {b['se_cos_valence_h']:.4f}, "
                f"SE[cos(h, a)] = {b['se_cos_h_a']:.4f}, "
                f"cos(h_loo, h_full) min/mean = {b['min_cos_h_loo_h_full']:.3f}"
                f"/{b['mean_cos_h_loo_h_full']:.3f}")

    tc = payload.get("trait_comparability", {})
    if "traits" in tc:
        lines += ["", "**Trait comparability gate (CPU half)** — |cos(bank dir, "
                      "Lu trait)| vs matched null p95 over the other "
                      f"{tc['n_trait_vectors'] - 1} traits:"]
        for trait, block in tc["traits"].items():
            if "status" in block:
                lines.append(f"- {trait}: {block['status']}")
            else:
                cells = ", ".join(
                    f"L{l} {block[str(l)]['abs_cos']:.3f} vs p95 "
                    f"{block[str(l)]['null_p95']:.3f}"
                    f"{' PASS' if block[str(l)]['beats_p95'] else ' FAIL'}"
                    for l in layers if str(l) in block)
                lines.append(f"- {trait}: {cells}")
        lines.append("- register-direction partial-out: gated on story cache.")
    else:
        lines += ["", f"**Trait comparability:** {tc.get('status', 'not run')}"]

    sn = payload["story_nulls"]
    if isinstance(sn, dict) and sn.get("status") == "gated on story cache":
        lines += ["", "**Story nulls:** GATED — per-story residual cache "
                      "(`scripts/cache_story_residuals.py`) not built yet. The two "
                      "decision-grade nulls (story-label permutation, topic-contrast) "
                      "and any B-substrate verdict wait on it."]
    else:
        lines += ["", "**Story nulls** (decision-grade, "
                      f"n_perm = {sn['n_perm']}):"]
        for l, block in sn["layers"].items():
            if "status" in block:
                lines.append(f"- L{l}: {block['status']}")
                continue
            lp = block["label_permutation"]
            cells = ", ".join(f"{t}: |cos| {v['abs_observed']:.4f} p {v['p_perm']:.4g}"
                              for t, v in lp.items())
            lines.append(f"- L{l} label-permutation — {cells}")
            tcb = block["topic_contrast"]
            if "status" in tcb:
                lines.append(f"  - topic-contrast: {tcb['status']}")
            else:
                cells = ", ".join(
                    f"{t}: obs {v['observed_abs_cos']:.4f} vs p95 "
                    f"{v['control_p95_abs_cos']:.4f}"
                    f"{' BEATS' if v['beats_p95'] else ''}"
                    for t, v in tcb["targets"].items())
                lines.append(f"  - topic-contrast ({tcb['n_directions']} dirs) — {cells}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="E1 geometry atlas (CPU): direction cosines, PCA persona basis, "
                    "exact valence-sign permutations, isotropic floor, story nulls.")
    parser.add_argument("--layers", nargs="+", type=int, default=DEFAULT_LAYERS,
                        help=f"analysis layers (default: {DEFAULT_LAYERS}; "
                             "42 headline, 32/50 secondary, 16 exploratory)")
    parser.add_argument("--story-acts", default=None,
                        help="per-story residual cache dir (from "
                             "cache_story_residuals.py); omit to gate the story nulls")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="atlas JSON path (sidecar *_directions.npz next to it)")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY),
                        help="human-readable markdown summary path")
    parser.add_argument("--trait-dir", default=str(DEFAULT_TRAIT_DIR),
                        help="Lu trait_vectors dir for the comparability gate")
    parser.add_argument("--n-perm", type=int, default=10000,
                        help="Monte-Carlo draws for the story-label null")
    parser.add_argument("--floor-n", type=int, default=2000,
                        help="random pairs for the isotropic floor")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bank = load_emotion_bank()
    ax = load_assistant_axis()
    subsets = load_role_subsets()

    payload, arrays = build_atlas(
        bank, ax, subsets, layers=args.layers, story_acts_dir=args.story_acts,
        n_perm=args.n_perm, seed=args.seed, floor_n=args.floor_n,
        trait_dir=args.trait_dir)
    payload["assets"] = {"emotion_bank": bank.path, "assistant_axis": ax.path,
                         "role_subsets": str(pa.DEFAULT_ROLE_SUBSETS)}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    npz = out.with_name(out.stem + "_directions.npz")
    np.savez_compressed(npz, **arrays)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(render_summary(payload))

    print(f"wrote {out}\nwrote {npz}\nwrote {summary_path}\n")
    iso = payload["isotropic_floor"]
    print(f"isotropic floor: mean |cos| {iso['mean_abs_cos']:.4f} "
          f"(analytic {iso['analytic_mean']:.4f}), p95 {iso['p95']:.4f}")
    for layer in [l for l in HEADLINE_ORDER if str(l) in payload["layers"]]:
        b = payload["layers"][str(layer)]
        print(f"L{layer}: cos(valence,a) = {b['signed']['cos_valence_a']:+.4f} "
              f"(p {_fmt_p(b['valence_sign_perm']['a'])}), "
              f"cos(valence,h) = {b['signed']['cos_valence_h']:+.4f} "
              f"(p {_fmt_p(b['valence_sign_perm']['h'])}), "
              f"cos(valence,h_purged) = {b['signed']['cos_valence_h_purged']:+.4f} "
              f"({_fmt_p(b['valence_sign_perm']['h_purged'])}), "
              f"mean|cos(v_e,a)| = {b['emotion_summary']['mean_abs_cos_v_a']:.4f}, "
              f"mean|cos(v_e,h)| = {b['emotion_summary']['mean_abs_cos_v_h']:.4f}, "
              f"|cos(PC1,a)| = {b['pca']['mean']['cos_axis']['pc1']:.4f} (mean-ctr)")
    if payload["story_nulls"].get("status") == "gated on story cache":
        print("story nulls: GATED on story cache (cache_story_residuals.py)")


if __name__ == "__main__":
    main()
