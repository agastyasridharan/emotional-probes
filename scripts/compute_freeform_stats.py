"""
CPU statistics for the free-form self-attribution experiment. Reads the scored
per-model file ``qualia_freeform_scored.json`` and computes:

* KEYSTONE — the SELF-vs-OTHER valence-following interaction. Regress a valence
  outcome on ``valence*strength`` interacted with ``is_self`` over the qualia
  valenced emotions x nonzero strengths x {SELF, OTHER} frames, clustered by emotion.
  The interaction ``b3`` (SELF slope − OTHER slope) is the genuine-self-attribution
  estimand. Significance by both the shared cluster-robust SE AND a by-emotion
  sign-permutation test (trusted, given ~10 clusters).
* DiD — SELF minus OTHER (and the SCENE expressibility floor) per condition/class.
* Per-frame valence slopes (SELF / OTHER / SCENE) as a summary + manipulation check.
* Stance distribution (self_attribute / deny / deflect_topical / refuse / none) and
  coherence, both coherence-gated. Judge independence comes from the model-geometry
  read-back projection (a primary keystone measure), not a second LLM judge.

Everything reuses :mod:`compute_qualia_stats` (``linreg``, ``summarize``,
``cluster_robust_interaction``, ``_snap``, ``SUITE_SIZES``) — no stats are reimplemented.
Pure numpy/scipy, CPU, importable/testable without a GPU.

  python3 scripts/compute_freeform_stats.py --suite-root suite \
      --out scripts/qualia_freeform_stats_generated.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import compute_qualia_stats as cq   # sibling in scripts/ — reuse its stat primitives

GATE = 3                        # coherence gate: drop samples with judge coherence < 3
PRIMARY_MEASURES = ["expressed_valence", "self_attribution", "proj_valence_axis"]
N_PERM = 2000


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _condition_class(cond: dict) -> str:
    kind = cond.get("kind")
    if kind == "qualia":
        v = cond.get("valence") or 0
        return "qualia_pos" if v > 0 else "qualia_neg" if v < 0 else "qualia_neutral"
    if kind == "appraisal":
        return "appraisal"
    return cond.get("name")   # synthetic sentinels are their own class


# --------------------------------------------------------------------------- #
# Run wrapper
# --------------------------------------------------------------------------- #
class FreeformRun:
    """Parsed one-model scored free-form results, with a gated per-cell accessor."""

    def __init__(self, raw: dict, gate: int = GATE):
        self.raw = raw
        self.model_id = raw.get("model_id")
        self.suite_key = raw.get("suite_key")
        self.layer = raw.get("layer")
        self.num_layers = raw.get("num_layers")
        self.gate = gate
        self.conditions = {c["name"]: c for c in raw.get("conditions", [])}
        self.frame_keys = [f["key"] for f in raw.get("frames", [])]
        self.strengths = [float(s) for s in raw.get("strengths", [])]
        self._rec: dict = {}
        for r in raw.get("records", []):
            self._rec.setdefault(
                (r["condition"], cq._snap(r["strength"]), r["frame"]), []).append(r)

    # -- accessors --
    def size(self) -> float:
        return cq.SUITE_SIZES.get(self.suite_key, float("nan"))

    def frames(self) -> list[str]:
        return list(self.frame_keys)

    def nonzero_strengths(self) -> list[float]:
        return [s for s in self.strengths if cq._snap(s) != 0.0]

    def qualia_valenced(self) -> list[str]:
        return [n for n, c in self.conditions.items()
                if c.get("kind") == "qualia" and (c.get("valence") or 0) != 0]

    def condition_class(self, name: str) -> str:
        return _condition_class(self.conditions[name])

    def _passes(self, r: dict) -> bool:
        j = r.get("judge") or {}
        return bool(j.get("parse_ok")) and isinstance(j.get("coherence"), int) \
            and j["coherence"] >= self.gate

    @staticmethod
    def _measure(r: dict, measure: str):
        if measure in ("expressed_valence", "self_attribution", "coherence", "stance"):
            j = r.get("judge") or {}
            return j.get(measure) if j.get("parse_ok") else None
        if measure == "proj_valence_axis":
            return r.get("proj_valence_axis")
        if measure == "objective_valence":
            return (r.get("objective") or {}).get("valence")
        return None

    def values(self, cond: str, strength, frame: str, measure: str, gated: bool = True) -> list:
        out = []
        for r in self._rec.get((cond, cq._snap(strength), frame), []):
            if gated and not self._passes(r):
                continue
            v = self._measure(r, measure)
            if v is not None:
                out.append(v)
        return out


def load_freeform_run(path) -> FreeformRun:
    with open(path) as f:
        return FreeformRun(json.load(f))


# --------------------------------------------------------------------------- #
# KEYSTONE — SELF-vs-OTHER valence-following interaction
# --------------------------------------------------------------------------- #
def keystone(run: FreeformRun, measure: str = "expressed_valence",
             frames: tuple[str, str] = ("self", "other"),
             n_perm: int = N_PERM, seed: int = 0) -> dict | None:
    """Interaction ``b3`` = (frame[0] slope − frame[1] slope) of `measure` on
    valence*strength, clustered by emotion, with a by-emotion sign-permutation p."""
    hi, lo = frames
    if hi not in run.frames() or lo not in run.frames():
        return None
    qualia = run.qualia_valenced()
    val = {n: float(run.conditions[n]["valence"]) for n in qualia}
    emos = list(val)
    emo_ix, s_arr, g_arr, y = [], [], [], []
    for name in qualia:
        for s in run.nonzero_strengths():
            for fr, g in ((hi, 1.0), (lo, 0.0)):
                for yv in run.values(name, s, fr, measure):
                    emo_ix.append(emos.index(name)); s_arr.append(float(s))
                    g_arr.append(g); y.append(float(yv))
    if len(y) < 6 or len(set(emo_ix)) < 3:
        return None
    emo_ix = np.asarray(emo_ix); s_arr = np.asarray(s_arr, float)
    g_arr = np.asarray(g_arr, float); yv = np.asarray(y, float)
    signs = np.array([val[e] for e in emos], float)
    clusters = [emos[i] for i in emo_ix]

    def rows_for(sgn):
        x = sgn[emo_ix] * s_arr
        return np.column_stack([x, g_arr, x * g_arr])

    obs = cq.cluster_robust_interaction(rows_for(signs).tolist(), y, clusters)
    if obs is None:
        return None

    def b3(sgn):
        A = np.column_stack([np.ones(len(yv)), rows_for(sgn)])
        beta = np.linalg.pinv(A.T @ A) @ A.T @ yv
        return float(beta[3])

    obs_b3 = obs["contrast"]
    rng = np.random.default_rng(seed)
    count = sum(1 for _ in range(n_perm)
                if abs(b3(rng.permutation(signs))) >= abs(obs_b3) - 1e-12)
    return {
        "measure": measure, "frames": [hi, lo],
        f"{hi}_slope": obs["interacted_slope"], f"{lo}_slope": obs["base_slope"],
        "contrast": obs["contrast"], "n_obs": obs["n_obs"], "n_clusters": obs["n_clusters"],
        "t_cluster": obs["t_cluster"], "p_cluster": obs["p_cluster"],
        "p_perm": round((1 + count) / (n_perm + 1), 5),
    }


def frame_valence_slope(run: FreeformRun, frame: str, measure: str) -> dict:
    """OLS slope of `measure` on valence*strength within one frame (a summary/floor)."""
    xs, ys = [], []
    for name in run.qualia_valenced():
        v = float(run.conditions[name]["valence"])
        for s in run.nonzero_strengths():
            for y in run.values(name, s, frame, measure):
                xs.append(v * float(s)); ys.append(float(y))
    return cq.linreg(xs, ys)


# --------------------------------------------------------------------------- #
# DiD (SELF − reference) + per-class aggregation
# --------------------------------------------------------------------------- #
def did(run: FreeformRun, measure: str = "expressed_valence", ref: str = "other") -> dict:
    per_cell = []
    for name in run.conditions:
        cls = run.condition_class(name)
        for s in run.nonzero_strengths():
            sm = _mean(run.values(name, s, "self", measure))
            rm = _mean(run.values(name, s, ref, measure))
            if sm is None or rm is None:
                continue
            per_cell.append({"condition": name, "class": cls, "strength": float(s),
                             "self": round(sm, 5), ref: round(rm, 5), "did": round(sm - rm, 5)})
    by_emotion = defaultdict(lambda: defaultdict(list))
    for r in per_cell:
        by_emotion[r["class"]][r["condition"]].append(r["did"])
    by_class = {cls: cq.summarize([m for m in (_mean(v) for v in emos.values()) if m is not None])
                for cls, emos in by_emotion.items()}
    return {"measure": measure, "ref": ref, "per_cell": per_cell, "by_class": by_class}


# --------------------------------------------------------------------------- #
# Stance distribution + coherence
# --------------------------------------------------------------------------- #
def stance_distribution(run: FreeformRun) -> list:
    agg = defaultdict(Counter)
    for name in run.conditions:
        cls = run.condition_class(name)
        for s in run.nonzero_strengths():
            for fr in run.frames():
                for st in run.values(name, s, fr, "stance"):
                    agg[(fr, cls, float(s))][st] += 1
    out = []
    for (fr, cls, s), c in agg.items():
        n = sum(c.values())
        out.append({"frame": fr, "class": cls, "strength": s, "n": n,
                    "frac": {k: round(v / n, 4) for k, v in c.items()}})
    return out


def coherence(run: FreeformRun) -> dict:
    per, overall_ok, overall_n = {}, 0, 0
    # per (frame, strength): mean coherence over parse_ok, gated fraction, n
    buckets = defaultdict(list)
    for (cond, s, fr), recs in run._rec.items():
        for r in recs:
            j = r.get("judge") or {}
            if j.get("parse_ok") and isinstance(j.get("coherence"), int):
                buckets[(fr, s)].append(j["coherence"])
    for (fr, s), cohs in buckets.items():
        n = len(cohs); ok = sum(1 for c in cohs if c >= run.gate)
        per[f"{fr}@{s:g}"] = {"n": n, "mean_coherence": round(float(np.mean(cohs)), 3),
                              "gated_fraction": round(ok / n, 4)}
        overall_ok += ok; overall_n += n
    return {"per_frame_strength": per,
            "overall_gated_fraction": round(overall_ok / overall_n, 4) if overall_n else None}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def compute_run(run: FreeformRun) -> dict:
    ks = {m: keystone(run, m) for m in PRIMARY_MEASURES}
    ks_scene = {m: keystone(run, m, frames=("self", "scene")) for m in PRIMARY_MEASURES}
    slopes = {m: {fr: frame_valence_slope(run, fr, m) for fr in run.frames()}
              for m in ("expressed_valence", "objective_valence", "proj_valence_axis")}
    return {
        "model_id": run.model_id, "suite_key": run.suite_key, "size": run.size(),
        "layer": run.layer, "num_layers": run.num_layers,
        "n_records": sum(len(v) for v in run._rec.values()),
        "keystone_self_vs_other": ks,
        "keystone_self_vs_scene": ks_scene,
        "frame_slopes": slopes,
        "did_self_vs_other": {m: did(run, m, "other") for m in ("expressed_valence", "proj_valence_axis")},
        "did_self_vs_scene": {m: did(run, m, "scene") for m in ("expressed_valence", "proj_valence_axis")},
        "stance_distribution": stance_distribution(run),
        "coherence": coherence(run),
    }


def compute_all(runs) -> dict:
    per_model = {}
    for r in runs:
        key = r.suite_key or r.model_id or "?"
        per_model[key] = compute_run(r)
    return {
        "unit_note": "keystone contrast = SELF valence-slope − OTHER valence-slope, "
                     "clustered by emotion; p_perm = by-emotion sign permutation.",
        "gate": GATE, "n_perm": N_PERM,
        "models": list(per_model), "per_model": per_model,
    }


def discover_paths(suite_root=None) -> list[str]:
    root = suite_root or os.environ.get("EMOTION_PROBES_SUITE_ROOT") or "suite"
    return sorted(glob.glob(os.path.join(root, "*", "analysis", "qualia_freeform_scored.json")))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="explicit qualia_freeform_scored.json files")
    ap.add_argument("--suite-root", default=None)
    ap.add_argument("--out", default="scripts/qualia_freeform_stats_generated.json")
    args = ap.parse_args()
    paths = args.paths or discover_paths(args.suite_root)
    if not paths:
        raise SystemExit("no qualia_freeform_scored.json found (pass paths or --suite-root).")
    runs = [load_freeform_run(p) for p in paths]
    stats = compute_all(runs)
    Path(args.out).write_text(json.dumps(stats, indent=2, allow_nan=True))
    print(f"wrote {args.out}  ({len(runs)} model(s))")
    for key in stats["models"]:
        pm = stats["per_model"][key]
        print(f"\n=== {key}  (layer {pm['layer']}/{pm['num_layers']}, n={pm['n_records']}) ===")
        for m in PRIMARY_MEASURES:
            k = pm["keystone_self_vs_other"].get(m)
            if k:
                print(f"  keystone[{m:18}] SELF−OTHER slope={k['contrast']:+.4f}  "
                      f"(self={k[f'self_slope']:+.3f} other={k[f'other_slope']:+.3f})  "
                      f"p_cluster={cq_fmt(k['p_cluster'])} p_perm={k['p_perm']:.4f}  "
                      f"[{k['n_clusters']} clusters]")
    return 0


def cq_fmt(p):
    return "n/a" if p is None else (f"{p:.2g}")


if __name__ == "__main__":
    raise SystemExit(main())
