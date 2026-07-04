#!/usr/bin/env python3
"""Merge the two valence-split E5 halves into one ``axis_mediation.json``.

The full E5 run was split across two GPUs by emotion (positive emotions with
``--tag _pos``, negative with ``--tag _neg``) — a clean disjoint partition of
the by-emotion clusters. The emotion-independent control conditions
(``__none__``, ``__neg_axis__``) ran in BOTH halves; the merge keeps the _pos
copy and drops the _neg duplicates (recorded in provenance).

Safety: the two halves are only the same experiment if the shared context is
identical — same tau (cap thresholds), same fixed audit text, same calibration
norm, same grids. Scalars/strings must match exactly; float structures must
match to ``--float-tol`` (default 1e-6 relative; both halves compute tau from
the same seeded unsteered baseline, so any real difference means the halves
diverged and the merge REFUSES).

Usage:
    python3 scripts/merge_e5_halves.py \
        suite/qwen3-32b/analysis/axis_mediation_pos.json \
        suite/qwen3-32b/analysis/axis_mediation_neg.json \
        --out suite/qwen3-32b/analysis/axis_mediation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CONTROL_CONDITIONS = {"__none__", "__neg_axis__"}
RECORD_LISTS = ("battery_records", "freeform_records", "audit_records")
# shared context that must be identical between halves
MUST_MATCH_EXACT = ("model_id", "suite_key", "experiment", "layer", "num_layers",
                    "enable_thinking", "yes_token_id", "no_token_id",
                    "yes_token_str", "no_token_str", "cap_layers",
                    "cap_percentile", "capping_config", "axis_source",
                    "audit_text", "audit_frame", "n_samples", "max_new_tokens",
                    "temperature", "seed", "strengths", "lambda_grid",
                    "interventions", "frames", "questions")
MUST_MATCH_FLOAT = ("calibration_norm", "tau_by_layer", "baseline_projection",
                    "baseline_profile")


def _float_close(a, b, tol):
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_float_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_float_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        scale = max(abs(float(a)), abs(float(b)), 1.0)
        return abs(float(a) - float(b)) <= tol * scale
    return a == b


def merge(pos_path: str, neg_path: str, out_path: str, float_tol: float) -> dict:
    pos = json.loads(Path(pos_path).read_text())
    neg = json.loads(Path(neg_path).read_text())

    errors = []
    for k in MUST_MATCH_EXACT:
        if pos.get(k) != neg.get(k):
            errors.append(f"exact-match key differs between halves: {k!r}")
    for k in MUST_MATCH_FLOAT:
        if not _float_close(pos.get(k), neg.get(k), float_tol):
            errors.append(f"float key differs beyond tol={float_tol}: {k!r} "
                          f"(halves diverged — tau/baseline must be identical)")
    if errors:
        for e in errors:
            print(f"MERGE REFUSED: {e}", file=sys.stderr)
        raise SystemExit(1)

    out = dict(pos)  # _pos supplies the shared context and the control copies

    # conditions: pos's own + controls from pos, plus neg's emotions
    pos_conds = {c["name"]: c for c in pos.get("conditions", [])}
    neg_only = [c for c in neg.get("conditions", [])
                if c["name"] not in pos_conds]
    out["conditions"] = list(pos.get("conditions", [])) + neg_only

    dropped = {}
    for key in RECORD_LISTS:
        keep_neg = [r for r in neg.get(key, [])
                    if r.get("condition") not in CONTROL_CONDITIONS]
        dropped[key] = len(neg.get(key, [])) - len(keep_neg)
        out[key] = list(pos.get(key, [])) + keep_neg

    # per-condition Δ_emo rows: serialized as a LIST of {condition, strength,
    # delta} (the halves' emotion panels are disjoint; controls carry no rows).
    # A dict-shaped union here previously dropped the neg half's rows entirely,
    # making its emotions unauditable downstream — hence the closing assert.
    pos_cells = list(pos.get("delta_by_cell", []))
    seen_cells = {(c["condition"], c["strength"]) for c in pos_cells}
    out["delta_by_cell"] = pos_cells + [
        c for c in neg.get("delta_by_cell", [])
        if (c["condition"], c["strength"]) not in seen_cells]
    # panel_mean_delta is runtime provenance: the kept control cells (from
    # _pos) were generated with the POS panel's pooled Δ_emo — keep pos's value
    # rather than fabricating a 10-emotion pool no run ever used.
    out["panel_mean_delta"] = pos.get("panel_mean_delta")

    # regression guard: every merged qualia condition must keep its Δ_emo rows
    delta_conds = {c["condition"] for c in out["delta_by_cell"]}
    qualia_conds = {c["name"] for c in out["conditions"]
                    if c.get("kind") == "qualia"}
    missing = sorted(qualia_conds - delta_conds)
    if missing:
        raise SystemExit(f"MERGE BUG: delta_by_cell lost conditions {missing}")

    out["merge_provenance"] = {
        "panel_mean_delta_source": "pos half (its controls are the kept copies)",
        "halves": {"pos": str(pos_path), "neg": str(neg_path)},
        "control_conditions_deduped": sorted(CONTROL_CONDITIONS),
        "neg_control_records_dropped": dropped,
        "float_tol": float_tol,
    }
    Path(out_path).write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pos"), ap.add_argument("neg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--float-tol", type=float, default=1e-6)
    a = ap.parse_args()
    out = merge(a.pos, a.neg, a.out, a.float_tol)
    conds = [c["name"] for c in out["conditions"]]
    print(f"wrote {a.out}")
    print(f"  conditions ({len(conds)}): {conds}")
    for key in RECORD_LISTS:
        print(f"  {key}: {len(out[key])} "
              f"(neg control dupes dropped: {out['merge_provenance']['neg_control_records_dropped'][key]})")


if __name__ == "__main__":
    main()
