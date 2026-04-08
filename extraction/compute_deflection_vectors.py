"""
Compute deflection (suppression) vectors from the extracted deflection activations.

Same recipe as compute_vectors.py:
  1. Load per-emotion mean activations (from deflection dialogues)
  2. Subtract global mean across emotions
  3. Project out top PCs from neutral activations (confound removal)
  4. Unit normalise

Produces vectors for:
  - target_vectors: "emotion X is being hidden" direction
  - displayed_vectors: "emotion Y is being used as a mask" direction

Usage (on the VM):
    ~/venv/bin/python3 compute_deflection_vectors.py
"""

import torch
from pathlib import Path

DEFLECTION_PATH = Path("/mnt/ssd/data/deflection_activations.pt")
NEUTRAL_PATH = Path("/mnt/ssd/data/neutral_dialogue_activations.pt")
EXPRESSION_PATH = Path.home() / "data" / "emotion_vectors_all_layers.pt"
OUTPUT_PATH = Path.home() / "data" / "deflection_vectors.pt"
VARIANCE_THRESHOLD = 0.5
EXPRESSION_VARIANCE_THRESHOLD = 0.99
NUM_LAYERS = 42


def top_pcs_for_variance(X, threshold):
    """Return top PCs of X (rows = samples) covering at least `threshold` fraction
    of total variance. Returned matrix has shape (K, D) with orthonormal rows."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum()
    k = int((cum < threshold).sum().item()) + 1
    return Vt[:k]


def build_vectors(emotion_means, pcs_per_layer, baseline_means=None):
    """Subtract global mean (or baseline), project out neutral PCs, unit normalise.

    If baseline_means is provided, each vector is computed as
    (emotion_mean - baseline_mean) for the same emotion, instead of
    (emotion_mean - global_mean). This lets us capture the scenario→dialogue
    transition directly.
    """
    vectors = {}
    global_means = {}

    for layer_idx in range(NUM_LAYERS):
        emotions_at_layer = [e for e in emotion_means if layer_idx in emotion_means[e]]
        if not emotions_at_layer:
            continue

        means = {e: emotion_means[e][layer_idx] for e in emotions_at_layer}
        global_mean = torch.stack(list(means.values())).float().mean(dim=0)
        global_means[layer_idx] = global_mean
        pcs = pcs_per_layer[layer_idx]

        layer_vectors = {}
        for emotion, mean in means.items():
            if baseline_means and emotion in baseline_means and layer_idx in baseline_means[emotion]:
                # Use scenario mean as baseline — captures the masking transition.
                vec = mean.float() - baseline_means[emotion][layer_idx].float()
            else:
                vec = mean.float() - global_mean
            vec = vec - pcs.T @ (pcs @ vec)
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
            layer_vectors[emotion] = vec
        vectors[layer_idx] = layer_vectors

    return vectors, global_means


def main():
    print(f"Loading deflection activations from {DEFLECTION_PATH}...")
    data = torch.load(DEFLECTION_PATH, weights_only=False)

    target_dialogue_means = data["target_dialogue_means"]
    target_scenario_means = data["target_scenario_means"]
    displayed_dialogue_means = data["displayed_dialogue_means"]
    pair_dialogue_means = data.get("pair_dialogue_means", {})

    print(f"  {len(target_dialogue_means)} target emotions")
    print(f"  {len(displayed_dialogue_means)} displayed emotions")
    print(f"  {len(pair_dialogue_means)} (target, displayed) pairs")

    print(f"Loading neutral activations from {NEUTRAL_PATH}...")
    neutral_acts = torch.load(NEUTRAL_PATH, weights_only=True)
    print(f"  shape {tuple(neutral_acts.shape)}")

    # Compute PCs per layer from neutral activations.
    print("Computing PCs for confound removal...")
    pcs_per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        layer_neutral = neutral_acts[:, layer_idx, :].float()
        pcs = top_pcs_for_variance(layer_neutral, VARIANCE_THRESHOLD)
        pcs_per_layer[layer_idx] = pcs
        if layer_idx % 10 == 0:
            print(f"  layer {layer_idx}: {pcs.shape[0]} PCs")

    # Load expression vectors for orthogonalisation.
    print(f"Loading expression vectors from {EXPRESSION_PATH}...")
    expr_data = torch.load(EXPRESSION_PATH, weights_only=False)
    expr_vectors = expr_data["vectors"]

    print("Computing expression-space PCs for orthogonalisation...")
    expr_pcs_per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        if layer_idx not in expr_vectors:
            continue
        expr_mat = torch.stack(list(expr_vectors[layer_idx].values())).float()
        expr_pcs = top_pcs_for_variance(expr_mat, EXPRESSION_VARIANCE_THRESHOLD)
        expr_pcs_per_layer[layer_idx] = expr_pcs
        if layer_idx % 10 == 0:
            print(f"  layer {layer_idx}: {expr_pcs.shape[0]} PCs (of {expr_mat.shape[0]} expression vectors)")

    # Build target vectors (hidden emotion direction).
    print("Building target (suppression) vectors...")
    target_vectors, target_global_means = build_vectors(target_dialogue_means, pcs_per_layer)

    # Orthogonalise target vectors against expression space.
    print("Orthogonalising target vectors against expression space...")
    for layer_idx in target_vectors:
        if layer_idx not in expr_pcs_per_layer:
            continue
        pcs = expr_pcs_per_layer[layer_idx]
        for emotion in target_vectors[layer_idx]:
            vec = target_vectors[layer_idx][emotion]
            vec = vec - pcs.T @ (pcs @ vec)
            if vec.norm() > 0:
                vec = vec / vec.norm()
            target_vectors[layer_idx][emotion] = vec

    # Build displayed vectors (mask emotion direction).
    print("Building displayed (mask) vectors...")
    displayed_vectors, displayed_global_means = build_vectors(displayed_dialogue_means, pcs_per_layer)

    # Build scenario vectors (emotion as named in preamble).
    print("Building scenario vectors...")
    scenario_vectors, scenario_global_means = build_vectors(target_scenario_means, pcs_per_layer)

    # Build pair vectors if available.
    pair_vectors = {}
    if pair_dialogue_means:
        print(f"Building pair vectors ({len(pair_dialogue_means)} pairs)...")
        pair_vectors, _ = build_vectors(pair_dialogue_means, pcs_per_layer)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "target_vectors": target_vectors,
            "target_global_means": target_global_means,
            "displayed_vectors": displayed_vectors,
            "displayed_global_means": displayed_global_means,
            "scenario_vectors": scenario_vectors,
            "scenario_global_means": scenario_global_means,
            "pair_vectors": pair_vectors,
            "pcs": pcs_per_layer,
        },
        OUTPUT_PATH,
    )

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(f"\nSaved to {OUTPUT_PATH} ({size_mb:.1f} MB)")

    # Quick sanity check: cosine similarity between target and story vectors.
    print("\nSanity check — cosine similarity between target (deflection) and story vectors:")
    try:
        story_data = torch.load(Path.home() / "data" / "emotion_vectors_all_layers.pt", weights_only=False)
        story_vectors = story_data["vectors"]
        layer = 19
        shared = set(target_vectors.get(layer, {}).keys()) & set(story_vectors.get(layer, {}).keys())
        sims = []
        for e in sorted(shared)[:10]:
            sim = torch.dot(target_vectors[layer][e], story_vectors[layer][e].float()).item()
            sims.append(sim)
            print(f"  {e:20s}: {sim:.4f}")
        if sims:
            print(f"  Mean cosine sim: {sum(sims)/len(sims):.4f} (expect low / near-zero if orthogonal)")
    except Exception as ex:
        print(f"  Could not load story vectors for comparison: {ex}")


if __name__ == "__main__":
    main()
