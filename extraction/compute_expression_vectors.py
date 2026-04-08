"""
Compute emotion vectors at all 42 layers from the raw activation data.

Reads the 83GB of per-story activations and produces a single 74MB file
containing normalised emotion vectors for all 171 emotions at all 42 layers.

Usage:
    python compute_vectors.py
"""

import torch
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

ACTIVATIONS_DIR = Path("/mnt/ssd/data/activations_all_layers")
NEUTRAL_ACTIVATIONS_PATH = Path("/mnt/ssd/data/neutral_activations.pt")
OUTPUT_PATH = Path.home() / "data" / "emotion_vectors_all_layers.pt"
VARIANCE_THRESHOLD = 0.5  # project out top PCs until cumulative variance ≥ this

layer_emotion_means = defaultdict(dict)

for f in tqdm(sorted(ACTIVATIONS_DIR.glob("*.pt")), desc="Loading emotions"):
    emotion = f.stem
    data = torch.load(f, weights_only=True)
    for layer_idx in data:
        layer_emotion_means[layer_idx][emotion] = data[layer_idx].mean(dim=0)
    del data

# Load neutral activations for PCA confound removal.
print(f"Loading neutral activations from {NEUTRAL_ACTIVATIONS_PATH}...")
neutral_acts = torch.load(NEUTRAL_ACTIVATIONS_PATH, weights_only=True)  # (N, 42, 2560)
print(f"  shape {tuple(neutral_acts.shape)}")


def top_pcs_for_variance(X, threshold):
    """Return top PCs of X (rows = samples) covering at least `threshold` fraction
    of total variance. Returned matrix has shape (K, D) with orthonormal rows."""
    X_centered = X - X.mean(dim=0, keepdim=True)
    # SVD is more numerically stable than covariance eigendecomposition.
    _, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum()
    k = int((cum < threshold).sum().item()) + 1
    return Vt[:k]  # (k, D)


# Compute normalised, confound-stripped emotion vectors per layer
print("Computing normalised vectors with PCA confound removal...")
emotion_vectors = {}
global_means = {}
pcs_per_layer = {}

for layer_idx in range(42):
    means = layer_emotion_means[layer_idx]
    global_mean = torch.stack(list(means.values())).mean(dim=0)
    global_means[layer_idx] = global_mean

    # Top PCs from neutral activations at this layer.
    layer_neutral = neutral_acts[:, layer_idx, :].float()
    pcs = top_pcs_for_variance(layer_neutral, VARIANCE_THRESHOLD)
    pcs_per_layer[layer_idx] = pcs

    vectors = {}
    for emotion, mean in means.items():
        vec = mean - global_mean
        # Project out the PC subspace: v - V^T (V v)
        vec = vec - pcs.T @ (pcs @ vec)
        vectors[emotion] = vec / vec.norm()
    emotion_vectors[layer_idx] = vectors

    if layer_idx % 5 == 0:
        print(f"  layer {layer_idx}: stripped {pcs.shape[0]} PCs")

torch.save(
    {"vectors": emotion_vectors, "global_means": global_means, "pcs": pcs_per_layer},
    OUTPUT_PATH,
)
size_mb = OUTPUT_PATH.stat().st_size / 1e6
print(f"Saved to {OUTPUT_PATH} ({size_mb:.1f} MB)")
print(f"{len(emotion_vectors[0])} emotions x {len(emotion_vectors)} layers")
