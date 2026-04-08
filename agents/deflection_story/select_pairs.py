"""
Select (target, displayed) emotion pairs for the deflection dataset.

For each of the 171 emotions, we:
  1. Compute cosine similarity to every other emotion (using the existing
     emotion vectors at the target layer).
  2. Take the top 50% most *dissimilar* emotions (lowest cosine similarity).
  3. Randomly sample 14 from that pool as "displayed" emotion pairings.

Rationale: when the model is masking emotion X, its displayed emotion should
be genuinely different from X, not a synonym. This prevents trivial pairings
like "hiding angry, displaying furious".

Output:
    data/deflection_pairs.json
        {target_emotion: [14 displayed_emotions], ...}

Usage:
    python select_deflection_pairs.py
"""

import json
import random
from pathlib import Path

import torch

VECTORS_PATH = Path.home() / "data" / "emotion_vectors_all_layers.pt"
TARGET_LAYER = 19
N_DISPLAYED_PER_TARGET = 14
DISSIMILAR_FRACTION = 0.25  # top quarter most dissimilar
OUTPUT_PATH = Path(__file__).parent / "data" / "deflection_pairs.json"
SEED = 42


def main():
    random.seed(SEED)

    print(f"Loading emotion vectors from {VECTORS_PATH}...")
    data = torch.load(VECTORS_PATH, weights_only=True)
    emotion_vectors = data["vectors"][TARGET_LAYER]  # {emotion: tensor(2560,)}

    emotions = sorted(emotion_vectors.keys())
    n = len(emotions)
    print(f"  {n} emotions at layer {TARGET_LAYER}")

    # Stack into (n, d) matrix. Vectors are already unit-normalized, so
    # V @ V.T is the cosine-similarity matrix.
    V = torch.stack([emotion_vectors[e] for e in emotions]).float()
    sim = V @ V.T  # (n, n)
    sim.fill_diagonal_(float("inf"))  # exclude self

    pool_size = int((n - 1) * DISSIMILAR_FRACTION)  # top half of the other 170
    print(f"  pool size per target: {pool_size} (top {DISSIMILAR_FRACTION:.0%} most dissimilar)")

    pairs = {}
    for i, target in enumerate(emotions):
        # Lowest cosine sim = most dissimilar.
        ranked = torch.argsort(sim[i]).tolist()  # ascending
        dissimilar_pool = [emotions[j] for j in ranked[:pool_size]]
        displayed = random.sample(dissimilar_pool, N_DISPLAYED_PER_TARGET)
        pairs[target] = displayed

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(pairs, indent=2))

    total_pairs = sum(len(v) for v in pairs.values())
    print(f"\nWrote {total_pairs} (target, displayed) pairs to {OUTPUT_PATH}")
    print(f"  {len(pairs)} targets × {N_DISPLAYED_PER_TARGET} displayed each")

    # Sanity check: show a few examples
    print("\nExamples:")
    for target in ["desperate", "angry", "calm", "happy", "loving"]:
        if target in pairs:
            print(f"  {target:15s} → {', '.join(pairs[target][:6])}...")


if __name__ == "__main__":
    main()
