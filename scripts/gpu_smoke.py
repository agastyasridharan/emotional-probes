"""Tiny GPU smoke test — validate the full GPU path before any large run.

Loads the probed model, extracts activations for a few stories across 2 emotions,
builds their diff-of-means directions at the analysis layer, and prints sanity
stats. Run with CUDA_VISIBLE_DEVICES set EXTERNALLY (never hardcode it).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from emotion_probes.config import Config
from emotion_probes.models import ProbedModel

cfg = Config()
print(f"model={cfg.model_id} device_map={cfg.device_map} "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

model = ProbedModel(cfg)
model.load()
L = cfg.analysis_layer(model.num_layers)
print(f"loaded: num_layers={model.num_layers} hidden={model.hidden_size} analysis_layer={L}")

df = pd.read_csv(cfg.stories_csv)
emos = list(df.emotion.unique()[:2])
sub = df[df.emotion.isin(emos)].groupby("emotion").head(4).reset_index(drop=True)
print(f"smoke emotions={emos} n_stories={len(sub)}")

means = model.extract_means(sub.story.tolist())  # (N, L, H)
print(f"means shape={means.shape}")

labels = sub.emotion.tolist()
g = means[:, L, :].mean(0)
d = {e: means[[i for i, l in enumerate(labels) if l == e], L, :].mean(0) - g for e in emos}
d0, d1 = d[emos[0]], d[emos[1]]
cos = float(d0 @ d1 / (np.linalg.norm(d0) * np.linalg.norm(d1) + 1e-9))
print(f"L={L} |d0|={np.linalg.norm(d0):.3f} |d1|={np.linalg.norm(d1):.3f} cos(d0,d1)={cos:.3f}")
print("SMOKE_OK")
