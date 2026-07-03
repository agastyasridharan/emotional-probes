"""
Llama-3.3-70B staging smoke test (Athena, GPUs 0,2,3 only).

Validates the 70B can load and probe correctly BEFORE committing the multi-day
full extraction:
  * confirms exactly 3 CUDA devices are visible (CUDA_VISIBLE_DEVICES=0,2,3),
  * loads the model sharded with device_map="auto",
  * checks layer/dim/analysis-layer auto-detection (expect 80 layers -> layer 53),
  * runs a real forward pass across the shards and builds a tiny diff-of-means
    (afraid - calm) at the analysis layer, asserting it is non-trivial.

Run from the repo root (so `import emotion_probes` works) with the env block:
  EMOTION_PROBES_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
  EMOTION_PROBES_DEVICE_MAP=auto
  EMOTION_PROBES_DATA=/data/agastyas/emotion-probes/data-llama70b
  CUDA_VISIBLE_DEVICES=0,2,3
This file NEVER sets CUDA_VISIBLE_DEVICES itself (see Athena rules).
"""

import time

t0 = time.time()
import numpy as np
import torch

from emotion_probes.config import Config
from emotion_probes.models.language_model import ProbedModel

cfg = Config()
print("MODEL_ID", cfg.model_id, flush=True)
print("DEVICE_MAP", cfg.device_map, flush=True)
print("DATA_DIR", cfg.data_dir, flush=True)

n_vis = torch.cuda.device_count()
print("VISIBLE_CUDA_DEVICES", n_vis, flush=True)
for i in range(n_vis):
    print("  logical cuda:%d ->" % i, torch.cuda.get_device_name(i), flush=True)
assert n_vis == 3, f"expected 3 visible GPUs (0,2,3), saw {n_vis}"

print("LOADING ...", flush=True)
m = ProbedModel(cfg).load()
print("LOADED_IN_SEC %.1f" % (time.time() - t0), flush=True)
print("NUM_LAYERS", m.num_layers, flush=True)
print("HIDDEN", m.hidden_size, flush=True)
L = cfg.analysis_layer(m.num_layers)
print("ANALYSIS_LAYER", L, flush=True)

dm = getattr(m.model, "hf_device_map", None)
if dm:
    devs = sorted({str(v) for v in dm.values()})
    print("HF_DEVICE_MAP_DEVICES", devs, flush=True)
print("INPUT_DEVICE", m._input_device, flush=True)

afraid = [
    "I am terrified, my hands are shaking and my heart pounds with dread.",
    "A cold fear grips me; something terrible is about to happen.",
]
calm = [
    "I feel completely at peace, relaxed and serene in the quiet morning.",
    "A gentle calm settles over me; everything is tranquil and still.",
]
A = m.extract_means(afraid, skip=0, layers=[L])[:, 0, :]  # (2, H)
C = m.extract_means(calm, skip=0, layers=[L])[:, 0, :]
v = A.mean(0) - C.mean(0)
diff_norm = float(np.linalg.norm(v))
print("A_MEAN_NORM", float(np.linalg.norm(A.mean(0))), flush=True)
print("C_MEAN_NORM", float(np.linalg.norm(C.mean(0))), flush=True)
print("DIFF_OF_MEANS_NORM", diff_norm, flush=True)
assert diff_norm > 0.0, "diff-of-means is zero -> forward pass / extraction is a no-op"

print("ELAPSED_SEC %.1f" % (time.time() - t0), flush=True)
print("SMOKE_OK", flush=True)
