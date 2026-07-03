"""
Batch-size throughput benchmark for Llama-3.3-70B on Athena (GPUs 0,2,3).

Answers "does a bigger batch help, and how big can we go?" empirically:
loads the 70B ONCE (device_map=auto over the 3 visible cards), then sweeps
batch sizes on the REAL story path (extract_means, all 80 layers captured,
512 tokens) using real corpus text. Reports stories/s, tokens/s, and peak
GPU memory per card, and stops at the first OOM (= the memory ceiling).

Never sets CUDA_VISIBLE_DEVICES (the runner does: 0,2,3 only).
"""

import time

import numpy as np
import pandas as pd
import torch

from emotion_probes.config import Config
from emotion_probes.models.language_model import ProbedModel

cfg = Config()
ndev = torch.cuda.device_count()
print("MODEL_ID", cfg.model_id, "| visible GPUs", ndev, flush=True)


def reset_peaks():
    for d in range(ndev):
        torch.cuda.reset_peak_memory_stats(d)


def peaks_gb():
    return [round(torch.cuda.max_memory_allocated(d) / 1e9, 1) for d in range(ndev)]


# --- load once -------------------------------------------------------------
t0 = time.time()
m = ProbedModel(cfg).load()
print("LOADED_IN_SEC %.1f | layers=%d hidden=%d analysis_layer=%d"
      % (time.time() - t0, m.num_layers, m.hidden_size, cfg.analysis_layer(m.num_layers)),
      flush=True)
# steady resident (memory_allocated, NOT max) = the true weight footprint per card
resident = [round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in range(ndev)]
print("STEADY_RESIDENT_GB_PER_CARD", resident, "| (bf16 should be ~47, fp32-bloat ~93)", flush=True)
# HYPOTHESIS: a fresh first load orphans a ~1x duplicate pending GC; reclaim it.
import gc  # noqa: E402
gc.collect()
torch.cuda.empty_cache()
resident_gc = [round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in range(ndev)]
print("AFTER_GC_RESIDENT_GB_PER_CARD", resident_gc, "| if this drops to ~47, gc is the fix", flush=True)

# --- real corpus text (pick the longest-text column robustly) --------------
df = pd.read_csv(cfg.stories_csv)
text_col = max(df.columns, key=lambda c: df[c].astype(str).str.len().mean())
texts = df[text_col].astype(str).tolist()
print("STORIES_CSV rows=%d text_col=%r" % (len(df), text_col), flush=True)

N = 256  # plenty of passes even at large batch
pool = texts[:N]

# warm up the kernels/pipeline so the first timed batch isn't penalized
_ = m.extract_means(pool[:8], skip=0, batch_size=8, layers=None, max_length=512)

print("=== STORY PATH SWEEP (all 80 layers captured, 512 tok, N=%d) ===" % N, flush=True)
for B in [16, 32, 48, 64, 96, 128, 192]:
    torch.cuda.empty_cache()
    reset_peaks()
    try:
        t1 = time.time()
        out = m.extract_means(pool, skip=0, batch_size=B, layers=None, max_length=512)
        dt = time.time() - t1
        sps = N / dt
        tps = N * 512 / dt
        print("BATCH %3d | %5.1fs | %5.2f stories/s | %6.0f tok/s | peak/card GB %s"
              % (B, dt, sps, tps, peaks_gb()), flush=True)
        del out
    except RuntimeError as e:
        msg = str(e).split("\n")[0]
        print("BATCH %3d | OOM/ERROR: %s" % (B, msg), flush=True)
        torch.cuda.empty_cache()
        break

# --- lighter deflection-path probe (iter_token_activations, 2048 tok) ------
print("=== DEFLECTION PATH PROBE (all layers, 2048 tok) ===", flush=True)
try:
    dfd = pd.read_csv(cfg.deflection_dialogues_csv)
    dcol = max(dfd.columns, key=lambda c: dfd[c].astype(str).str.len().mean())
    dtexts = dfd[dcol].astype(str).tolist()[:64]
    for B in [8, 16, 24, 32]:
        torch.cuda.empty_cache()
        reset_peaks()
        try:
            t1 = time.time()
            n = 0
            for _item in m.iter_token_activations(
                dtexts, batch_size=B, layers=None, max_length=2048, with_offsets=True
            ):
                n += 1
            dt = time.time() - t1
            print("DEFL_BATCH %3d | %d dialogues | %5.1fs | %5.2f dlg/s | peak/card GB %s"
                  % (B, n, dt, n / dt, peaks_gb()), flush=True)
        except RuntimeError as e:
            print("DEFL_BATCH %3d | OOM/ERROR: %s" % (B, str(e).split(chr(10))[0]), flush=True)
            torch.cuda.empty_cache()
            break
except Exception as e:  # noqa: BLE001 - probe only
    print("DEFL_PROBE_SKIPPED", repr(e), flush=True)

print("BENCH_DONE", flush=True)
