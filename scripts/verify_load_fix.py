"""
Confirm the in-load gc fix on the PRODUCTION path (ProbedModel.load), with NO
external gc. Expect resident ~47 GB/card straight after load, and the default
batches (story 32 / deflection 8) to run with comfortable headroom.
"""

import time

import pandas as pd
import torch

from emotion_probes.config import Config
from emotion_probes.models.language_model import ProbedModel

ndev = torch.cuda.device_count()
cfg = Config()
t0 = time.time()
m = ProbedModel(cfg)  # production pattern: __init__ auto-loads + reclaims (no double .load())
resident = [round(torch.cuda.memory_allocated(d) / 1e9, 1) for d in range(ndev)]
print("LOADED %.1fs | RESIDENT_GB/card %s  (expect ~47, NOT ~93)" % (time.time() - t0, resident), flush=True)
assert max(resident) < 70, "in-load gc fix did NOT take effect (residency still bloated)"

L = cfg.analysis_layer(m.num_layers)
stories = pd.read_csv(cfg.stories_csv)["story"].astype(str).tolist()[:64]
for d in range(ndev):
    torch.cuda.reset_peak_memory_stats(d)
t1 = time.time()
out = m.extract_means(stories, skip=0, batch_size=cfg.batch_size, layers=None, max_length=512)
peak = [round(torch.cuda.max_memory_allocated(d) / 1e9, 1) for d in range(ndev)]
print("STORY batch=%d | %d stories %.1fs | %.2f st/s | peak/card GB %s"
      % (cfg.batch_size, len(stories), time.time() - t1, len(stories) / (time.time() - t1), peak), flush=True)

print("VERIFY_OK", flush=True)
