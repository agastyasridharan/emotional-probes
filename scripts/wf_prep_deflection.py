"""Pre-slice the WORKFLOW share of the DEFLECTION work-list into per-agent
assignment files (the Stage-2 analogue of ``wf_prep.py``).

Share = the 30% of items whose hash falls in buckets {0,1,2} — the exact
complement of ``gen_channel.py deflection --channel api --workflow-frac 0.3``,
so the API channel and the workflow channel are disjoint and jointly complete.

Each assignment item carries everything an agent needs to write one dialogue:
the hash plus (real_emotion, displayed_emotion, topic, name_a, name_b).
"""
from __future__ import annotations

import json
import math
import os
import shutil

from emotion_probes.config import Config
from emotion_probes.generation.generators import DeflectionDialogueGenerator

BS = int(os.environ.get("WF_BATCH", "30"))
ASSIGN_DIR = os.environ.get("WF_ASSIGN_DIR", "/tmp/wf_assign_def")

cfg = Config(max_topics=int(os.environ.get("EMOTION_PROBES_MAX_TOPICS", "14")))
gen = DeflectionDialogueGenerator(cfg, None)
full = gen.work_items()
wf = [it for it in full if int(it["hash"][:8], 16) % 10 < 3]

if os.path.isdir(ASSIGN_DIR):
    shutil.rmtree(ASSIGN_DIR)
os.makedirs(ASSIGN_DIR, exist_ok=True)

keep = ("hash", "real_emotion", "displayed_emotion", "topic", "name_a", "name_b")
n = math.ceil(len(wf) / BS)
for i in range(n):
    chunk = wf[i * BS:(i + 1) * BS]
    json.dump([{k: it[k] for k in keep} for it in chunk], open(f"{ASSIGN_DIR}/{i}.json", "w"))

print(json.dumps({"workflow_items": len(wf), "of_total": len(full),
                  "num_chunks": n, "batch_size": BS, "assign_dir": ASSIGN_DIR}))
