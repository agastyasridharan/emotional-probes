"""Pre-slice the WORKFLOW share of the emotion-story work-list into per-agent
assignment files, so each ultracode workflow agent just reads one small JSON
and generates that batch. Keeps the workflow's `args` tiny (only a count).

Share = the 30% of items whose hash falls in buckets {0,1,2} — the exact
complement of what `gen_channel.py --channel api --workflow-frac 0.3` generates,
so the two channels are disjoint and jointly complete.
"""
from __future__ import annotations

import json
import math
import os
import shutil

from emotion_probes.config import Config
from emotion_probes.generation.generators import EmotionStoryGenerator

BS = int(os.environ.get("WF_BATCH", "30"))
ASSIGN_DIR = os.environ.get("WF_ASSIGN_DIR", "/tmp/wf_assign")

cfg = Config(
    max_topics=int(os.environ.get("EMOTION_PROBES_MAX_TOPICS", "25")),
    stories_per_topic=int(os.environ.get("EMOTION_PROBES_STORIES_PER_TOPIC", "4")),
)
gen = EmotionStoryGenerator(cfg, None)
full = gen.work_items()
wf = [it for it in full if int(it["hash"][:8], 16) % 10 < 3]

if os.path.isdir(ASSIGN_DIR):
    shutil.rmtree(ASSIGN_DIR)
os.makedirs(ASSIGN_DIR, exist_ok=True)

n = math.ceil(len(wf) / BS)
for i in range(n):
    chunk = wf[i * BS:(i + 1) * BS]
    json.dump(
        [{"hash": it["hash"], "emotion": it["emotion"], "topic": it["topic"]} for it in chunk],
        open(f"{ASSIGN_DIR}/{i}.json", "w"),
    )

print(json.dumps({"workflow_items": len(wf), "of_total": len(full),
                  "num_chunks": n, "batch_size": BS, "assign_dir": ASSIGN_DIR}))
