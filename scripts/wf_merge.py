"""Merge ultracode workflow batch files (delimiter format) into per-hash story
JSONs in ``config.stories_dir`` — the SAME format/dir the API channel writes, so
the standard ``consolidate()`` merges both channels into one stories.csv.

Tolerant by design: blocks with an unknown hash or empty body are skipped and
counted (those work-items can be backfilled via the API channel, which simply
skips the hashes that already exist).
"""
from __future__ import annotations

import json
import os
import re

from emotion_probes.config import Config
from emotion_probes.generation.generators import EmotionStoryGenerator

cfg = Config(
    max_topics=int(os.environ.get("EMOTION_PROBES_MAX_TOPICS", "25")),
    stories_per_topic=int(os.environ.get("EMOTION_PROBES_STORIES_PER_TOPIC", "4")),
)
gen = EmotionStoryGenerator(cfg, None)
meta = {it["hash"]: it for it in gen.work_items()}
out_dir = cfg.stories_dir
out_dir.mkdir(parents=True, exist_ok=True)

BATCH = os.environ.get("WF_BATCH_DIR", "/tmp/wf_batches")
marker = re.compile(r"^@@@STORY\s+([0-9a-f]+)\s*$", re.M)

files = written = unknown = empty = leaked = 0
for fn in sorted(os.listdir(BATCH)):
    if not fn.endswith(".txt"):
        continue
    files += 1
    parts = marker.split(open(os.path.join(BATCH, fn)).read())
    for i in range(1, len(parts), 2):
        h = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if h not in meta:
            unknown += 1
            continue
        if not body:
            empty += 1
            continue
        m = meta[h]
        if m["emotion"].lower() in body.lower():
            leaked += 1  # construct-validity check: emotion word literally present
        json.dump({"emotion": m["emotion"], "topic": m["topic"], "stories": [body]},
                  open(out_dir / f"{h}.json", "w"), indent=2)
        written += 1

print(json.dumps({"batch_files": files, "written": written, "unknown_hash": unknown,
                  "empty": empty, "leaked_emotion_word": leaked}))
