"""Merge ultracode workflow batch files (delimiter format) into per-hash
deflection JSONs in ``config.deflection_dialogues_dir`` — the SAME format/dir the
API channel writes, so the standard ``consolidate()`` merges both channels into
one deflection_dialogues.csv.

Each saved record is {real_emotion, displayed_emotion, topic, name_a, name_b,
dialogue}: the agent supplies only the ``dialogue`` body; the other five fields
are looked up from the work-list meta (keyed by hash). Tolerant by design —
unknown-hash / empty blocks are skipped and counted (backfilled via the API
channel, which skips hashes that already exist).
"""
from __future__ import annotations

import json
import os
import re

from emotion_probes.config import Config
from emotion_probes.generation.generators import DeflectionDialogueGenerator

cfg = Config(max_topics=int(os.environ.get("EMOTION_PROBES_MAX_TOPICS", "14")))
gen = DeflectionDialogueGenerator(cfg, None)
meta = {it["hash"]: it for it in gen.work_items()}
out_dir = cfg.deflection_dialogues_dir
out_dir.mkdir(parents=True, exist_ok=True)

BATCH = os.environ.get("WF_BATCH_DIR", "/tmp/wf_batches_def")
marker = re.compile(r"^@@@DEF\s+([0-9a-f]+)\s*$", re.M)
fields = ("real_emotion", "displayed_emotion", "topic", "name_a", "name_b")

files = written = unknown = empty = no_scenario = 0
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
        if "scenario" not in body.lower():
            no_scenario += 1  # construct check: a deflection sample needs a scenario line
        m = meta[h]
        rec = {k: m[k] for k in fields}
        rec["dialogue"] = body
        json.dump(rec, open(out_dir / f"{h}.json", "w"), indent=2)
        written += 1

print(json.dumps({"batch_files": files, "written": written, "unknown_hash": unknown,
                  "empty": empty, "missing_scenario": no_scenario}))
