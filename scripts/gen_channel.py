"""Generate one dataset via the Anthropic API backend — optionally only a
deterministic *partition share* of the work-list.

This supports splitting generation across two billing channels:
  * this script  -> the Anthropic API key   (the "api" share)
  * an ultracode workflow of Sonnet agents -> the remaining "workflow" share

Partition is by work-item hash (stable, disjoint), so the two channels never
duplicate or miss an item, and both write per-unit JSON into the SAME out_dir;
a final ``--consolidate`` (or a separate consolidate run) merges everything.

Scale + model come from the EMOTION_PROBES_* env vars (see Config).
"""
from __future__ import annotations

import argparse
import asyncio
import os

from emotion_probes.config import Config
from emotion_probes.generation.backends import ApiBackend
from emotion_probes.generation.generators import GENERATORS


def _int_env(name: str) -> int | None:
    v = os.environ.get(name)
    return int(v) if v else None


def _config_from_env() -> Config:
    overrides: dict = {}
    for env, key in [
        ("EMOTION_PROBES_MAX_EMOTIONS", "max_emotions"),
        ("EMOTION_PROBES_MAX_TOPICS", "max_topics"),
        ("EMOTION_PROBES_STORIES_PER_TOPIC", "stories_per_topic"),
    ]:
        if _int_env(env) is not None:
            overrides[key] = _int_env(env)
    if os.environ.get("EMOTION_PROBES_API_MODEL"):
        overrides["api_generator_model"] = os.environ["EMOTION_PROBES_API_MODEL"]
        overrides["api_generator_model_fast"] = os.environ["EMOTION_PROBES_API_MODEL"]
    return Config(**overrides)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", choices=list(GENERATORS))
    p.add_argument("--channel", choices=["api", "workflow", "all"], default="all")
    p.add_argument("--workflow-frac", type=float, default=0.0,
                   help="fraction (rounded to tenths) of items reserved for the workflow channel")
    p.add_argument("--concurrency", type=int, default=20)
    p.add_argument("--consolidate", action="store_true")
    args = p.parse_args()

    cfg = _config_from_env()
    model = cfg.api_generator_model_fast if args.dataset == "deflection" else cfg.api_generator_model
    gen = GENERATORS[args.dataset](cfg, ApiBackend(model))

    full = gen.work_items()
    wf_buckets = round(args.workflow_frac * 10)

    def is_workflow(it: dict) -> bool:
        return int(it["hash"][:8], 16) % 10 < wf_buckets

    if args.channel == "api":
        items = [it for it in full if not is_workflow(it)]
    elif args.channel == "workflow":
        items = [it for it in full if is_workflow(it)]
    else:
        items = full

    gen.work_items = lambda: items  # type: ignore[method-assign]
    print(f"{args.dataset}: channel={args.channel} workflow_frac={args.workflow_frac} "
          f"-> {len(items)}/{len(full)} items, model={model}, concurrency={args.concurrency}")
    asyncio.run(gen.generate_all(args.concurrency))
    if args.consolidate:
        gen.consolidate()


if __name__ == "__main__":
    main()
