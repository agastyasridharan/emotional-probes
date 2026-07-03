"""
Dataset generators — build the synthetic text the probes are extracted from.

Four datasets, one shared orchestration class:

* :class:`EmotionStoryGenerator`    — 171 emotions x 100 topics x 12 stories.
* :class:`NeutralStoryGenerator`    — 100 topics x 12 emotionless stories (PCA baseline).
* :class:`NeutralDialogueGenerator` — 100 topics x 12 neutral Human/Assistant dialogues.
* :class:`DeflectionDialogueGenerator` — dialogues where a speaker hides REAL behind DISPLAYED.

:class:`DatasetGenerator` handles everything common: enumerating work, hashing
each unit to a filename, skipping work that already exists (safe re-runs),
running with bounded async concurrency + retries, and consolidating per-unit
JSON into one CSV.

Issue #6 (intra-call homogeneity) is fixed by ``config.one_story_per_call``:
when True, each story is its own generation call instead of asking for 12 at once.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from emotion_probes.config import Config
from emotion_probes.data import EMOTIONS, TOPICS
from emotion_probes.generation import prompts
from emotion_probes.generation.backends import GenerationBackend


def _hash(*parts: str) -> str:
    """Stable filename hash for a unit of work."""
    return hashlib.blake2b(":".join(parts).encode(), digest_size=16).hexdigest()


def _subset(items: list, n: int | None) -> list:
    """First ``n`` items (all if ``n`` is None) — supports reduced-scale runs."""
    return items if n is None else items[:n]


class DatasetGenerator(ABC):
    """Shared async generation + caching + consolidation."""

    def __init__(self, config: Config, backend: GenerationBackend):
        self.config = config
        self.backend = backend

    # ---- subclasses implement these ----
    @property
    @abstractmethod
    def out_dir(self) -> Path: ...

    @property
    @abstractmethod
    def csv_path(self) -> Path: ...

    @property
    @abstractmethod
    def columns(self) -> list[str]: ...

    @abstractmethod
    def work_items(self) -> list[dict]:
        """Each item is a dict of parameters; must include a unique ``'hash'``."""

    @abstractmethod
    async def _produce(self, item: dict) -> dict:
        """Call the backend for one item and return the JSON record to save."""

    @abstractmethod
    def _records(self, saved: dict) -> list[dict]:
        """Turn one saved JSON record into one or more CSV rows (dicts)."""

    # ---- orchestration (shared) ----
    async def _process_one(self, item: dict, semaphore: asyncio.Semaphore) -> None:
        out_file = self.out_dir / f"{item['hash']}.json"
        if out_file.exists():
            return
        async with semaphore:
            try:
                record = await self._produce_with_retry(item)
            except Exception as exc:  # noqa: BLE001 - log + skip a single failure
                print(f"  SKIPPED {item['hash'][:8]}: {type(exc).__name__}: {exc}")
                return
            out_file.write_text(json.dumps(record, indent=2))

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
    async def _produce_with_retry(self, item: dict) -> dict:
        return await self._produce(item)

    async def generate_all(self, concurrency: int | None = None) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        concurrency = concurrency or self.backend.default_concurrency
        semaphore = asyncio.Semaphore(concurrency)
        items = self.work_items()
        print(f"{type(self).__name__}: {len(items)} work items, concurrency {concurrency}")
        await tqdm.gather(*(self._process_one(it, semaphore) for it in items), desc="generating")

    def consolidate(self) -> Path:
        """Flatten all per-unit JSON files into one CSV."""
        import pandas as pd

        rows: list[dict] = []
        for path in sorted(self.out_dir.glob("*.json")):
            rows.extend(self._records(json.loads(path.read_text())))
        df = pd.DataFrame(rows, columns=self.columns)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        print(f"Wrote {len(df)} rows to {self.csv_path}")
        return self.csv_path

    async def run(self, concurrency: int | None = None) -> None:
        await self.generate_all(concurrency)
        self.consolidate()


# --------------------------------------------------------------------------- #
class EmotionStoryGenerator(DatasetGenerator):
    """Stories where a character feels a target emotion (never naming it)."""

    @property
    def out_dir(self) -> Path:
        return self.config.stories_dir

    @property
    def csv_path(self) -> Path:
        return self.config.stories_csv

    @property
    def columns(self) -> list[str]:
        return ["emotion", "topic", "story"]

    def work_items(self) -> list[dict]:
        items = []
        for emotion in _subset(EMOTIONS, self.config.max_emotions):
            for topic in _subset(TOPICS, self.config.max_topics):
                if self.config.one_story_per_call:
                    for i in range(self.config.stories_per_topic):
                        items.append({
                            "emotion": emotion, "topic": topic, "index": i,
                            "hash": _hash(emotion, topic, str(i)),
                        })
                else:
                    items.append({
                        "emotion": emotion, "topic": topic,
                        "hash": _hash(emotion, topic),
                    })
        return items

    async def _produce(self, item: dict) -> dict:
        if self.config.one_story_per_call:
            user = prompts.EMOTION_STORY_USER_SINGLE.format(topic=item["topic"], emotion=item["emotion"])
            text = await self.backend.generate_text(prompts.EMOTION_STORY_SYSTEM, user)
            stories = [text.strip()]
        else:
            user = prompts.EMOTION_STORY_USER_MULTI.format(
                n=self.config.stories_per_topic, topic=item["topic"], emotion=item["emotion"]
            )
            stories = await self.backend.generate_list(
                prompts.EMOTION_STORY_SYSTEM, user, self.config.stories_per_topic
            )
        return {"emotion": item["emotion"], "topic": item["topic"], "stories": stories}

    def _records(self, saved: dict) -> list[dict]:
        return [{"emotion": saved["emotion"], "topic": saved["topic"], "story": s} for s in saved["stories"]]


# --------------------------------------------------------------------------- #
class NeutralStoryGenerator(DatasetGenerator):
    """Emotionless stories on the same topics (confound-removal baseline)."""

    @property
    def out_dir(self) -> Path:
        return self.config.neutral_stories_dir

    @property
    def csv_path(self) -> Path:
        return self.config.neutral_stories_csv

    @property
    def columns(self) -> list[str]:
        return ["topic", "story"]

    def work_items(self) -> list[dict]:
        items = []
        for topic in _subset(TOPICS, self.config.max_topics):
            if self.config.one_story_per_call:
                for i in range(self.config.stories_per_topic):
                    items.append({"topic": topic, "index": i, "hash": _hash("neutral", topic, str(i))})
            else:
                items.append({"topic": topic, "hash": _hash("neutral", topic)})
        return items

    async def _produce(self, item: dict) -> dict:
        if self.config.one_story_per_call:
            user = prompts.NEUTRAL_STORY_USER_SINGLE.format(topic=item["topic"])
            stories = [(await self.backend.generate_text(prompts.NEUTRAL_STORY_SYSTEM, user)).strip()]
        else:
            user = prompts.NEUTRAL_STORY_USER_MULTI.format(n=self.config.stories_per_topic, topic=item["topic"])
            stories = await self.backend.generate_list(
                prompts.NEUTRAL_STORY_SYSTEM, user, self.config.stories_per_topic
            )
        return {"topic": item["topic"], "stories": stories}

    def _records(self, saved: dict) -> list[dict]:
        return [{"topic": saved["topic"], "story": s} for s in saved["stories"]]


# --------------------------------------------------------------------------- #
class NeutralDialogueGenerator(DatasetGenerator):
    """Neutral Human/Assistant dialogues (deflection PCA baseline)."""

    @property
    def out_dir(self) -> Path:
        return self.config.neutral_dialogues_dir

    @property
    def csv_path(self) -> Path:
        return self.config.neutral_dialogues_csv

    @property
    def columns(self) -> list[str]:
        return ["topic", "dialogue"]

    def work_items(self) -> list[dict]:
        return [{"topic": topic, "hash": _hash("neutral_dialogue", topic)}
                for topic in _subset(TOPICS, self.config.max_topics)]

    async def _produce(self, item: dict) -> dict:
        user = prompts.NEUTRAL_DIALOGUE_USER.format(n=self.config.stories_per_topic, topic=item["topic"])
        dialogues = await self.backend.generate_list(
            prompts.NEUTRAL_DIALOGUE_SYSTEM, user, self.config.stories_per_topic
        )
        # Relabel to the paper's transcript format (hygiene #8).
        dialogues = [d.replace("Person:", "Human:").replace("AI:", "Assistant:") for d in dialogues]
        return {"topic": item["topic"], "dialogues": dialogues}

    def _records(self, saved: dict) -> list[dict]:
        return [{"topic": saved["topic"], "dialogue": d} for d in saved["dialogues"]]


# --------------------------------------------------------------------------- #
class DeflectionDialogueGenerator(DatasetGenerator):
    """Dialogues where NAME_A truly feels REAL_EMOTION but displays DISPLAYED."""

    def __init__(self, config: Config, backend: GenerationBackend):
        super().__init__(config, backend)
        self.pairs = json.loads(config.deflection_pairs_json.read_text())

    @property
    def out_dir(self) -> Path:
        return self.config.deflection_dialogues_dir

    @property
    def csv_path(self) -> Path:
        return self.config.deflection_dialogues_csv

    @property
    def columns(self) -> list[str]:
        return ["real_emotion", "displayed_emotion", "topic", "name_a", "name_b", "dialogue"]

    def work_items(self) -> list[dict]:
        import random

        rng = random.Random(0)  # deterministic name sampling
        from emotion_probes.generation.names import NAMES

        items = []
        for real_emotion, displayed_list in self.pairs.items():
            for displayed_emotion in displayed_list:
                for topic in _subset(TOPICS, self.config.max_topics):
                    name_a, name_b = rng.sample(NAMES, 2)
                    items.append({
                        "real_emotion": real_emotion,
                        "displayed_emotion": displayed_emotion,
                        "topic": topic,
                        "name_a": name_a,
                        "name_b": name_b,
                        "hash": _hash(real_emotion, displayed_emotion, topic),
                    })
        return items

    async def _produce(self, item: dict) -> dict:
        user = prompts.DEFLECTION_USER.format(
            name_a=item["name_a"], name_b=item["name_b"],
            real_emotion=item["real_emotion"], displayed_emotion=item["displayed_emotion"],
            topic=item["topic"],
        )
        dialogue = await self.backend.generate_text(prompts.DEFLECTION_SYSTEM, user)
        return {**{k: item[k] for k in ("real_emotion", "displayed_emotion", "topic", "name_a", "name_b")},
                "dialogue": dialogue}

    def _records(self, saved: dict) -> list[dict]:
        return [saved]


# --------------------------------------------------------------------------- #
GENERATORS = {
    "emotion_stories": EmotionStoryGenerator,
    "neutral_stories": NeutralStoryGenerator,
    "neutral_dialogues": NeutralDialogueGenerator,
    "deflection": DeflectionDialogueGenerator,
}


def main() -> None:
    """CLI: ``python -m emotion_probes.generation.generators <dataset>``.

    Uses the external API backend by default. For self-generation by the probed
    model, set ``EMOTION_PROBES_SELF_GENERATE=1`` (loads the model — needs a GPU).
    """
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate a synthetic dataset.")
    parser.add_argument("dataset", choices=list(GENERATORS))
    parser.add_argument("--concurrency", type=int, default=None)
    args = parser.parse_args()

    from emotion_probes.generation.backends import build_backend

    def _int_env(name: str) -> int | None:
        v = os.environ.get(name)
        return int(v) if v else None

    overrides: dict = {"self_generate": bool(int(os.environ.get("EMOTION_PROBES_SELF_GENERATE", "0")))}
    if _int_env("EMOTION_PROBES_MAX_EMOTIONS") is not None:
        overrides["max_emotions"] = _int_env("EMOTION_PROBES_MAX_EMOTIONS")
    if _int_env("EMOTION_PROBES_MAX_TOPICS") is not None:
        overrides["max_topics"] = _int_env("EMOTION_PROBES_MAX_TOPICS")
    if _int_env("EMOTION_PROBES_STORIES_PER_TOPIC") is not None:
        overrides["stories_per_topic"] = _int_env("EMOTION_PROBES_STORIES_PER_TOPIC")
    if os.environ.get("EMOTION_PROBES_API_MODEL"):
        overrides["api_generator_model"] = os.environ["EMOTION_PROBES_API_MODEL"]
        overrides["api_generator_model_fast"] = os.environ["EMOTION_PROBES_API_MODEL"]
    config = Config(**overrides)
    probed = None
    if config.self_generate:
        from emotion_probes.models import ProbedModel

        probed = ProbedModel(config)
    # Deflection uses the cheaper API model by default.
    model_name = config.api_generator_model_fast if args.dataset == "deflection" else config.api_generator_model
    backend = build_backend(config, probed_model=probed, model_name=model_name)

    generator = GENERATORS[args.dataset](config, backend)
    asyncio.run(generator.run(args.concurrency))


if __name__ == "__main__":
    main()
