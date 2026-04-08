import asyncio
import hashlib
import json
import random
from pathlib import Path

import pandas as pd
from agent import BaseAgent
from pydantic import BaseModel, Field
from story_agent.ideas import TOPICS
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from deflection_story_agent.names import NAMES
from deflection_story_agent.prompts import SYSTEM_PROMPT, USER_PROMPT

PAIRS_PATH = Path(__file__).parent.parent / "data" / "deflection_pairs.json"


class DeflectionOutput(BaseModel):
    dialogue: str = Field(description="Scenario description followed by the dialogue between the two characters")


class DeflectionAgent(BaseAgent):
    model = "google:gemini-3-flash-preview"
    system_prompt = SYSTEM_PROMPT
    output_type = DeflectionOutput


DATA_DIR = Path(__file__).parent.parent / "data" / "deflection_dialogues"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def dialogue_hash(real_emotion: str, displayed_emotion: str, topic: str) -> str:
    """Reproducible hash for a single (real, displayed, topic) triple."""
    key = f"{real_emotion}:{displayed_emotion}:{topic}"
    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
async def create_dialogue(
    real_emotion: str, displayed_emotion: str, topic: str, name_a: str, name_b: str
) -> DeflectionOutput:
    agent = DeflectionAgent()
    prompt = USER_PROMPT.format(
        NAME_A=name_a,
        NAME_B=name_b,
        REAL_EMOTION=real_emotion,
        DISPLAYED_EMOTION=displayed_emotion,
        TOPIC=topic,
    )
    return await agent.run(prompt)


async def generate_one(
    real_emotion: str, displayed_emotion: str, topic: str, semaphore: asyncio.Semaphore
) -> str | None:
    file_hash = dialogue_hash(real_emotion, displayed_emotion, topic)
    out_file = DATA_DIR / f"{file_hash}.json"

    if out_file.exists():
        return file_hash

    # Sample two distinct names per dialogue.
    name_a, name_b = random.sample(NAMES, 2)

    async with semaphore:
        try:
            output = await create_dialogue(real_emotion, displayed_emotion, topic, name_a, name_b)
        except Exception as e:
            print(f"\n  SKIPPED: {real_emotion}→{displayed_emotion} / {topic[:40]} — {type(e).__name__}: {e}")
            return None
        out_file.write_text(
            json.dumps(
                {
                    "real_emotion": real_emotion,
                    "displayed_emotion": displayed_emotion,
                    "topic": topic,
                    "name_a": name_a,
                    "name_b": name_b,
                    "dialogue": output.dialogue,
                },
                indent=2,
            )
        )
        return file_hash


async def generate_all(concurrency: int = 100):
    pairs = json.loads(PAIRS_PATH.read_text())
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for real_emotion, displayed_list in pairs.items():
        for displayed_emotion in displayed_list:
            for topic in TOPICS:
                tasks.append(generate_one(real_emotion, displayed_emotion, topic, semaphore))
    print(
        f"Generating {len(tasks)} dialogues "
        f"({len(pairs)} targets × {len(next(iter(pairs.values())))} displayed × {len(TOPICS)} topics)"
    )
    results = await tqdm.gather(*tasks, desc="Generating deflection dialogues")
    print(f"\nDone. {sum(r is not None for r in results)}/{len(results)} processed.")


def consolidate_data():
    files = [json.loads(path.read_text()) for path in sorted(DATA_DIR.glob("*.json"))]
    rows = [
        (f["real_emotion"], f["displayed_emotion"], f["topic"], f["name_a"], f["name_b"], f["dialogue"]) for f in files
    ]
    df = pd.DataFrame(rows, columns=["real_emotion", "displayed_emotion", "topic", "name_a", "name_b", "dialogue"])
    df.to_csv(DATA_DIR.parent / "deflection_dialogues.csv", index=False)
    print(f"Wrote {len(df)} dialogues to {DATA_DIR.parent / 'deflection_dialogues.csv'}")


async def main():
    await generate_all()
    consolidate_data()


if __name__ == "__main__":
    asyncio.run(main())
