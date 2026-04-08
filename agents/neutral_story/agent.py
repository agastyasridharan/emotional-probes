import asyncio
import hashlib
import json
from pathlib import Path

import pandas as pd
from agent import BaseAgent
from pydantic import BaseModel, Field
from story_agent.ideas import TOPICS

STORIES_PER_TOPIC = 12
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from neutral_story_agent.prompts import SYSTEM_PROMPT, USER_PROMPT


class StoryOutput(BaseModel):
    stories: list[str] = Field(description="List of short neutral passages, each 100-150 words")


class NeutralStoryAgent(BaseAgent):
    model = "google:gemini-3.1-pro-preview"
    system_prompt = SYSTEM_PROMPT
    output_type = StoryOutput


DATA_DIR = Path(__file__).parent.parent / "data" / "neutral_stories"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def story_hash(topic: str) -> str:
    """Create a reproducible hash from topic using blake2b."""
    return hashlib.blake2b(topic.encode(), digest_size=16).hexdigest()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
async def create_story(topic: str, n_stories: int = STORIES_PER_TOPIC) -> StoryOutput:
    """Create a new NeutralStoryAgent and generate passages for a single topic."""
    agent = NeutralStoryAgent()
    prompt = USER_PROMPT.format(n_stories=n_stories, topic=topic)
    return await agent.run(prompt)


async def generate_one(topic: str, semaphore: asyncio.Semaphore) -> str | None:
    """Generate passages for a single topic, saving to a hashed filename."""
    file_hash = story_hash(topic)
    out_file = DATA_DIR / f"{file_hash}.json"

    if out_file.exists():
        return file_hash

    async with semaphore:
        try:
            output = await create_story(topic)
        except Exception as e:
            print(f"\n  SKIPPED: {topic} — {type(e).__name__}: {e}")
            return None
        out_file.write_text(
            json.dumps(
                {
                    "topic": topic,
                    "stories": output.stories,
                },
                indent=2,
            )
        )
        return file_hash


async def generate_all(concurrency: int = 100):
    """Generate neutral passages for all topics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [generate_one(topic, semaphore) for topic in TOPICS]
    results = await tqdm.gather(*tasks, desc="Generating neutral passages")
    print(f"\nDone. {len(results)} processed.")


def consolidate_data():
    files = [json.loads(path.read_text()) for path in sorted(DATA_DIR.glob("*.json"))]
    rows = [(f["topic"], s) for f in files for s in f["stories"]]
    df = pd.DataFrame(rows, columns=["topic", "story"])
    df.to_csv(DATA_DIR.parent / "neutral_stories.csv", index=False)
    print(f"Wrote {len(df)} passages to {DATA_DIR.parent / 'neutral_stories.csv'}")


async def main():
    await generate_all()
    consolidate_data()


if __name__ == "__main__":
    asyncio.run(main())
