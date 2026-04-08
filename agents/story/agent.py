import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from agent import BaseAgent
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

from agents.story.ideas import EMOTIONS, STORIES_PER_TOPIC, TOPICS
from agents.story.prompts import SYSTEM_PROMPT, USER_PROMPT


class StoryOutput(BaseModel):
    stories: list[str] = Field(description="List of short stories, each 100-150 words")


class StoryAgent(BaseAgent):
    model = "google:gemini-3.1-pro-preview"
    system_prompt = SYSTEM_PROMPT
    output_type = StoryOutput


DATA_DIR = ROOT / "data" / "stories"


def story_hash(emotion: str, topic: str) -> str:
    """Create a reproducible hash from emotion + topic using blake2b."""
    key = f"{emotion}:{topic}"
    return hashlib.blake2b(key.encode(), digest_size=16).hexdigest()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True)
async def create_story(emotion: str, topic: str, n_stories: int = STORIES_PER_TOPIC) -> StoryOutput:
    """Create a new StoryAgent and generate stories for a single emotion/topic pair."""
    agent = StoryAgent()
    prompt = USER_PROMPT.format(n_stories=n_stories, topic=topic, emotion=emotion)
    return await agent.run(prompt)


async def generate_one(emotion: str, topic: str, semaphore: asyncio.Semaphore) -> str | None:
    """Generate stories for a single emotion/topic pair, saving to a hashed filename."""
    file_hash = story_hash(emotion, topic)
    out_file = DATA_DIR / f"{file_hash}.json"

    if out_file.exists():
        return file_hash

    async with semaphore:
        try:
            output = await create_story(emotion, topic)
        except Exception as e:
            print(f"\n  SKIPPED: {emotion} / {topic} — {type(e).__name__}: {e}")
            return None
        out_file.write_text(
            json.dumps(
                {
                    "emotion": emotion,
                    "topic": topic,
                    "stories": output.stories,
                },
                indent=2,
            )
        )
        return file_hash


async def generate_all(concurrency: int = 100):
    """Generate stories for all emotion/topic combinations."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = []
    for emotion in EMOTIONS:
        for topic in TOPICS:
            tasks.append(generate_one(emotion, topic, semaphore))
    results = await tqdm.gather(*tasks, desc="Generating stories")
    print(f"\nDone. {len(results)} processed.")


def consolidate_data():
    files = [json.loads(path.read_text()) for path in sorted(DATA_DIR.glob("*.json"))]
    rows = [(f["emotion"], f["topic"], s) for f in files for s in f["stories"]]
    df = pd.DataFrame(rows, columns=["emotion", "topic", "story"])
    df.to_csv(DATA_DIR.parent / "stories.csv", index=False)
    print(f"Wrote {len(df)} stories to {DATA_DIR.parent / 'stories.csv'}")


async def main():
    await generate_all()
    consolidate_data()


if __name__ == "__main__":
    asyncio.run(main())
