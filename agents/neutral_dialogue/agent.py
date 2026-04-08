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

from agents.story.ideas import TOPICS
from agents.neutral_dialogue.prompts import SYSTEM_PROMPT, USER_PROMPT

STORIES_PER_TOPIC = 12


class DialogueOutput(BaseModel):
    stories: list[str] = Field(description="List of neutral dialogues between Person and AI, free-text format")


class NeutralDialogueAgent(BaseAgent):
    model = "google:gemini-3.1-pro-preview"
    system_prompt = SYSTEM_PROMPT
    output_type = DialogueOutput


DATA_DIR = ROOT / "data" / "neutral_dialogues"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def story_hash(topic: str) -> str:
    return hashlib.blake2b(topic.encode(), digest_size=16).hexdigest()


def log_retry(retry_state):
    print(f"\n  RETRY {retry_state.attempt_number}/5: {retry_state.outcome.exception()}")

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=60), reraise=True, before_sleep=log_retry)
async def create_dialogues(topic: str, n_stories: int = STORIES_PER_TOPIC) -> DialogueOutput:
    agent = NeutralDialogueAgent()
    prompt = USER_PROMPT.format(n_stories=n_stories, topic=topic)
    return await agent.run(prompt)


async def generate_one(topic: str, semaphore: asyncio.Semaphore) -> str | None:
    file_hash = story_hash(topic)
    out_file = DATA_DIR / f"{file_hash}.json"

    if out_file.exists():
        return file_hash

    async with semaphore:
        try:
            output = await create_dialogues(topic)
        except Exception as e:
            print(f"\n  SKIPPED: {topic} — {type(e).__name__}: {e}")
            return None
        # Post-hoc: convert Person→Human, AI→Assistant (per paper).
        dialogues = []
        for d in output.stories:
            d = d.replace("Person:", "Human:")
            d = d.replace("AI:", "Assistant:")
            dialogues.append(d)
        out_file.write_text(
            json.dumps(
                {
                    "topic": topic,
                    "dialogues": dialogues,
                },
                indent=2,
            )
        )
        return file_hash


async def generate_all(concurrency: int = 100):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [generate_one(topic, semaphore) for topic in TOPICS]
    results = await tqdm.gather(*tasks, desc="Generating neutral dialogues")
    print(f"\nDone. {len(results)} processed.")


def consolidate_data():
    files = [json.loads(path.read_text()) for path in sorted(DATA_DIR.glob("*.json"))]
    rows = [(f["topic"], d) for f in files for d in f["dialogues"]]
    df = pd.DataFrame(rows, columns=["topic", "dialogue"])
    df.to_csv(DATA_DIR.parent / "neutral_dialogues.csv", index=False)
    print(f"Wrote {len(df)} dialogues to {DATA_DIR.parent / 'neutral_dialogues.csv'}")


async def main():
    await generate_all()
    consolidate_data()


if __name__ == "__main__":
    asyncio.run(main())
