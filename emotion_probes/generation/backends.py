"""
Generation backends — *where* the synthetic text comes from.

Two interchangeable backends implement the same tiny interface, so the
generators don't care which is used:

* :class:`ApiBackend` — an external API model (e.g. Gemini) via ``pydantic-ai``.
  This is the practical default for the very large datasets.

* :class:`LocalModelBackend` — the **probed open model itself** (via
  :class:`~emotion_probes.models.ProbedModel`). This fixes Issue #2: the paper
  used the model it was probing to author the stories, so the emotion vectors
  reflect *that model's* associations. With an open model we can finally do the
  same. (Self-generation pairs naturally with ``one_story_per_call`` so a small
  model never has to emit a JSON array.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing torch on non-GPU machines
    from emotion_probes.models.language_model import ProbedModel


class GenerationBackend(ABC):
    """Minimal text-generation interface used by the dataset generators."""

    #: A sensible default async concurrency for this backend.
    default_concurrency: int = 1

    @abstractmethod
    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        """Return a single free-text completion."""

    @abstractmethod
    async def generate_list(self, system_prompt: str, user_prompt: str, n: int) -> list[str]:
        """Return up to ``n`` strings (the generators ask for a JSON array)."""


class ApiBackend(GenerationBackend):
    """Generate with an external API model through ``pydantic-ai``."""

    default_concurrency = 100

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        from pydantic_ai import Agent

        agent = Agent(self.model_name, system_prompt=system_prompt, output_type=str)
        result = await agent.run(user_prompt)
        return result.output

    async def generate_list(self, system_prompt: str, user_prompt: str, n: int) -> list[str]:
        from pydantic import BaseModel, Field
        from pydantic_ai import Agent

        class _Items(BaseModel):
            items: list[str] = Field(description="The list of generated passages")

        agent = Agent(self.model_name, system_prompt=system_prompt, output_type=_Items)
        result = await agent.run(user_prompt)
        return result.output.items


class LocalModelBackend(GenerationBackend):
    """Generate with the probed open model itself (Issue #2 — self-generation).

    Runs the (synchronous) :meth:`ProbedModel.generate` in a thread so it still
    fits the async generator orchestration. Concurrency is 1 (a single model on
    the GPU).
    """

    default_concurrency = 1

    def __init__(self, model: "ProbedModel", max_new_tokens: int = 320, temperature: float = 1.0):
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        import asyncio

        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        return await asyncio.to_thread(
            self.model.generate,
            prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            chat=True,
        )

    async def generate_list(self, system_prompt: str, user_prompt: str, n: int) -> list[str]:
        """Best-effort list generation from a local model.

        Tries to parse a JSON array; if the model didn't produce clean JSON,
        falls back to splitting on blank lines. For self-generation prefer
        ``one_story_per_call`` (text mode), which avoids this entirely.
        """
        import json

        raw = await self.generate_text(system_prompt, user_prompt)
        try:
            start, end = raw.index("["), raw.rindex("]") + 1
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                return [str(x) for x in parsed][:n]
        except (ValueError, json.JSONDecodeError):
            pass
        chunks = [c.strip() for c in raw.split("\n\n") if c.strip()]
        return chunks[:n]


def build_backend(config, probed_model: "ProbedModel | None" = None, model_name: str | None = None) -> GenerationBackend:
    """Choose a backend from the config.

    ``config.self_generate`` selects :class:`LocalModelBackend` (requires a loaded
    ``probed_model``); otherwise :class:`ApiBackend` with ``model_name`` (defaults
    to ``config.api_generator_model``).
    """
    if config.self_generate:
        if probed_model is None:
            raise ValueError("self_generate=True requires a loaded ProbedModel")
        return LocalModelBackend(probed_model)
    return ApiBackend(model_name or config.api_generator_model)
