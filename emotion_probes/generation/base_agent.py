"""
Thin base classes over `pydantic-ai` for the external-API generation backend.

Moved verbatim (in spirit) from the original top-level ``agent.py`` so the
dataset-generation code keeps working unchanged; it now lives inside the
``generation`` package next to the code that uses it.
"""

from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import AsyncIterator, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.run import AgentRunResultEvent


class BaseAgent(ABC):
    """A minimal stateful wrapper around a `pydantic-ai` Agent.

    Subclasses set ``model``, ``system_prompt``, and (optionally) ``output_type``
    (e.g. a Pydantic model for structured output). Each instance keeps its own
    message history across ``run`` calls.
    """

    model: str
    system_prompt: str
    output_type: type | None = str

    def __init__(self) -> None:
        self.message_history: list[ModelMessage] = []

    @cached_property
    def pydantic_agent(self) -> Agent:
        """Build the underlying agent once, on first use."""
        return Agent(
            self.model,
            deps_type=BaseAgent,
            system_prompt=self.system_prompt,
            output_type=self.output_type,
        )

    async def stream(self, user_prompt: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream events; append new messages to history when the run completes."""
        async for event in self.pydantic_agent.run_stream_events(
            user_prompt, deps=self, message_history=self.message_history
        ):
            yield event
            if isinstance(event, AgentRunResultEvent):
                self.message_history.extend(event.result.new_messages())

    async def run(self, user_prompt: str):
        """Run once and return the final (possibly structured) output."""
        result = await self.pydantic_agent.run(
            user_prompt, deps=self, message_history=self.message_history
        )
        self.message_history.extend(result.new_messages())
        return result.output


class ToolAgent(BaseAgent):
    """A :class:`BaseAgent` that can register tool functions."""

    tools: list[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.tools = []

    @classmethod
    def tool(cls, func: Callable) -> Callable:
        cls.tools.append(func)
        return func

    @cached_property
    def pydantic_agent(self) -> Agent:
        agent = super().pydantic_agent
        for tool in self.tools:
            agent.tool(tool)
        return agent
