from abc import ABC
from functools import cached_property
from typing import AsyncIterator, Callable

from pydantic_ai import Agent
from pydantic_ai.messages import AgentStreamEvent, ModelMessage
from pydantic_ai.run import AgentRunResultEvent


class BaseAgent(ABC):
    model: str
    system_prompt: str
    output_type: type | None = str

    def __init__(self) -> None:
        """Initialise message history. Tracks messages per instance across multiple run() calls."""
        self.message_history: list[ModelMessage] = []

    @cached_property
    def pydantic_agent(self) -> Agent:
        """Create the underlying Pydantic AI agent. Built once on first access."""
        return Agent(self.model, deps_type=BaseAgent, system_prompt=self.system_prompt, output_type=self.output_type)

    async def stream(self, user_prompt: str) -> AsyncIterator[AgentStreamEvent]:
        """Stream events from the agent. Yields stream events. Appends new messages to history on completion."""
        async for event in self.pydantic_agent.run_stream_events(
            user_prompt, deps=self, message_history=self.message_history
        ):
            yield event
            if isinstance(event, AgentRunResultEvent):
                self.message_history.extend(event.result.new_messages())

    async def run(self, user_prompt: str):
        """Run the agent and return the final output."""
        result = await self.pydantic_agent.run(user_prompt, deps=self, message_history=self.message_history)
        self.message_history.extend(result.new_messages())
        return result.output


class ToolAgent(BaseAgent):
    tools: list[Callable]

    def __init_subclass__(cls, **kwargs) -> None:
        """Give each subclass its own tool list so @tool decorators don't leak between agent types."""
        super().__init_subclass__(**kwargs)
        cls.tools = []

    @classmethod
    def tool(cls, func: Callable) -> Callable:
        """Register a function as a tool on this agent class."""
        cls.tools.append(func)
        return func

    @cached_property
    def pydantic_agent(self) -> Agent:
        """Extend the base agent with registered tools."""
        agent = super().pydantic_agent
        for tool in self.tools:
            agent.tool(tool)
        return agent
