"""A stub chat model: returns queued AIMessages in order and records what it was sent."""
from typing import Callable
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeLLM(BaseChatModel):
    responses: list[AIMessage] = Field(default_factory=list)
    responder: Callable[[list], AIMessage] | None = None  # computes each reply; for parallel branches
    calls: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(messages)
        message = self.responder(messages) if self.responder else self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake"


def tool_call(name: str, args: dict) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid4().hex[:8]}"}])


def page(url: str, text: str, title: str = "Page", published: str | None = None) -> dict:
    """A trimmed search result, shaped like travelex.tools.search returns."""
    return {"url": url, "title": title, "text": text, "published": published,
            "retrieved_at": "2026-10-27T00:00:00+00:00", "kind": "web"}
