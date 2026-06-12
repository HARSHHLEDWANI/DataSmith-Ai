from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.config import settings
from app.ingestion.models import ExtractedInput


def truncate_for_llm(text: str, budget: int | None = None) -> str:
    budget = budget or settings.max_tool_input_chars
    if len(text) <= budget:
        return text
    head = text[: int(budget * 0.7)]
    tail = text[-int(budget * 0.25) :]
    return f"{head}\n...[truncated {len(text) - budget} chars]...\n{tail}"


@dataclass
class ToolContext:

    inputs: list[ExtractedInput]
    query: str
    upstream: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def combined_text(self) -> str:
        parts = [f"[{i.source}]\n{i.text}" for i in self.inputs if i.ok and i.text]
        return truncate_for_llm("\n\n".join(parts))

    def primary_text(self) -> str:
        if self.upstream.strip():
            return truncate_for_llm(self.upstream)
        combined = self.combined_text()
        return combined if combined.strip() else self.query


ToolFn = Callable[[ToolContext], Awaitable[str]]


@dataclass
class Tool:
    name: str
    description: str
    input_hint: str
    func: ToolFn


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def describe_for_planner(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description} (input: {tool.input_hint})")
        return "\n".join(lines)

    def as_list(self) -> list[dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "input_hint": t.input_hint}
            for t in self._tools.values()
        ]


registry = ToolRegistry()


def register_all_tools() -> ToolRegistry:
    import app.tools.code_explain
    import app.tools.compare
    import app.tools.qa
    import app.tools.sentiment
    import app.tools.summarize
    import app.tools.youtube

    _ = (
        app.tools.code_explain,
        app.tools.compare,
        app.tools.qa,
        app.tools.sentiment,
        app.tools.summarize,
        app.tools.youtube,
    )
    return registry
