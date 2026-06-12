from __future__ import annotations

from app.llm.client import llm_client
from app.tools.registry import Tool, ToolContext, registry

_SYSTEM = (
    "You are an expert code reviewer. Analyse the provided code and reply in "
    "EXACTLY this format:\n\n"
    "What it does: <plain-language explanation>\n\n"
    "Bugs / issues detected:\n"
    "- <issue or 'None found'>\n\n"
    "Time complexity: <Big-O with a one-line reason>\n\n"
    "Be specific and concrete; cite variable/function names from the code."
)


async def code_explain(ctx: ToolContext) -> str:
    code = ctx.primary_text()
    if not code.strip():
        return "No code provided to explain."
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Code:\n\n```\n{code}\n```"},
    ]
    return await llm_client.chat(messages, temperature=0.1, max_tokens=800)


registry.register(
    Tool(
        name="code_explain",
        description=(
            "Explain a code snippet: what it does, bugs/issues detected, and "
            "time complexity (Big-O). Use when the input is source code or the "
            "user asks to explain/review/debug code."
        ),
        input_hint="a source-code snippet",
        func=code_explain,
    )
)
