from __future__ import annotations

from app.llm.client import llm_client
from app.tools.registry import Tool, ToolContext, apply_instructions, registry

_SYSTEM = (
    "You are a precise summarization engine. You ALWAYS reply in EXACTLY this "
    "format and nothing else:\n\n"
    "One-line summary: <a single sentence>\n\n"
    "Key points:\n"
    "- <bullet 1>\n"
    "- <bullet 2>\n"
    "- <bullet 3>\n\n"
    "Detailed summary: <exactly five sentences>\n\n"
    "Use exactly three bullets and exactly five sentences in the detailed summary."
)


async def summarize(ctx: ToolContext) -> str:
    text = ctx.primary_text()
    if not text.strip():
        return "Nothing to summarize: no input text was provided."
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": apply_instructions(ctx, f"Summarize the following content:\n\n{text}"),
        },
    ]
    return await llm_client.chat(messages, temperature=0.2, max_tokens=700)


registry.register(
    Tool(
        name="summarize",
        description=(
            "Summarize text/content into a fixed 3-part format: a one-line "
            "summary, 3 bullet points, and a 5-sentence detailed summary. Use "
            "for any 'summarize/tl;dr/give me the gist' request."
        ),
        input_hint="any text (raw inputs or a previous step's output)",
        func=summarize,
    )
)
