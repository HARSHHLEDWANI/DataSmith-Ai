from __future__ import annotations

from app.llm.client import llm_client
from app.tools.registry import Tool, ToolContext, registry, truncate_for_llm

_SYSTEM = (
    "You are a cross-document analysis engine. You are given several inputs from "
    "different sources (which may be PDFs, images, audio transcripts or text). "
    "Reason ACROSS all of them to answer the user's question. Explicitly compare "
    "and contrast: note where sources agree, disagree, or complement each other. "
    "Reference sources by their [name]. Output plain text only."
)


async def compare(ctx: ToolContext) -> str:
    labelled = []
    for inp in ctx.inputs:
        if inp.ok and inp.text:
            labelled.append(f"=== Source: {inp.source} (type: {inp.type}) ===\n{inp.text}")
    if ctx.upstream.strip():
        labelled.append(f"=== Upstream step output ===\n{ctx.upstream}")

    if len(labelled) < 1:
        return "Cross-input comparison needs at least one readable input."

    body = truncate_for_llm("\n\n".join(labelled))
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": f"User question: {ctx.query}\n\nInputs:\n\n{body}",
        },
    ]
    return await llm_client.chat(messages, temperature=0.3, max_tokens=1000)


registry.register(
    Tool(
        name="compare",
        description=(
            "Reason across MULTIPLE inputs at once to answer a single unified "
            "question (e.g. 'do these two documents discuss the same topic?', "
            "'compare the report and the audio'). Use whenever the goal requires "
            "combining/contrasting more than one input."
        ),
        input_hint="two or more inputs of any type",
        func=compare,
    )
)
