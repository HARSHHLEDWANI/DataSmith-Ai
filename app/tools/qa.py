from __future__ import annotations

from app.llm.client import llm_client
from app.tools.registry import Tool, ToolContext, apply_instructions, registry

_SYSTEM = (
    "You are a helpful, concise assistant. Answer the user's question using the "
    "provided context when it is relevant. If the context does not contain the "
    "answer, say so and answer from general knowledge. Be friendly and direct. "
    "If the context contains several distinct sources (multiple files, a video "
    "transcript, a screenshot), attribute your conclusions to the source they "
    "came from (e.g. 'From the PDF: ... From the linked video: ...'). "
    "Output plain text only."
)


async def qa(ctx: ToolContext) -> str:
    context = ctx.primary_text()
    user_block = f"Question: {ctx.query}"
    if context.strip() and context.strip() != ctx.query.strip():
        user_block = f"Context:\n{context}\n\n{user_block}"
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": apply_instructions(ctx, user_block)},
    ]
    return await llm_client.chat(messages, temperature=0.4, max_tokens=900)


registry.register(
    Tool(
        name="qa",
        description=(
            "Answer a question or hold a conversation, grounded in the uploaded "
            "context when relevant. The general-purpose fallback tool; also use "
            "as the final step to answer the user's actual question after other "
            "tools have gathered information."
        ),
        input_hint="the user's question + any context",
        func=qa,
    )
)
