from __future__ import annotations

from app.llm.client import llm_client
from app.tools.registry import Tool, ToolContext, apply_instructions, registry

LABELS = {"positive", "negative", "neutral", "mixed"}

_SYSTEM = (
    "You are a sentiment analysis engine. Analyse the sentiment of the text and "
    "reply in EXACTLY this format:\n\n"
    "Sentiment: <positive|negative|neutral|mixed>\n"
    "Confidence: <0-100>%\n"
    "Justification: <one sentence explaining the label>\n\n"
    "Choose the label only from: positive, negative, neutral, mixed."
)


async def sentiment(ctx: ToolContext) -> str:
    text = ctx.primary_text()
    if not text.strip():
        return "Sentiment: neutral\nConfidence: 0%\nJustification: No text provided."
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": apply_instructions(ctx, f"Text:\n\n{text}")},
    ]
    return await llm_client.chat(messages, temperature=0.0, max_tokens=200)


registry.register(
    Tool(
        name="sentiment",
        description=(
            "Analyse the emotional tone of text. Returns a label "
            "(positive/negative/neutral/mixed), a confidence percentage, and a "
            "one-line justification. Use for 'how does this feel / is this "
            "positive / analyse sentiment' requests."
        ),
        input_hint="any text (reviews, comments, transcripts)",
        func=sentiment,
    )
)
