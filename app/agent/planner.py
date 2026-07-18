from __future__ import annotations

from app.agent.trace import Plan
from app.config import settings
from app.ingestion.models import ExtractedInput
from app.llm.client import llm_client
from app.logging_config import get_logger
from app.tools.registry import ToolRegistry

logger = get_logger("agent.planner")

_PLANNER_SYSTEM = """You are the planning brain of triage, an agent built for messy, \
composite input. You decide which tools to run, in what order, to accomplish the \
user's goal across ALL of their inputs (text, images, PDFs, audio transcripts) — \
including references embedded inside those inputs, such as a YouTube link buried \
in a PDF.

AVAILABLE TOOLS:
{tools}

{{
  "needs_clarification": <true|false>,
  "clarifying_question": <string or null>,
  "plan": [
    {{"step": 1, "tool": "<tool_name>", "input_from": "<context|query|step:N>", "reasoning": "<why this tool now>"}}
  ]
}}

Respond ONLY with a valid JSON object — no markdown, no extra text.

RULES:
1. If the goal is ambiguous, missing, or refers to something not present in any \
input, set needs_clarification=true, provide ONE specific clarifying_question, \
and return an empty plan. Do NOT guess between equally-plausible tasks.
2. Otherwise set needs_clarification=false and produce a MINIMAL ordered plan \
(no redundant steps). Use only tool names from the list above.
3. Chain steps with input_from="step:N" when a tool should consume a previous \
step's output (e.g. fetch a transcript, then summarize step:1).
4. "input_from":"context" means operate on the uploaded inputs; "query" means \
the raw user text. Prefer "context" when files are present.
5. The final step should produce the user-facing answer (often qa, summarize, or compare).
6. Keep plans short: 1-4 steps is typical. Never exceed 6 steps.
7. A "Detected inputs" manifest may list references (YouTube links, URLs) found \
INSIDE uploads. Treat each embedded reference as an input in its own right: if the \
goal involves it, plan a step to resolve it (e.g. youtube_transcript) BEFORE the \
analysis steps that need its content.

EXAMPLES:

User goal: "Here is a file." (a PDF was uploaded, no actual task)
Response: {{"needs_clarification": true, "clarifying_question": "I've read your PDF. What would you like me to do with it — summarize it, answer a question, or something else?", "plan": []}}

User goal: "Hit the YouTube link inside this PDF and summarize the video."
(inputs: report.pdf)
Response: {{"needs_clarification": false, "clarifying_question": null, "plan": [
  {{"step": 1, "tool": "youtube_transcript", "input_from": "context", "reasoning": "Detect the YouTube URL inside the PDF text and fetch its transcript."}},
  {{"step": 2, "tool": "summarize", "input_from": "step:1", "reasoning": "Summarize the fetched transcript in the required 3-part format."}}
]}}

User goal: "Do these two documents discuss the same topic?" (inputs: a.pdf, b.pdf)
Response: {{"needs_clarification": false, "clarifying_question": null, "plan": [
  {{"step": 1, "tool": "compare", "input_from": "context", "reasoning": "Reason across both documents to determine topical overlap."}}
]}}
"""


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    head = text[: int(budget * 0.7)]
    tail = text[-int(budget * 0.25) :]
    return f"{head}\n...[truncated {len(text) - budget} chars]...\n{tail}"


def build_context_digest(inputs: list[ExtractedInput]) -> str:
    if not inputs:
        return "(no files uploaded)"
    lines = []
    for inp in inputs:
        if inp.error:
            lines.append(f"- {inp.source} ({inp.type}): ERROR — {inp.error}")
            continue
        preview = _truncate(inp.text, settings.max_context_chars)
        meta_bits = []
        if "ocr_confidence" in inp.meta:
            meta_bits.append(f"ocr_conf={inp.meta['ocr_confidence']}")
        if "duration_seconds" in inp.meta:
            meta_bits.append(f"duration={inp.meta['duration_seconds']}s")
        if inp.meta.get("ocr_pages"):
            meta_bits.append(f"ocr_pages={inp.meta['ocr_pages']}")
        meta_str = f" [{', '.join(meta_bits)}]" if meta_bits else ""
        lines.append(f"- {inp.source} ({inp.type}){meta_str}:\n{preview}")
    return "\n".join(lines)


async def make_plan(
    query: str,
    inputs: list[ExtractedInput],
    registry: ToolRegistry,
    clarification_history: list[str] | None = None,
    manifest: str | None = None,
) -> Plan:
    digest = build_context_digest(inputs)
    system = _PLANNER_SYSTEM.format(tools=registry.describe_for_planner())

    user_parts = [f"User goal: {query or '(no text query provided)'}"]
    if manifest:
        user_parts.append(f"\n{manifest}")
    user_parts.append(f"\nUploaded inputs:\n{digest}")
    if clarification_history:
        user_parts.append(
            "\nEarlier clarification exchange (already resolved, use it):\n"
            + "\n".join(clarification_history)
        )
    user = "\n".join(user_parts)

    logger.info("planning", extra={"data": {"query": query[:200], "n_inputs": len(inputs)}})

    raw = await llm_client.chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=900,
    )

    plan = Plan.model_validate(raw)
    valid_names = set(registry.names())
    plan.plan = [s for s in plan.plan if s.tool in valid_names]
    if not plan.needs_clarification and not plan.plan:
        from app.agent.trace import PlanStep
        plan.plan = [PlanStep(step=1, tool="qa", input_from="context", reasoning="Fallback: answer directly.")]

    logger.info(
        "plan ready",
        extra={"data": {"needs_clarification": plan.needs_clarification, "steps": [s.tool for s in plan.plan]}},
    )
    return plan
