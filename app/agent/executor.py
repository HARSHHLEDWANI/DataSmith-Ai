from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from app.agent.trace import Plan, PlanStep, RunState, StepResult, StepTimer
from app.ingestion.models import ExtractedInput
from app.llm.client import LLMNotConfiguredError, llm_client
from app.logging_config import get_logger
from app.tools.registry import ToolContext, ToolRegistry, truncate_for_llm

logger = get_logger("agent.executor")

MAX_STEPS = 6

ProgressCb = Callable[[RunState], Awaitable[None]] | None

_SYNTHESIS_SYSTEM = (
    "You merge the results of several analysis steps, each derived from a distinct "
    "source, into ONE coherent answer to the user's goal. Structure the answer with "
    "explicit attribution using the provided source labels, e.g.:\n"
    "From the PDF (report.pdf): ...\n"
    "From the linked video: ...\n"
    "From the image (chart.png): ...\n"
    "Then finish with a short 'Across sources:' paragraph that connects or contrasts "
    "the sources with respect to the user's goal. Use ONLY facts present in the step "
    "outputs. Output plain text only."
)


def _summarize_io(text: str, limit: int = 280) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


async def execute_plan(
    plan: Plan,
    query: str,
    inputs: list[ExtractedInput],
    registry: ToolRegistry,
    state: RunState,
    progress: ProgressCb = None,
) -> RunState:

    steps = plan.plan[:MAX_STEPS]
    state.total_steps = len(steps)
    state.status = "executing"
    outputs_by_step: dict[int, str] = {}

    for spec in steps:
        result = StepResult(step=spec.step, tool=spec.tool, reasoning=spec.reasoning, status="running")
        state.current_step = spec.step
        state.steps.append(result)
        if progress:
            await progress(state)

        tool = registry.get(spec.tool)
        if tool is None:
            result.status = "skipped"
            result.error = f"Unknown tool '{spec.tool}'."
            if progress:
                await progress(state)
            continue

        upstream = ""
        if spec.input_from.startswith("step:"):
            try:
                ref = int(spec.input_from.split(":", 1)[1])
                upstream = outputs_by_step.get(ref, "")
            except ValueError:
                upstream = ""
        params = {"instructions": spec.instructions} if spec.instructions else {}
        ctx = ToolContext(inputs=inputs, query=query, upstream=upstream, params=params)
        result.input_summary = _summarize_io(upstream or ctx.combined_text() or query, limit=200)

        output = await _run_with_retry(tool.func, ctx, result)
        outputs_by_step[spec.step] = output
        result.output = output
        if progress:
            await progress(state)

    terminals = _terminal_results(steps, state, outputs_by_step)
    if len(terminals) >= 2:
        state.final_answer = await _synthesize(terminals, steps, query, inputs, state, progress)
    else:
        state.final_answer = _pick_final_answer(state, outputs_by_step)
    state.status = "done"
    if progress:
        await progress(state)
    return state


def _terminal_results(
    specs: list[PlanStep], state: RunState, outputs: dict[int, str]
) -> list[StepResult]:
    """Successful steps whose output is not consumed by a later step — i.e. the
    leaves of the plan. Linear chains have exactly one, so single-source runs
    keep the original final-answer behavior."""
    consumed: set[int] = set()
    for spec in specs:
        if spec.input_from.startswith("step:"):
            try:
                consumed.add(int(spec.input_from.split(":", 1)[1]))
            except ValueError:
                pass
    return [
        r
        for r in state.steps
        if r.status == "success"
        and (outputs.get(r.step) or "").strip()
        and r.step not in consumed
    ]


def _provenance_label(
    step_num: int, spec_by_step: dict[int, PlanStep], inputs: list[ExtractedInput]
) -> str:
    """Human label for where a terminal step's content came from, by walking its
    input_from chain back to the root."""
    tools_in_chain: list[str] = []
    root_input = "context"
    seen: set[int] = set()
    cur = step_num
    while cur in spec_by_step and cur not in seen:
        seen.add(cur)
        spec = spec_by_step[cur]
        tools_in_chain.append(spec.tool)
        if spec.input_from.startswith("step:"):
            try:
                cur = int(spec.input_from.split(":", 1)[1])
                continue
            except ValueError:
                break
        root_input = spec.input_from
        break

    if "youtube_transcript" in tools_in_chain:
        return "the linked video"
    if root_input == "query":
        return "your message"
    files = [i for i in inputs if i.source != "user_query" and i.ok and i.text]
    if len(files) == 1:
        return f"the {files[0].type} ({files[0].source})"
    if files:
        names = ", ".join(f.source for f in files[:4])
        return f"the uploaded files ({names})"
    return "your message"


async def _synthesize(
    terminals: list[StepResult],
    specs: list[PlanStep],
    query: str,
    inputs: list[ExtractedInput],
    state: RunState,
    progress: ProgressCb = None,
) -> str:
    spec_by_step = {s.step: s for s in specs}
    labelled: list[tuple[str, str]] = []
    used: set[str] = set()
    for res in terminals:
        label = _provenance_label(res.step, spec_by_step, inputs)
        if label in used:
            label = f"{label} (step {res.step}: {res.tool})"
        used.add(label)
        labelled.append((label, res.output))

    result = StepResult(
        step=max(r.step for r in state.steps) + 1 if state.steps else 1,
        tool="synthesize",
        reasoning="Merge multi-source results into one answer with per-source attribution.",
        status="running",
        input_summary=_summarize_io("; ".join(label for label, _ in labelled), limit=200),
    )
    state.steps.append(result)
    state.total_steps += 1
    state.current_step = result.step
    if progress:
        await progress(state)

    blocks = "\n\n".join(f"[{label}]\n{truncate_for_llm(text)}" for label, text in labelled)
    messages = [
        {"role": "system", "content": _SYNTHESIS_SYSTEM},
        {
            "role": "user",
            "content": f"User goal: {query or '(none provided)'}\n\nStep results by source:\n\n{blocks}",
        },
    ]
    try:
        with StepTimer(result):
            answer = await llm_client.chat(messages, temperature=0.2, max_tokens=900)
        result.status = "success"
        result.output = answer
    except Exception as exc:
        logger.warning("synthesis failed, degrading to labeled blocks", extra={"data": {"error": str(exc)}})
        result.status = "failure"
        result.error = str(exc)
        answer = "\n\n".join(f"From {label}:\n{text}" for label, text in labelled)
        result.output = answer
    if progress:
        await progress(state)
    return answer


async def _run_with_retry(
    func: Callable[[ToolContext], Awaitable[str]],
    ctx: ToolContext,
    result: StepResult,
) -> str:
    for attempt in range(2):
        try:
            with StepTimer(result):
                output = await func(ctx)
            result.status = "success"
            return output
        except LLMNotConfiguredError as exc:
            result.status = "failure"
            result.error = str(exc)
            return f"[Step failed: {exc}]"
        except Exception as exc:
            logger.warning(
                "step failed",
                extra={"data": {"tool": result.tool, "attempt": attempt, "error": str(exc)}},
            )
            result.status = "failure"
            result.error = str(exc)
            if attempt == 0:
                await asyncio.sleep(0.5)
    return f"[Step '{result.tool}' failed after retry: {result.error}]"


def _pick_final_answer(state: RunState, outputs: dict[int, str]) -> str:
    for step in reversed(state.steps):
        if step.status == "success" and step.output.strip():
            return step.output
    if outputs:
        last = outputs[max(outputs)]
        if last.strip():
            return last
    return "Couldn't complete the request — all steps failed. See the plan trace."
