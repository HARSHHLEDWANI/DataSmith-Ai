from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from app.agent.trace import Plan, RunState, StepResult, StepTimer
from app.ingestion.models import ExtractedInput
from app.llm.client import LLMNotConfiguredError
from app.logging_config import get_logger
from app.tools.registry import ToolContext, ToolRegistry

logger = get_logger("agent.executor")

MAX_STEPS = 6

ProgressCb = Callable[[RunState], Awaitable[None]] | None


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
        ctx = ToolContext(inputs=inputs, query=query, upstream=upstream)
        result.input_summary = _summarize_io(upstream or ctx.combined_text() or query, limit=200)

        output = await _run_with_retry(tool.func, ctx, result)
        outputs_by_step[spec.step] = output
        result.output = output
        if progress:
            await progress(state)

    state.final_answer = _pick_final_answer(state, outputs_by_step)
    state.status = "done"
    if progress:
        await progress(state)
    return state


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
