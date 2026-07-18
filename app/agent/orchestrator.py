from __future__ import annotations

import uuid

from app.agent.cost import estimate_cost
from app.agent.executor import ProgressCb, execute_plan
from app.agent.planner import make_plan
from app.agent.session_store import (
    PendingSession,
    conversation_store,
    run_store,
    session_store,
)
from app.agent.trace import RunState
from app.config import settings
from app.ingestion.models import ExtractedInput
from app.ingestion.references import describe_manifest, detect_references
from app.logging_config import get_logger
from app.tools.registry import ToolRegistry, register_all_tools

logger = get_logger("agent.orchestrator")


def _carry_forward(
    prior: list[ExtractedInput], current: list[ExtractedInput]
) -> list[ExtractedInput]:
    seen = {(i.source, i.text[:80]) for i in current}
    carried: list[ExtractedInput] = []
    for inp in prior:
        key = (inp.source, inp.text[:80])
        if key in seen or not inp.text:
            continue
        seen.add(key)
        label = inp.source if inp.source.endswith("(earlier)") else f"{inp.source} (earlier)"
        carried.append(inp.model_copy(update={"source": label}))
    return carried


def _trim_to_budget(inputs: list[ExtractedInput], budget: int) -> list[ExtractedInput]:
    kept: list[ExtractedInput] = []
    total = 0
    for inp in reversed(inputs):
        total += len(inp.text)
        if total > budget and kept:
            break
        kept.append(inp)
    return list(reversed(kept))


class Orchestrator:

    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or register_all_tools()

    async def run(
        self,
        *,
        conversation_id: str,
        query: str,
        inputs: list[ExtractedInput],
        run_id: str | None = None,
        progress: ProgressCb = None,
    ) -> RunState:
        run_id = run_id or uuid.uuid4().hex
        state = run_store.get(run_id) or run_store.create(run_id)
        state.extracted_inputs = [i.model_dump() for i in inputs]

        history: list[str] = []
        pending = session_store.pop_pending(conversation_id)
        if pending is not None:
            history = pending.history + [
                f"Assistant asked: {pending.clarifying_question}",
                f"User answered: {query}",
            ]
            inputs = pending.inputs or inputs
            query = f"{pending.query}\n\n[Clarification from user]: {query}".strip()
            state.extracted_inputs = [i.model_dump() for i in inputs]

        convo = conversation_store.get_or_create(conversation_id)
        carried = _carry_forward(convo.inputs, inputs)
        effective_inputs = carried + inputs
        if convo.transcript:
            history = history + ["Earlier in this conversation:"] + convo.transcript[-6:]

        refs = detect_references(effective_inputs)
        manifest = describe_manifest(effective_inputs, refs)
        state.detected_references = [r.model_dump() for r in refs]
        state.input_manifest = manifest

        state.status = "planning"
        if progress:
            await progress(state)
        plan = await make_plan(query, effective_inputs, self.registry, history, manifest=manifest)

        if plan.needs_clarification and plan.clarifying_question:
            question = plan.clarifying_question
            session_store.set_pending(
                conversation_id,
                PendingSession(query=query, inputs=inputs, clarifying_question=question, history=history),
            )
            state.status = "clarifying"
            state.clarifying_question = question
            if progress:
                await progress(state)
            logger.info("clarification requested", extra={"data": {"q": question}})
            return state

        state.cost = estimate_cost(plan, query, effective_inputs)
        if progress:
            await progress(state)

        await execute_plan(plan, query, effective_inputs, self.registry, state, progress)

        new_inputs = [i for i in inputs if i.ok and i.text]
        convo.inputs = _trim_to_budget(convo.inputs + new_inputs, settings.max_conversation_chars)
        convo.transcript += [f"User: {query[:500]}", f"Assistant: {state.final_answer[:800]}"]
        convo.transcript = convo.transcript[-12:]

        logger.info("run complete", extra={"data": {"run_id": run_id, "steps": len(state.steps)}})
        return state


orchestrator = Orchestrator()
