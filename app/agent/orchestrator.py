from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.agent.cost import estimate_cost
from app.agent.executor import ProgressCb, execute_plan
from app.agent.planner import make_plan
from app.agent.plan_ops import to_api_steps
from app.agent.session_store import (
    PendingSession,
    StagedPlan,
    conversation_store,
    plan_store,
    run_store,
    session_store,
)
from app.agent.trace import CostEstimate, Plan, RunState
from app.config import settings
from app.ingestion.models import ExtractedInput
from app.ingestion.references import describe_manifest, detect_references
from app.logging_config import get_logger
from app.tools.registry import ToolRegistry, register_all_tools

logger = get_logger("agent.orchestrator")


class PlanNotFoundError(RuntimeError):
    """A plan_id was submitted to /execute that is unknown or has been evicted."""


@dataclass
class PreparedPlan:
    """Result of the planning stage: either a ready-to-run plan (with plan_id) or
    a clarifying question."""

    conversation_id: str
    plan_id: str | None = None
    steps: list[dict] = field(default_factory=list)
    detected_refs: list[dict] = field(default_factory=list)
    manifest: str = ""
    cost: CostEstimate | None = None
    clarifying_question: str | None = None


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

    async def prepare_plan(
        self,
        *,
        conversation_id: str,
        query: str,
        inputs: list[ExtractedInput],
        replan_notes: str | None = None,
    ) -> PreparedPlan:
        """Planning stage of the two-stage flow: assemble context, detect embedded
        references, run the planner, and stage the plan for later execution. Runs
        NO tools. Returns either an editable plan (with plan_id) or a clarifying
        question."""
        conversation_id = conversation_id or uuid.uuid4().hex

        history: list[str] = []
        pending = session_store.pop_pending(conversation_id)
        raw_new_inputs = inputs
        if pending is not None:
            history = pending.history + [
                f"Assistant asked: {pending.clarifying_question}",
                f"User answered: {query}",
            ]
            raw_new_inputs = pending.inputs or inputs
            query = f"{pending.query}\n\n[Clarification from user]: {query}".strip()

        convo = conversation_store.get_or_create(conversation_id)
        carried = _carry_forward(convo.inputs, raw_new_inputs)
        effective_inputs = carried + raw_new_inputs
        if convo.transcript:
            history = history + ["Earlier in this conversation:"] + convo.transcript[-6:]

        refs = detect_references(effective_inputs)
        manifest = describe_manifest(effective_inputs, refs)
        detected = [r.model_dump() for r in refs]

        plan = await make_plan(
            query, effective_inputs, self.registry, history,
            manifest=manifest, replan_notes=replan_notes,
        )

        if plan.needs_clarification and plan.clarifying_question:
            session_store.set_pending(
                conversation_id,
                PendingSession(
                    query=query, inputs=raw_new_inputs,
                    clarifying_question=plan.clarifying_question, history=history,
                ),
            )
            logger.info("clarification requested", extra={"data": {"q": plan.clarifying_question}})
            return PreparedPlan(
                conversation_id=conversation_id,
                detected_refs=detected,
                manifest=manifest,
                clarifying_question=plan.clarifying_question,
            )

        cost = estimate_cost(plan, query, effective_inputs)
        plan_id = uuid.uuid4().hex
        plan_store.put(
            StagedPlan(
                plan_id=plan_id,
                conversation_id=conversation_id,
                query=query,
                inputs=effective_inputs,
                new_inputs=raw_new_inputs,
                manifest=manifest,
                detected_refs=detected,
                history=history,
                plan=plan,
            )
        )
        logger.info(
            "plan staged",
            extra={"data": {"plan_id": plan_id, "steps": [s.tool for s in plan.plan]}},
        )
        return PreparedPlan(
            conversation_id=conversation_id,
            plan_id=plan_id,
            steps=to_api_steps(plan, self.registry),
            detected_refs=detected,
            manifest=manifest,
            cost=cost,
        )

    async def execute_staged(
        self,
        *,
        run_id: str,
        plan_id: str,
        plan: Plan,
        progress: ProgressCb = None,
    ) -> RunState:
        """Execution stage: run a (possibly user-edited) plan against the context
        staged under plan_id, streaming progress via the callback."""
        staged = plan_store.get(plan_id)
        if staged is None:
            raise PlanNotFoundError(plan_id)

        state = run_store.get(run_id) or run_store.create(run_id)
        state.extracted_inputs = [i.model_dump() for i in staged.new_inputs]
        state.detected_references = staged.detected_refs
        state.input_manifest = staged.manifest
        state.cost = estimate_cost(plan, staged.query, staged.inputs)
        if progress:
            await progress(state)

        await execute_plan(plan, staged.query, staged.inputs, self.registry, state, progress)

        convo = conversation_store.get_or_create(staged.conversation_id)
        new_inputs = [i for i in staged.new_inputs if i.ok and i.text]
        convo.inputs = _trim_to_budget(convo.inputs + new_inputs, settings.max_conversation_chars)
        convo.transcript += [f"User: {staged.query[:500]}", f"Assistant: {state.final_answer[:800]}"]
        convo.transcript = convo.transcript[-12:]

        logger.info("staged run complete", extra={"data": {"run_id": run_id, "steps": len(state.steps)}})
        return state

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
