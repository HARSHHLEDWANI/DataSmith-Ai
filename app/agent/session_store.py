from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.agent.trace import Plan, RunState
from app.ingestion.models import ExtractedInput


@dataclass
class PendingSession:

    query: str
    inputs: list[ExtractedInput]
    clarifying_question: str
    history: list[str] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._pending: dict[str, PendingSession] = {}

    def set_pending(self, conversation_id: str, session: PendingSession) -> None:
        self._pending[conversation_id] = session

    def pop_pending(self, conversation_id: str) -> PendingSession | None:
        return self._pending.pop(conversation_id, None)

    def has_pending(self, conversation_id: str) -> bool:
        return conversation_id in self._pending


class RunStore:
    def __init__(self) -> None:
        self._runs: dict[str, RunState] = {}

    def create(self, run_id: str) -> RunState:
        state = RunState(run_id=run_id)
        self._runs[run_id] = state
        return state

    def get(self, run_id: str) -> RunState | None:
        return self._runs.get(run_id)

    def put(self, state: RunState) -> None:
        self._runs[state.run_id] = state


@dataclass
class StagedPlan:
    """A generated plan awaiting user review/edit before execution. Holds the
    ingested context so /execute doesn't need the files re-uploaded."""

    plan_id: str
    conversation_id: str
    query: str
    inputs: list[ExtractedInput]  # effective inputs (carried + new) used for execution
    new_inputs: list[ExtractedInput] = field(default_factory=list)  # this turn's raw uploads
    manifest: str = ""
    detected_refs: list[dict[str, Any]] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    plan: Plan | None = None


class PlanStore:
    """In-memory, capacity-bounded store of staged plans (FIFO eviction).
    Fits the Render free-tier single-instance model like RunStore."""

    def __init__(self, capacity: int = 64) -> None:
        self._plans: OrderedDict[str, StagedPlan] = OrderedDict()
        self._capacity = capacity

    def put(self, staged: StagedPlan) -> None:
        self._plans[staged.plan_id] = staged
        self._plans.move_to_end(staged.plan_id)
        while len(self._plans) > self._capacity:
            self._plans.popitem(last=False)

    def get(self, plan_id: str) -> StagedPlan | None:
        return self._plans.get(plan_id)


@dataclass
class Conversation:

    inputs: list[ExtractedInput] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)


class ConversationStore:

    def __init__(self) -> None:
        self._convos: dict[str, Conversation] = {}

    def get_or_create(self, conversation_id: str) -> Conversation:
        convo = self._convos.get(conversation_id)
        if convo is None:
            convo = Conversation()
            self._convos[conversation_id] = convo
        return convo

    def reset(self) -> None:
        self._convos.clear()


session_store = SessionStore()
run_store = RunStore()
plan_store = PlanStore()
conversation_store = ConversationStore()
