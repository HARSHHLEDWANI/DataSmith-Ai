from __future__ import annotations

from dataclasses import dataclass, field

from app.agent.trace import RunState
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
conversation_store = ConversationStore()
