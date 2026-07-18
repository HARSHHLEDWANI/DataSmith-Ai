

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class StepResult(BaseModel):
    step: int
    tool: str
    reasoning: str = ""
    input_summary: str = ""
    output: str = ""
    status: str = "pending"  
    duration_ms: int = 0
    error: str | None = None


class PlanStep(BaseModel):
    step: int
    tool: str
    input_from: str = "context"  
    reasoning: str = ""


class Plan(BaseModel):
    needs_clarification: bool = False
    clarifying_question: str | None = None
    plan: list[PlanStep] = Field(default_factory=list)


class CostEstimate(BaseModel):
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_usd: float = 0.0
    note: str = "Heuristic estimate (chars/4); actual usage may differ."


class RunState(BaseModel):
    run_id: str
    status: str = "planning" 
    current_step: int = 0
    total_steps: int = 0
    steps: list[StepResult] = Field(default_factory=list)
    final_answer: str = ""
    clarifying_question: str | None = None
    cost: CostEstimate | None = None
    extracted_inputs: list[dict[str, Any]] = Field(default_factory=list)
    detected_references: list[dict[str, Any]] = Field(default_factory=list)
    input_manifest: str = ""
    error: str | None = None


class StepTimer:


    def __init__(self, step: StepResult) -> None:
        self.step = step
        self._start = 0.0

    def __enter__(self) -> "StepTimer":
        self._start = time.perf_counter()
        self.step.status = "running"
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.step.duration_ms = int((time.perf_counter() - self._start) * 1000)
        if exc_type is not None:
            self.step.status = "failure"
            self.step.error = str(exc)
            return False  # let executor handle retry/degrade
        if self.step.status == "running":
            self.step.status = "success"
        return False
