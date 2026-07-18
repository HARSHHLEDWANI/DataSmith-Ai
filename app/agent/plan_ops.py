"""Translate between the internal `Plan` and the editable API plan schema, and
validate/re-sequence a plan that the user has edited in the frontend."""
from __future__ import annotations

import re
from typing import Any

from app.agent.trace import Plan, PlanStep
from app.tools.registry import ToolRegistry


def _describe(step: PlanStep, registry: ToolRegistry) -> str:
    if step.reasoning.strip():
        return step.reasoning.strip()
    tool = registry.get(step.tool)
    return tool.description if tool else step.tool


def to_api_steps(plan: Plan, registry: ToolRegistry) -> list[dict[str, Any]]:
    """Internal Plan -> the editable `steps[]` the frontend renders."""
    steps: list[dict[str, Any]] = []
    for i, step in enumerate(plan.plan, start=1):
        steps.append(
            {
                "id": i,
                "tool": step.tool,
                "description": _describe(step, registry),
                "input_from": step.input_from,
                "instructions": step.instructions,
            }
        )
    return steps


def validate_edited_steps(
    steps: list[dict[str, Any]], registry: ToolRegistry
) -> tuple[Plan, list[str]]:
    """Turn the frontend's edited step array into an executable Plan.

    Order of the array is the execution order. Steps are renumbered 1..N and any
    `input_from: "step:X"` is remapped to X's new position. Unknown tools are
    dropped; dangling/forward dependencies degrade to "context". Never raises."""
    warnings: list[str] = []
    valid_names = set(registry.names())

    # 1. Drop unknown tools, preserving order and recording original ids.
    kept: list[dict[str, Any]] = []
    for raw in steps:
        tool = str(raw.get("tool", "")).strip()
        if tool not in valid_names:
            warnings.append(f"Removed step with unknown tool '{tool or '(blank)'}'.")
            continue
        kept.append(raw)

    # 2. Map each surviving step's original id -> its new 1-based position.
    old_to_new: dict[int, int] = {}
    for new_pos, raw in enumerate(kept, start=1):
        old_id = raw.get("id")
        if isinstance(old_id, int):
            old_to_new[old_id] = new_pos

    # 3. Build the executable steps, remapping/validating dependencies.
    plan_steps: list[PlanStep] = []
    for new_pos, raw in enumerate(kept, start=1):
        input_from = str(raw.get("input_from", "context")).strip() or "context"
        if input_from.startswith("step:"):
            # The planner may emit a single ("step:2") or compound ("step:2, step:3")
            # reference. Resolve to the first prior step that still exists.
            refs_old = [int(n) for n in re.findall(r"\d+", input_from)]
            resolved: int | None = None
            moved_before = False
            for ref_old in refs_old:
                new_ref = old_to_new.get(ref_old)
                if new_ref is None:
                    continue
                if new_ref < new_pos:
                    resolved = new_ref
                    break
                moved_before = True
            if resolved is not None:
                input_from = f"step:{resolved}"
            else:
                if moved_before:
                    warnings.append(
                        f"Step {new_pos} ({raw.get('tool')}) was moved before the step it "
                        f"depended on; it will run on all inputs instead."
                    )
                elif refs_old:
                    warnings.append(
                        f"Step {new_pos} ({raw.get('tool')}) depended on a step that was "
                        f"removed; it will run on all inputs instead."
                    )
                input_from = "context"

        plan_steps.append(
            PlanStep(
                step=new_pos,
                tool=str(raw["tool"]),
                input_from=input_from,
                reasoning=str(raw.get("description", "") or ""),
                instructions=str(raw.get("instructions", "") or ""),
            )
        )

    # 4. Empty plan -> qa fallback (mirrors planner.make_plan).
    if not plan_steps:
        warnings.append("No runnable steps remained; falling back to a single Q&A step.")
        plan_steps = [
            PlanStep(step=1, tool="qa", input_from="context", reasoning="Fallback: answer directly.")
        ]

    return Plan(needs_clarification=False, clarifying_question=None, plan=plan_steps), warnings
