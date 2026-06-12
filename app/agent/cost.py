from __future__ import annotations

from app.agent.trace import CostEstimate, Plan
from app.ingestion.models import ExtractedInput

USD_PER_MILLION_INPUT = 0.59
USD_PER_MILLION_OUTPUT = 0.79
OUTPUT_TOKENS_PER_STEP = 400  


def estimate_cost(plan: Plan, query: str, inputs: list[ExtractedInput]) -> CostEstimate:
    context_chars = sum(len(i.text) for i in inputs if i.ok) + len(query)
    per_step_input_tokens = max(context_chars // 4, 50)
    n_steps = max(len(plan.plan), 1)
    input_tokens = per_step_input_tokens * n_steps
    output_tokens = OUTPUT_TOKENS_PER_STEP * n_steps
    usd = (
        input_tokens / 1_000_000 * USD_PER_MILLION_INPUT
        + output_tokens / 1_000_000 * USD_PER_MILLION_OUTPUT
    )
    return CostEstimate(
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        estimated_usd=round(usd, 6),
    )
