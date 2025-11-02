from __future__ import annotations
from dataclasses import dataclass

DEFAULT_PRICING = {
    "gpt-5": {"input": 5.00, "output": 15.00},      # USD per 1M tokens
    "gpt-5-mini": {"input": 0.60, "output": 2.50},
}

@dataclass
class CostBreakdown:
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


def estimate_cost(model: str, input_tokens: int, output_tokens: int, pricing: dict, batch_discount: float = 0.0) -> CostBreakdown:
    pi = pricing[model]["input"]
    po = pricing[model]["output"]
    input_cost = (input_tokens / 1_000_000) * pi
    output_cost = (output_tokens / 1_000_000) * po
    total = input_cost + output_cost
    if batch_discount > 0:
        total *= (1.0 - batch_discount)
    return CostBreakdown(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total,
    )