"""Play-only safety transforms for G1 Filter actions."""

from __future__ import annotations

import torch


def suppress_output_for_zero_input(
    actions: torch.Tensor,
    input_commands: torch.Tensor,
    previous_actions: torch.Tensor,
    atol: float = 1.0e-6,
) -> torch.Tensor:
    zero_input = torch.all(torch.abs(input_commands) <= atol, dim=-1)
    if not torch.any(zero_input):
        return actions
    actions = actions.clone()
    actions[zero_input] = 0.0
    previous_actions[zero_input] = 0.0
    return actions
