import sys
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from filter_play_control import suppress_output_for_zero_input  # noqa: E402


def test_zero_input_clears_all_policy_and_ema_outputs():
    commands = torch.tensor([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    actions = torch.tensor([[0.2, -0.1, 0.7], [0.3, 0.1, -0.6]])
    previous_actions = torch.tensor([[0.5, 0.4, -0.9], [0.2, -0.2, 0.8]])

    result = suppress_output_for_zero_input(actions, commands, previous_actions)

    torch.testing.assert_close(result, torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.1, -0.6]]))
    torch.testing.assert_close(
        previous_actions, torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.2, 0.8]])
    )


def test_near_zero_input_uses_tolerance():
    commands = torch.tensor([[1.0e-7, -1.0e-7, 1.0e-7], [1.0e-4, 0.0, 0.0]])
    actions = torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    previous_actions = actions.clone()

    result = suppress_output_for_zero_input(actions, commands, previous_actions, atol=1.0e-6)

    assert result[:, 2].tolist() == [0.0, 0.5]


def test_nonzero_input_preserves_small_policy_output():
    commands = torch.tensor([[0.1, 0.0, 0.0]])
    actions = torch.tensor([[0.05, -0.03, 0.02]])
    previous_actions = torch.tensor([[0.2, 0.1, -0.1]])

    result = suppress_output_for_zero_input(actions, commands, previous_actions)

    torch.testing.assert_close(result, actions)
    torch.testing.assert_close(previous_actions, torch.tensor([[0.2, 0.1, -0.1]]))
