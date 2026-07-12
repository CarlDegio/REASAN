"""Batch-capable loader for Unitree RL Lab 2.1 G1 RSL-RL checkpoints."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class UnitreeG1LocoAdapter(nn.Module):
    """Frozen deterministic actor extracted from an RSL-RL ``model_*.pt`` file."""

    observation_dim = 480
    action_dim = 29

    def __init__(self, checkpoint: str | Path, device: str | torch.device):
        super().__init__()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"Unitree G1 checkpoint does not exist: {self.checkpoint}")

        self.actor = nn.Sequential(
            nn.Linear(480, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if "model_state_dict" not in payload:
            raise KeyError(f"Checkpoint has no model_state_dict: {self.checkpoint}")
        actor_state = {
            key.removeprefix("actor."): value
            for key, value in payload["model_state_dict"].items()
            if key.startswith("actor.")
        }
        self.actor.load_state_dict(actor_state, strict=True)
        self.to(device).eval()
        self.requires_grad_(False)

    @torch.inference_mode()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2 or observations.shape[1] != self.observation_dim:
            raise ValueError(f"Expected loco observations [N, 480], got {tuple(observations.shape)}")
        return self.actor(observations)

