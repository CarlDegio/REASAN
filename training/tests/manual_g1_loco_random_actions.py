# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the G1 locomotion environment with random actions for visual smoke testing.

Run from the ``training`` directory, for example:

    CUDA_VISIBLE_DEVICES=0,1 python tests/manual_g1_loco_random_actions.py --num_envs 1 --steps 2000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TRAINING_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

from reasan_kit_args import apply_default_reasan_kit_args  # noqa: E402


parser = argparse.ArgumentParser(description="Run Unitree G1 locomotion with random actions.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=2000, help="Number of env steps to run before exiting.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for env and action sampling.")
parser.add_argument("--action_scale", type=float, default=1.0, help="Multiplier for random actions in [-1, 1].")
parser.add_argument("--hold_steps", type=int, default=10, help="Reuse each random action for this many env steps.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric when parsing env cfg.")
parser.add_argument("--second_stage", action="store_true", default=False, help="Enable the G1 loco second-stage config.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
apply_default_reasan_kit_args(args_cli, __file__)

os.chdir(TRAINING_DIR)
sys.argv = [sys.argv[0]] + hydra_args

local_isaac_asset_root = TRAINING_DIR / "assets" / "omniverse" / "Assets" / "Isaac" / "4.5"
local_g1_usd = (
    TRAINING_DIR
    / "assets"
    / "unitree_model"
    / "G1"
    / "29dof"
    / "usd"
    / "g1_29dof_rev_1_0"
    / "g1_29dof_rev_1_0.usd"
)
if not local_g1_usd.exists():
    raise FileNotFoundError(
        f"G1 29dof USD is missing: {local_g1_usd}\n"
        "Run from training/: python assets/download_reasan_assets.py"
    )

print(f"[INFO] Working directory: {Path.cwd()}")
print(f"[INFO] Local Isaac asset root: {local_isaac_asset_root}")
print(f"[INFO] Local G1 USD: {local_g1_usd}")
print(f"[INFO] Kit args: {args_cli.kit_args}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

torch.backends.cudnn.enabled = False

TASK_NAME = "Unitree-G1-Locomotion"


def main():
    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.is_play_env = True
    if args_cli.second_stage:
        env_cfg.set_second_stage()

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode=None)
    obs, _ = env.reset()
    del obs

    action_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
    print(f"[INFO] Running {TASK_NAME} with random actions: num_envs={env.unwrapped.num_envs}, action_dim={action_dim}")

    step = 0
    actions = torch.zeros(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device)
    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.steps:
            if step % args_cli.hold_steps == 0:
                actions = torch.rand(env.unwrapped.num_envs, action_dim, device=env.unwrapped.device) * 2.0 - 1.0
                actions *= args_cli.action_scale
            env.step(actions)
            step += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
