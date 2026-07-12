"""Smoke-test the G1 filter environment skeleton with persistent visualization."""

import argparse

from isaaclab.app import AppLauncher

from reasan_kit_args import apply_default_reasan_kit_args

parser = argparse.ArgumentParser(description="Play the Unitree G1 filter environment skeleton.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="Zero runs until the window is closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
apply_default_reasan_kit_args(args_cli, __file__)
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main():
    cfg = parse_env_cfg("Unitree-G1-Filter", device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make("Unitree-G1-Filter", cfg=cfg)
    env.reset()
    step = 0
    while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
        with torch.inference_mode():
            actions = 2.0 * torch.rand(args_cli.num_envs, 3, device=env.unwrapped.device) - 1.0
            env.step(actions)
        step += 1
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()

