# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a REASAN-trained G1 PPO checkpoint and stream ten episodes to one MP4.

Run from ``training/``. The environment stage must match the stage used to train
the checkpoint.

First-stage checkpoint::

    CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n env_reasan \
        python tests/manual_g1_loco_ppo_video.py \
        --checkpoint logs/rsl_rl/g1_loco/<first_stage_run>/model_<iteration>.pt \
        --device cuda:0

Second-stage checkpoint::

    CUDA_VISIBLE_DEVICES=0,1 conda run --no-capture-output -n env_reasan \
        python tests/manual_g1_loco_ppo_video.py \
        --checkpoint logs/rsl_rl/g1_loco/<second_stage_run>/model_<iteration>.pt \
        --second_stage \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("WANDB_MODE", "disabled")
# This script is always headless. A stale DISPLAY can make Kit select GLX and
# abort with GLXBadFBConfig before its Vulkan renderer is initialized.
os.environ.pop("DISPLAY", None)
os.environ.pop("WAYLAND_DISPLAY", None)

TRAINING_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TRAINING_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

from reasan_kit_args import apply_default_reasan_kit_args  # noqa: E402


parser = argparse.ArgumentParser(description="Record ten G1 loco PPO evaluation episodes into one headless video.")
parser.add_argument("--checkpoint", type=Path, default=None, help="Path to a REASAN RSL-RL PPO model_*.pt file.")
parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record sequentially.")
parser.add_argument("--second_stage", action="store_true", help="Use the G1 second-stage environment settings.")
parser.add_argument("--seed", type=int, default=0, help="Environment seed.")
parser.add_argument("--output", type=Path, default=None, help="Output MP4 path.")
parser.add_argument("--video_width", type=int, default=1280, help="Output video width in pixels.")
parser.add_argument("--video_height", type=int, default=720, help="Output video height in pixels.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric in the environment.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.checkpoint is None:
    parser.error("--checkpoint is required")
args_cli.checkpoint = args_cli.checkpoint.expanduser().resolve()
if not args_cli.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args_cli.checkpoint}")
if args_cli.checkpoint.suffix != ".pt":
    parser.error(f"checkpoint must be an RSL-RL .pt file: {args_cli.checkpoint}")
if args_cli.episodes <= 0:
    parser.error("--episodes must be greater than zero")
if args_cli.video_width <= 0 or args_cli.video_height <= 0:
    parser.error("--video_width and --video_height must be greater than zero")

args_cli.headless = True
args_cli.enable_cameras = True
apply_default_reasan_kit_args(args_cli, __file__)

os.chdir(TRAINING_DIR)
sys.argv = [sys.argv[0]] + hydra_args

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

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import go2_lidar.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerLoco  # noqa: E402

torch.backends.cudnn.enabled = False

TASK_NAME = "Unitree-G1-Locomotion"


def annotate_frame(frame: np.ndarray, episode_id: int, timestep: int) -> np.ndarray:
    """Convert an Isaac RGB frame to BGR and draw the episode label."""
    if frame is None or frame.size == 0:
        raise RuntimeError("Isaac Lab returned an empty RGB frame.")

    rgb_frame = np.asarray(frame)
    if rgb_frame.ndim != 3 or rgb_frame.shape[2] < 3:
        raise ValueError(f"Expected an RGB frame with shape (H, W, C), got {rgb_frame.shape}.")
    rgb_frame = np.ascontiguousarray(rgb_frame[:, :, :3], dtype=np.uint8)
    bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

    lines = [f"Episode: {episode_id}", f"Timestep: {timestep}"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.65, min(bgr_frame.shape[:2]) / 900.0)
    thickness = max(1, round(font_scale * 2))
    line_gap = max(8, round(font_scale * 10))
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    line_height = max(size[1] for size in text_sizes)
    box_width = max(size[0] for size in text_sizes) + 24
    box_height = len(lines) * line_height + (len(lines) - 1) * line_gap + 24

    cv2.rectangle(bgr_frame, (12, 12), (12 + box_width, 12 + box_height), (0, 0, 0), thickness=-1)
    baseline_y = 24 + line_height
    for line in lines:
        cv2.putText(
            bgr_frame,
            line,
            (24, baseline_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        baseline_y += line_height + line_gap

    return bgr_frame


def create_video_writer(output_path: Path, frame: np.ndarray, fps: int) -> cv2.VideoWriter:
    """Create an MP4 writer whose frame size matches the first rendered frame."""
    frame_height, frame_width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"OpenCV could not create the video writer for {output_path}. "
            "Check that the OpenCV build provides an MP4 codec."
        )
    return writer


def resolve_output_path(checkpoint: Path) -> Path:
    """Resolve the output path without deriving the environment stage from the checkpoint name."""
    if args_cli.output is not None:
        output_path = args_cli.output.expanduser().resolve()
    else:
        stage_name = "second_stage" if args_cli.second_stage else "first_stage"
        output_path = TRAINING_DIR / "tests" / "videos" / f"g1_loco_ppo_{checkpoint.stem}_{stage_name}.mp4"
    if output_path.suffix.lower() != ".mp4":
        raise ValueError(f"Output path must end in .mp4: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def main():
    checkpoint = args_cli.checkpoint
    output_path = resolve_output_path(checkpoint)
    stage_name = "second" if args_cli.second_stage else "first"

    print(f"[INFO] Checkpoint: {checkpoint}")
    print(f"[INFO] Environment stage: {stage_name}")
    print(f"[INFO] Episodes: {args_cli.episodes}")
    print(f"[INFO] Video output: {output_path}")

    env_cfg = parse_env_cfg(
        TASK_NAME,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    env_cfg.is_play_env = True
    env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    if args_cli.second_stage:
        env_cfg.set_second_stage()

    agent_cfg = load_cfg_from_registry(TASK_NAME, "rsl_rl_cfg_entry_point")
    agent_cfg.device = args_cli.device

    gym_env = None
    env = None
    writer = None
    completed_episodes = 0
    try:
        gym_env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
        env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)

        ppo_runner = OnPolicyRunnerLoco(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(checkpoint, load_optimizer=False)
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

        obs, _ = env.reset()
        gym_env.unwrapped.render(recompute=True)

        fps = int(round(1.0 / env.unwrapped.step_dt))
        episode_id = 1
        timestep = 0

        with torch.inference_mode():
            while simulation_app.is_running() and completed_episodes < args_cli.episodes:
                frame = gym_env.unwrapped.render()
                annotated_frame = annotate_frame(frame, episode_id, timestep)
                if writer is None:
                    writer = create_video_writer(output_path, annotated_frame, fps)
                writer.write(annotated_frame)

                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
                ppo_runner.alg.policy.reset(dones)

                if bool(dones[0].item()):
                    completed_episodes += 1
                    print(f"[INFO] Completed episode {completed_episodes}/{args_cli.episodes} at timestep {timestep}.")
                    if completed_episodes < args_cli.episodes:
                        episode_id += 1
                        timestep = 0
                else:
                    timestep += 1

        if completed_episodes != args_cli.episodes:
            raise RuntimeError(
                f"Simulation stopped after {completed_episodes}/{args_cli.episodes} episodes; video is incomplete."
            )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.release()
        if env is not None:
            env.close()
        elif gym_env is not None:
            gym_env.close()
        simulation_app.close()

    print(f"[INFO] Wrote one continuous video: {output_path}")


if __name__ == "__main__":
    main()
