# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the REASAN G1 locomotion env with Unitree RL Lab's exported G1 velocity ONNX policy.

This script mirrors the observation order used by Unitree's task:

    base_ang_vel[history] + projected_gravity[history] + velocity_commands[history]
    + joint_pos_rel[history] + joint_vel_rel[history] + last_action[history]

Run from the ``training`` directory, for example:

    CUDA_VISIBLE_DEVICES=0,1 python tests/manual_g1_loco_unitree_onnx.py --num_envs 1 --steps 2000
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path

TRAINING_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TRAINING_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from isaaclab.app import AppLauncher  # noqa: E402

from reasan_kit_args import apply_default_reasan_kit_args  # noqa: E402


UNITREE_G1_VELOCITY_ONNX = (
    TRAINING_DIR
    / "assets"
    / "unitree_rl_lab"
    / "deploy"
    / "robots"
    / "g1_29dof"
    / "config"
    / "policy"
    / "velocity"
    / "v0"
    / "exported"
    / "policy.onnx"
)
UNITREE_G1_VELOCITY_DEPLOY_CFG = (
    TRAINING_DIR
    / "assets"
    / "unitree_rl_lab"
    / "deploy"
    / "robots"
    / "g1_29dof"
    / "config"
    / "policy"
    / "velocity"
    / "v0"
    / "params"
    / "deploy.yaml"
)

UNITREE_OBS_HISTORY_LENGTH = 5
UNITREE_SINGLE_FRAME_OBS_DIM = 96
UNITREE_OBS_TERM_ORDER = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
)


parser = argparse.ArgumentParser(description="Run REASAN G1 locomotion with Unitree's G1 velocity ONNX policy.")
parser.add_argument("--checkpoint", type=Path, default=UNITREE_G1_VELOCITY_ONNX, help="Unitree G1 velocity ONNX path.")
parser.add_argument("--deploy_cfg", type=Path, default=UNITREE_G1_VELOCITY_DEPLOY_CFG, help="Unitree deploy.yaml path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=2000, help="Number of env steps to run before exiting.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for env.")
parser.add_argument("--onnx_intra_op_threads", type=int, default=1, help="ONNXRuntime intra-op thread count.")
parser.add_argument("--onnx_inter_op_threads", type=int, default=1, help="ONNXRuntime inter-op thread count.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric when parsing env cfg.")
parser.add_argument("--second_stage", action="store_true", default=False, help="Enable the G1 loco second-stage config.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
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

print(f"[INFO] Working directory: {Path.cwd()}")
print(f"[INFO] Local G1 USD: {local_g1_usd}")
print(f"[INFO] Unitree ONNX: {args_cli.checkpoint}")
print(f"[INFO] Unitree deploy cfg: {args_cli.deploy_cfg}")
print(f"[INFO] Kit args: {args_cli.kit_args}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

torch.backends.cudnn.enabled = False

TASK_NAME = "Unitree-G1-Locomotion"


def _shape_from_onnx(value_info) -> list[int | str | None]:
    return [dim.dim_value or dim.dim_param or None for dim in value_info.type.tensor_type.shape.dim]


def _read_onnx_metadata(checkpoint: Path):
    checkpoint = checkpoint.expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Unitree G1 velocity ONNX policy is missing: {checkpoint}\n"
            "Run from training/: python assets/download_reasan_assets.py"
        )
    try:
        import onnx
    except ImportError:
        print("[WARN] onnx is not installed; skipping static ONNX metadata checks.")
        return None

    model = onnx.load(str(checkpoint))
    input_meta = model.graph.input[0]
    output_meta = model.graph.output[0]
    input_shape = _shape_from_onnx(input_meta)
    output_shape = _shape_from_onnx(output_meta)
    print(f"[INFO] ONNX metadata input: name={input_meta.name}, shape={input_shape}")
    print(f"[INFO] ONNX metadata output: name={output_meta.name}, shape={output_shape}")
    return input_meta.name, output_meta.name, input_shape, output_shape


def _load_onnxruntime_session(checkpoint: Path, intra_op_threads: int, inter_op_threads: int):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "Unitree ONNX inference requires onnxruntime. Install onnxruntime or onnxruntime-gpu in env_reasan."
        ) from exc

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = intra_op_threads
    session_options.inter_op_num_threads = inter_op_threads

    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(
        str(checkpoint.expanduser().resolve()),
        sess_options=session_options,
        providers=providers,
    )
    print(f"[INFO] ONNX providers: {session.get_providers()}")
    print(f"[INFO] ONNX thread counts: intra_op={intra_op_threads}, inter_op={inter_op_threads}")
    return session, session.get_inputs()[0].name, session.get_outputs()[0].name


def _load_deploy_cfg(path: Path, action_dim: int):
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Unitree deploy metadata is missing: {path}\n"
            "Run from training/: python assets/download_reasan_assets.py"
        )
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Unitree deploy.yaml checks require PyYAML.") from exc

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    joint_ids_map = cfg.get("joint_ids_map", [])
    if len(joint_ids_map) != action_dim:
        raise RuntimeError(
            f"Unitree deploy.yaml joint_ids_map has {len(joint_ids_map)} joints, but env action_dim={action_dim}."
        )
    print(f"[INFO] Unitree deploy joint_ids_map length: {len(joint_ids_map)}")
    return cfg


def _apply_unitree_deploy_action_scale(env, deploy_cfg: dict):
    action_scale = deploy_cfg["actions"]["JointPositionAction"]["scale"]
    first_scale = float(action_scale[0])
    if any(abs(float(scale) - first_scale) > 1e-6 for scale in action_scale):
        raise RuntimeError("This smoke script expects Unitree deploy.yaml to use a uniform action scale.")
    env.unwrapped.cfg.action_scale = first_scale
    print(f"[INFO] Applied Unitree deploy action scale to REASAN env: {first_scale}")


def _build_unitree_obs_terms(env) -> dict[str, torch.Tensor]:
    unwrapped = env.unwrapped
    controlled_joint_ids = unwrapped._controlled_joint_ids
    robot = unwrapped._robot
    return {
        # Matches Unitree velocity_env_cfg.py PolicyCfg term order and scales.
        "base_ang_vel": robot.data.root_ang_vel_b * 0.2,
        "projected_gravity": robot.data.projected_gravity_b,
        "velocity_commands": torch.cat([unwrapped._cmd_lin_vel, unwrapped._cmd_ang_vel], dim=-1),
        "joint_pos_rel": robot.data.joint_pos[:, controlled_joint_ids] - robot.data.default_joint_pos[:, controlled_joint_ids],
        "joint_vel_rel": robot.data.joint_vel[:, controlled_joint_ids] * 0.05,
        "last_action": unwrapped._actions,
    }


def _init_unitree_history(obs_terms: dict[str, torch.Tensor]) -> dict[str, deque[torch.Tensor]]:
    return {
        term_name: deque([value.clone()] * UNITREE_OBS_HISTORY_LENGTH, maxlen=UNITREE_OBS_HISTORY_LENGTH)
        for term_name, value in obs_terms.items()
    }


def _build_unitree_history_obs(history: dict[str, deque[torch.Tensor]]) -> torch.Tensor:
    # IsaacLab ObservationManager flattens history per term first, then concatenates terms in declaration order.
    flattened_terms = []
    for term_name in UNITREE_OBS_TERM_ORDER:
        term_history = torch.stack(tuple(history[term_name]), dim=1)
        flattened_terms.append(term_history.reshape(term_history.shape[0], -1))
    return torch.cat(flattened_terms, dim=-1)


def _append_unitree_history(history: dict[str, deque[torch.Tensor]], obs_terms: dict[str, torch.Tensor]):
    for term_name in UNITREE_OBS_TERM_ORDER:
        history[term_name].append(obs_terms[term_name].clone())


def _validate_shapes(history_obs: torch.Tensor, action_dim: int, onnx_metadata):
    expected_dim = UNITREE_SINGLE_FRAME_OBS_DIM * UNITREE_OBS_HISTORY_LENGTH
    if history_obs.shape[-1] != expected_dim:
        raise RuntimeError(f"Expected Unitree history obs dim {expected_dim}, got {history_obs.shape[-1]}.")
    if onnx_metadata is None:
        return

    input_shape = onnx_metadata[2]
    output_shape = onnx_metadata[3]
    onnx_batch = input_shape[0] if input_shape else None
    onnx_obs_dim = input_shape[-1] if input_shape else None
    onnx_action_dim = output_shape[-1] if output_shape else None
    if isinstance(onnx_batch, int) and onnx_batch != history_obs.shape[0]:
        raise RuntimeError(
            f"Unitree ONNX fixed batch is {onnx_batch}, but num_envs={history_obs.shape[0]}. Use --num_envs 1."
        )
    if isinstance(onnx_obs_dim, int) and onnx_obs_dim != history_obs.shape[-1]:
        raise RuntimeError(f"Unitree ONNX obs dim is {onnx_obs_dim}, but built history obs dim is {history_obs.shape[-1]}.")
    if isinstance(onnx_action_dim, int) and onnx_action_dim != action_dim:
        raise RuntimeError(f"Unitree ONNX action dim is {onnx_action_dim}, but env action dim is {action_dim}.")


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
    env.reset()

    action_dim = gym.spaces.flatdim(env.unwrapped.single_action_space)
    deploy_cfg = _load_deploy_cfg(args_cli.deploy_cfg, action_dim)
    _apply_unitree_deploy_action_scale(env, deploy_cfg)
    onnx_metadata = _read_onnx_metadata(args_cli.checkpoint)
    session, onnx_input_name, onnx_output_name = _load_onnxruntime_session(
        args_cli.checkpoint,
        args_cli.onnx_intra_op_threads,
        args_cli.onnx_inter_op_threads,
    )

    obs_terms = _build_unitree_obs_terms(env)
    history = _init_unitree_history(obs_terms)
    history_obs = _build_unitree_history_obs(history)
    _validate_shapes(history_obs, action_dim, onnx_metadata)

    print(f"[INFO] Unitree observation term order: {UNITREE_OBS_TERM_ORDER}")
    print(f"[INFO] Unitree single-frame dim: {UNITREE_SINGLE_FRAME_OBS_DIM}")
    print(f"[INFO] Unitree history length: {UNITREE_OBS_HISTORY_LENGTH}")
    print(f"[INFO] Unitree history obs dim: {history_obs.shape[-1]}")
    print(f"[INFO] Unitree action scale from deploy.yaml: {deploy_cfg['actions']['JointPositionAction']['scale'][0]}")

    step = 0
    with torch.inference_mode():
        while simulation_app.is_running() and step < args_cli.steps:
            onnx_input = history_obs.detach().cpu().numpy().astype("float32")
            onnx_actions = session.run([onnx_output_name], {onnx_input_name: onnx_input})[0]
            if onnx_actions.shape[-1] != action_dim:
                raise RuntimeError(f"ONNX returned action dim {onnx_actions.shape[-1]}, expected {action_dim}.")
            actions = torch.as_tensor(onnx_actions, dtype=torch.float32, device=env.unwrapped.device)
            _, _, terminated, truncated, _ = env.step(actions)

            obs_terms = _build_unitree_obs_terms(env)
            _append_unitree_history(history, obs_terms)
            history_obs = _build_unitree_history_obs(history)
            if torch.any(terminated | truncated):
                reset_terms = _build_unitree_obs_terms(env)
                history = _init_unitree_history(reset_terms)
                history_obs = _build_unitree_history_obs(history)
            step += 1

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
