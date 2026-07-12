"""Smoke-test the G1 filter environment skeleton with persistent visualization."""

import argparse

from isaaclab.app import AppLauncher

from reasan_kit_args import apply_default_reasan_kit_args

parser = argparse.ArgumentParser(description="Play the Unitree G1 filter environment skeleton.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Zero runs until the window is closed.")
parser.add_argument("--loco_checkpoint", type=str, default="", help="Unitree RL Lab model_*.pt checkpoint.")
parser.add_argument(
    "--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW")
)
parser.add_argument("--random_actions", action="store_true", help="Randomize the filter command every step.")
parser.add_argument("--onnx_reference", type=str, default="", help="Optional ONNX used only for PT parity checks.")
parser.add_argument("--with_dyn_obst", action="store_true")
parser.add_argument("--num_ray_centers", choices=("1x", "3x", "5x", "11x"), default="1x")
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
    cfg.seed = args_cli.seed
    cfg.loco_checkpoint = args_cli.loco_checkpoint
    cfg.wait_for_key = False
    cfg.is_play_env = True
    cfg.use_dynamic_obstacle = args_cli.with_dyn_obst
    cfg.set_raycaster_measure_pattern(args_cli.num_ray_centers)
    env = gym.make("Unitree-G1-Filter", cfg=cfg)
    env.reset()
    print(f"[VALIDATE] Runtime joint names ({len(env.unwrapped._robot.joint_names)}):")
    for index, name in enumerate(env.unwrapped._robot.joint_names):
        print(f"[VALIDATE]   action[{index:02d}] -> {name}")
    print(f"[VALIDATE] Default joint positions: {env.unwrapped._robot.data.default_joint_pos[0].tolist()}")
    print(f"[VALIDATE] Joint stiffness: {env.unwrapped._robot.data.joint_stiffness[0].tolist()}")
    print(f"[VALIDATE] Joint damping: {env.unwrapped._robot.data.joint_damping[0].tolist()}")
    materials = env.unwrapped._robot.root_physx_view.get_material_properties()[0]
    print(
        "[VALIDATE] Robot material ranges: "
        f"static=({materials[:, 0].min().item():.6g},{materials[:, 0].max().item():.6g}), "
        f"dynamic=({materials[:, 1].min().item():.6g},{materials[:, 1].max().item():.6g}), "
        f"restitution=({materials[:, 2].min().item():.6g},{materials[:, 2].max().item():.6g})"
    )

    onnx_session = None
    onnx_input_name = None
    onnx_output_name = None
    max_abs_error = 0.0
    sum_abs_error = 0.0
    compared_values = 0
    worst_action_index = -1
    if args_cli.onnx_reference:
        import onnxruntime as ort

        onnx_session = ort.InferenceSession(args_cli.onnx_reference, providers=["CPUExecutionProvider"])
        onnx_input_name = onnx_session.get_inputs()[0].name
        onnx_output_name = onnx_session.get_outputs()[0].name
        if args_cli.num_envs != 1:
            raise ValueError("The exported ONNX has fixed batch 1; parity validation requires --num_envs 1")
    command = torch.tensor(args_cli.command, dtype=torch.float32, device=env.unwrapped.device)
    lower = torch.tensor(cfg.command_lower, device=env.unwrapped.device)
    upper = torch.tensor(cfg.command_upper, device=env.unwrapped.device)
    if torch.any(command < lower) or torch.any(command > upper):
        raise ValueError(f"Command {tuple(args_cli.command)} is outside [{cfg.command_lower}, {cfg.command_upper}]")
    normalized_action = command.unsqueeze(0).repeat(args_cli.num_envs, 1)
    print(f"[INFO] Play command: vx={command[0].item()}, vy={command[1].item()}, yaw={command[2].item()}")
    step = 0
    while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
        with torch.inference_mode():
            actions = (
                lower + torch.rand(args_cli.num_envs, 3, device=env.unwrapped.device) * (upper - lower)
                if args_cli.random_actions
                else normalized_action
            )
            env.step(actions)
            if onnx_session is not None:
                pt_actions = env.unwrapped._loco_actions.detach().cpu()
                history_obs = env.unwrapped._flatten_loco_history().detach().cpu().numpy().astype("float32")
                onnx_actions = onnx_session.run([onnx_output_name], {onnx_input_name: history_obs})[0]
                error = torch.from_numpy(onnx_actions).sub(pt_actions).abs()
                step_max, flat_index = error.flatten().max(dim=0)
                if step_max.item() > max_abs_error:
                    max_abs_error = step_max.item()
                    worst_action_index = int(flat_index.item() % error.shape[-1])
                sum_abs_error += error.sum().item()
                compared_values += error.numel()
        step += 1
    if onnx_session is not None:
        mean_abs_error = sum_abs_error / max(compared_values, 1)
        print(
            f"[VALIDATE] PT-ONNX steps={step}, max_abs_error={max_abs_error:.9g}, "
            f"mean_abs_error={mean_abs_error:.9g}, worst_action_index={worst_action_index}"
        )
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
