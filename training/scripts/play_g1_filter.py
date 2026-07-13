"""Smoke-test the G1 filter environment skeleton with persistent visualization."""

import argparse

from isaaclab.app import AppLauncher

from reasan_kit_args import apply_default_reasan_kit_args

parser = argparse.ArgumentParser(description="Play the Unitree G1 filter environment skeleton.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Zero runs until the window is closed.")
parser.add_argument("--loco_checkpoint", type=str, default="", help="Unitree RL Lab model_*.pt checkpoint.")
parser.add_argument("--filter_checkpoint", type=str, default="", help="REASEN Filter model_*.pt checkpoint.")
parser.add_argument(
    "--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW")
)
parser.add_argument("--random_actions", action="store_true", help="Randomize the filter command every step.")
parser.add_argument("--onnx_reference", type=str, default="", help="Optional ONNX used only for PT parity checks.")
parser.add_argument("--with_dyn_obst", action="store_true")
parser.add_argument("--use_ray_predictor", action="store_true")
parser.add_argument(
    "--ray_predictor_checkpoint",
    type=str,
    default=str(
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "deployment/reasan/model/ray_predictor_1.onnx"
    ),
)
parser.add_argument("--num_ray_centers", choices=("1x", "3x", "5x", "11x"), default="5x")
parser.add_argument(
    "--debug_raycaster",
    action="store_true",
    help="Visualize the raw policy RayCaster rays and hit points.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
apply_default_reasan_kit_args(args_cli, __file__)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import go2_lidar.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def main():
    cfg = parse_env_cfg("Unitree-G1-Filter", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.loco_checkpoint = args_cli.loco_checkpoint
    cfg.wait_for_key = False
    cfg.is_play_env = True
    cfg.use_dynamic_obstacle = args_cli.with_dyn_obst
    cfg.use_predicted_rays = args_cli.use_ray_predictor
    cfg.ray_predictor_checkpoint = args_cli.ray_predictor_checkpoint
    cfg.set_raycaster_measure_pattern(args_cli.num_ray_centers)
    cfg.raycaster.debug_vis = args_cli.debug_raycaster
    if args_cli.debug_raycaster:
        cfg.raycaster.visualizer_cfg.prim_path = "/Visuals/RawMid360Hits"
        cfg.raycaster.visualizer_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 0.3, 1.0)
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
    # In Play, --command is the Filter input.  Do not let the training-time
    # command sampler silently replace it with an unrelated random command.
    env.unwrapped._cmd_buffer.copy_(normalized_action)
    env.unwrapped._cmd_resample_accums.zero_()
    env.unwrapped._cmd_resample_delays.fill_(float("inf"))
    print(f"[INFO] Play command: vx={command[0].item()}, vy={command[1].item()}, yaw={command[2].item()}")

    filter_policy = None
    filter_obs = None
    if args_cli.filter_checkpoint:
        checkpoint = __import__("pathlib").Path(args_cli.filter_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Filter checkpoint does not exist: {checkpoint}")
        agent_cfg = load_cfg_from_registry("Unitree-G1-Filter", "rsl_rl_cfg_entry_point")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=None,
            device=agent_cfg.device,
            use_ray=True,
        )
        runner.load(str(checkpoint), load_optimizer=False)
        filter_policy = runner.get_inference_policy(device=env.unwrapped.device)
        filter_obs, _ = env.get_observations()
        print(f"[INFO] Loaded Filter checkpoint: {checkpoint}")
    step = 0
    measured_body_velocity_sum = torch.zeros(3, device=env.unwrapped.device)
    while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
        with torch.inference_mode():
            # Re-apply after an episode reset as reset logic belongs to the
            # randomized training environment.
            env.unwrapped._cmd_buffer.copy_(normalized_action)
            env.unwrapped._cmd_resample_accums.zero_()
            env.unwrapped._cmd_resample_delays.fill_(float("inf"))
            if filter_policy is not None:
                actions = filter_policy(filter_obs)
                filter_obs, _, _, _ = env.step(actions)
            else:
                actions = (
                    lower + torch.rand(args_cli.num_envs, 3, device=env.unwrapped.device) * (upper - lower)
                    if args_cli.random_actions
                    else normalized_action
                )
                env.step(actions)
            measured_body_velocity_sum += env.unwrapped._robot.data.root_lin_vel_b[0]
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
    if step > 0:
        mean_velocity = measured_body_velocity_sum / step
        print(
            "[VALIDATE] Mean measured body velocity: "
            f"vx={mean_velocity[0].item():.6f}, vy={mean_velocity[1].item():.6f}, vz={mean_velocity[2].item():.6f}"
        )
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
