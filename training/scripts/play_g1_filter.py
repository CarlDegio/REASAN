"""Smoke-test the G1 filter environment skeleton with persistent visualization."""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from reasan_kit_args import apply_default_reasan_kit_args

parser = argparse.ArgumentParser(description="Play the Unitree G1 filter environment skeleton.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=0, help="Zero runs until the window is closed.")
parser.add_argument("--zmq_filter_endpoint", type=str, default="tcp://*:5558")
parser.add_argument("--zmq_filter_hz", type=float, default=10.0)
parser.add_argument("--navila_endpoint", type=str, default="")
parser.add_argument("--loco_checkpoint", type=str, default="", help="Unitree RL Lab model_*.pt checkpoint.")
parser.add_argument("--filter_checkpoint", type=str, default="", help="REASEN Filter model_*.pt checkpoint.")
parser.add_argument(
    "--command", type=float, nargs=3, default=(0.5, 0.0, 0.0), metavar=("VX", "VY", "YAW")
)
parser.add_argument("--random_actions", action="store_true", help="Randomize the filter command every step.")
parser.add_argument("--onnx_reference", type=str, default="", help="Optional ONNX used only for PT parity checks.")
parser.add_argument("--with_dyn_obst", action="store_true")
parser.add_argument(
    "--collect",
    choices=("none", "train", "val"),
    default="none",
    help="Collect sequential G1 Ray Predictor data; existing target HDF5 files are overwritten.",
)
parser.add_argument(
    "--obst_speed_range",
    type=float,
    nargs=2,
    default=(0.3, 0.8),
    metavar=("MIN", "MAX"),
    help="Dynamic-obstacle speed range in m/s.",
)
parser.add_argument(
    "--action_ema_alpha",
    type=float,
    default=0.0,
    help="EMA retention for the Filter's 3-D safe velocity output; 0 disables smoothing.",
)
parser.add_argument("--suppress_output_on_zero_input", action="store_true")
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
    "--cv_occupancy",
    action="store_true",
    help="Show physical measure-ray occupancy and normalized ActorRay in an OpenCV window.",
)
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
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from filter_velocity_zmq import FilterVelocityPublisher  # noqa: E402
from filter_play_control import suppress_output_for_zero_input  # noqa: E402
from navila_velocity_zmq import NavilaVelocitySubscriber  # noqa: E402


class OccupancyViewer:
    """Bird's-eye comparison of physical measure hits and normalized ActorRay."""

    WINDOW_NAME = "G1 Filter Occupancy | physical (m) vs normalized ActorRay"
    PANEL_SIZE = 520
    CENTER = (260, 285)
    PHYSICAL_RADIUS_M = 3.0
    DRAW_RADIUS_PX = 210

    def __init__(self, unwrapped_env):
        import cv2

        self.cv2 = cv2
        self.env = unwrapped_env
        self.enabled = True
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.PANEL_SIZE * 2, self.PANEL_SIZE)

    def _base_panel(self, title: str, radius_label: str) -> np.ndarray:
        cv2 = self.cv2
        panel = np.full((self.PANEL_SIZE, self.PANEL_SIZE, 3), 24, dtype=np.uint8)
        center = self.CENTER
        for fraction in (1.0 / 3.0, 2.0 / 3.0, 1.0):
            cv2.circle(panel, center, int(self.DRAW_RADIUS_PX * fraction), (65, 65, 65), 1, cv2.LINE_AA)
        cv2.line(
            panel,
            (center[0], center[1] - self.DRAW_RADIUS_PX),
            (center[0], center[1] + self.DRAW_RADIUS_PX),
            (50, 50, 50),
            1,
        )
        cv2.line(
            panel,
            (center[0] - self.DRAW_RADIUS_PX, center[1]),
            (center[0] + self.DRAW_RADIUS_PX, center[1]),
            (50, 50, 50),
            1,
        )
        cv2.putText(panel, title, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(
            panel,
            radius_label,
            (14, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (170, 170, 170),
            1,
            cv2.LINE_AA,
        )
        # Robot body and +X forward direction.
        cv2.circle(panel, center, 7, (255, 210, 80), -1, cv2.LINE_AA)
        cv2.arrowedLine(panel, center, (center[0], center[1] - 30), (255, 210, 80), 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(panel, "+X", (center[0] + 8, center[1] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 210, 80), 1)
        return panel

    def _xy_to_pixels(self, xy: np.ndarray, physical_radius: float) -> tuple[np.ndarray, np.ndarray]:
        scale = self.DRAW_RADIUS_PX / physical_radius
        # Robot +X is image-up; robot +Y is image-left.
        px = np.rint(self.CENTER[0] - xy[:, 1] * scale).astype(np.int32)
        py = np.rint(self.CENTER[1] - xy[:, 0] * scale).astype(np.int32)
        return px, py

    def _physical_panel(self) -> np.ndarray:
        cv2 = self.cv2
        panel = self._base_panel("Physical occupancy from raycaster_measure", "metric radius: 3.0 m")
        hits_w = self.env._raycaster_measure.data.ray_hits_w[0]
        finite = torch.all(torch.isfinite(hits_w), dim=-1)
        hits_w = hits_w[finite]
        if hits_w.numel() > 0:
            relative_w = hits_w - self.env._robot.data.root_pos_w[0]
            yaw = math_utils.yaw_quat(self.env._robot.data.root_quat_w[0].unsqueeze(0)).expand(
                relative_w.shape[0], -1
            )
            relative_b = math_utils.quat_apply_inverse(yaw, relative_w)
            # Match ActorRay's metric clipping decision: 3-D distance from the
            # articulation root, while plotting the corresponding XY position.
            inside = torch.linalg.norm(relative_b, dim=-1) < self.PHYSICAL_RADIUS_M
            xy = relative_b[inside, :2].detach().cpu().numpy()
            if xy.shape[0] > 0:
                px, py = self._xy_to_pixels(xy, self.PHYSICAL_RADIUS_M)
                valid = (px >= 0) & (px < self.PANEL_SIZE) & (py >= 0) & (py < self.PANEL_SIZE)
                occupancy = np.zeros_like(panel)
                occupancy[py[valid], px[valid]] = (0, 95, 255)
                occupancy = cv2.dilate(occupancy, np.ones((3, 3), dtype=np.uint8), iterations=1)
                mask = np.any(occupancy != 0, axis=-1)
                panel[mask] = occupancy[mask]
        cv2.putText(
            panel,
            "orange: actual measure hit positions within 3 m",
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (0, 165, 255),
            1,
            cv2.LINE_AA,
        )
        return panel

    def _normalized_panel(self) -> np.ndarray:
        cv2 = self.cv2
        panel = self._base_panel("Normalized ActorRay received by policy", "normalized radius: 1.0 == 3.0 m")
        start, end = self.env.cfg.obs_ranges["actor_ray"]
        normalized = self.env._obs_buf[0, start:end].detach().clamp(0.0, 1.0).cpu().numpy()
        directions = self.env._ray_directions_b[0, :, :2].detach().cpu().numpy()
        endpoints = directions * normalized[:, None]
        px, py = self._xy_to_pixels(endpoints, 1.0)
        for index in range(end - start):
            endpoint = (int(px[index]), int(py[index]))
            cv2.line(panel, self.CENTER, endpoint, (55, 72, 72), 1, cv2.LINE_AA)
            color = (60, 80, 235) if normalized[index] < 0.995 else (110, 110, 110)
            cv2.circle(panel, endpoint, 2, color, -1, cv2.LINE_AA)
        cv2.putText(
            panel,
            f"min={normalized.min():.3f}  mean={normalized.mean():.3f}  max={normalized.max():.3f}",
            (14, self.PANEL_SIZE - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            "red: occupied before 3 m; gray: clipped/no hit",
            (14, self.PANEL_SIZE - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (80, 130, 235),
            1,
            cv2.LINE_AA,
        )
        return panel

    def render(self) -> None:
        if not self.enabled:
            return
        cv2 = self.cv2
        frame = np.concatenate((self._physical_panel(), self._normalized_panel()), axis=1)
        command = self.env._cmd_buffer[0].detach().cpu().tolist()
        safe = self.env._high_actions[0].detach().cpu().tolist()
        cv2.putText(
            frame,
            f"input  vx={command[0]:+.3f} vy={command[1]:+.3f} wz={command[2]:+.3f}",
            (18, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (80, 230, 80),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"EMA out vx={safe[0]:+.3f} vy={safe[1]:+.3f} wz={safe[2]:+.3f}",
            (18, 97),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (80, 80, 240),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(self.WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    def close(self) -> None:
        if self.enabled:
            self.cv2.destroyWindow(self.WINDOW_NAME)
            self.enabled = False


def main():
    if not 0.0 <= args_cli.action_ema_alpha < 1.0:
        raise ValueError("--action_ema_alpha must be in [0, 1)")
    if args_cli.obst_speed_range[0] < 0.0 or args_cli.obst_speed_range[0] > args_cli.obst_speed_range[1]:
        raise ValueError("--obst_speed_range must satisfy 0 <= MIN <= MAX")
    collecting = args_cli.collect != "none"
    if collecting and not args_cli.filter_checkpoint:
        raise ValueError("--collect requires --filter_checkpoint so data is collected with a trained safety Filter")
    if collecting and args_cli.use_ray_predictor:
        raise ValueError("Do not use --use_ray_predictor while collecting Ray Predictor ground-truth data")
    if collecting:
        Path("ray_predictor/ray_predictor/data").mkdir(parents=True, exist_ok=True)

    cfg = parse_env_cfg("Unitree-G1-Filter", device=args_cli.device, num_envs=args_cli.num_envs)
    cfg.seed = args_cli.seed
    cfg.loco_checkpoint = args_cli.loco_checkpoint
    cfg.wait_for_key = False
    cfg.is_play_env = True
    cfg.high_action_ema_alpha = args_cli.action_ema_alpha
    cfg.use_dynamic_obstacle = args_cli.with_dyn_obst
    cfg.obst_speed_range = tuple(args_cli.obst_speed_range)
    cfg.data_collection_type = args_cli.collect
    cfg.use_predicted_rays = args_cli.use_ray_predictor
    cfg.ray_predictor_checkpoint = args_cli.ray_predictor_checkpoint
    cfg.set_raycaster_measure_pattern(args_cli.num_ray_centers)
    cfg.raycaster.debug_vis = args_cli.debug_raycaster
    if args_cli.debug_raycaster:
        cfg.raycaster.visualizer_cfg.prim_path = "/Visuals/RawMid360Hits"
        cfg.raycaster.visualizer_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 0.3, 1.0)
    env = gym.make("Unitree-G1-Filter", cfg=cfg)
    env.reset()
    occupancy_viewer = OccupancyViewer(env.unwrapped) if args_cli.cv_occupancy else None
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
    navila_subscriber = None
    if args_cli.navila_endpoint:
        navila_subscriber = NavilaVelocitySubscriber(args_cli.navila_endpoint)
        normalized_action.zero_()
        print(f"[NAVILA] Listening for velocity commands on {args_cli.navila_endpoint}")
    # In Play, --command is the Filter input.  Do not let the training-time
    # command sampler silently replace it with an unrelated random command.
    if not collecting:
        env.unwrapped._cmd_buffer.copy_(normalized_action)
        env.unwrapped._cmd_resample_accums.zero_()
        env.unwrapped._cmd_resample_delays.fill_(float("inf"))
        print(f"[INFO] Play command: vx={command[0].item()}, vy={command[1].item()}, yaw={command[2].item()}")
    else:
        print(
            f"[INFO] Collecting G1 Ray Predictor {args_cli.collect} data with randomized commands; "
            f"dynamic_obstacles={args_cli.with_dyn_obst}, speed_range={tuple(args_cli.obst_speed_range)} m/s"
        )
    print(f"[INFO] Filter action EMA alpha: {cfg.high_action_ema_alpha}")

    env_step_hz = 1.0 / (cfg.sim.dt * cfg.decimation)
    publish_interval_steps = max(1, round(env_step_hz / args_cli.zmq_filter_hz))
    actual_publish_hz = env_step_hz / publish_interval_steps
    velocity_publisher = FilterVelocityPublisher(args_cli.zmq_filter_endpoint)
    print(
        f"[INFO] Filter velocity ZMQ: {args_cli.zmq_filter_endpoint}, "
        f"requested={args_cli.zmq_filter_hz:g} Hz, actual={actual_publish_hz:g} Hz"
    )

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
            if navila_subscriber is not None:
                navila_command = torch.tensor(
                    navila_subscriber.poll(), dtype=torch.float32, device=env.unwrapped.device
                ).clamp(lower, upper)
                normalized_action.copy_(navila_command.unsqueeze(0).expand_as(normalized_action))
            # Re-apply after an episode reset as reset logic belongs to the
            # randomized training environment.
            if not collecting:
                env.unwrapped._cmd_buffer.copy_(normalized_action)
                env.unwrapped._cmd_resample_accums.zero_()
                env.unwrapped._cmd_resample_delays.fill_(float("inf"))
            if filter_policy is not None:
                actions = filter_policy(filter_obs)
                if args_cli.suppress_output_on_zero_input:
                    actions = suppress_output_for_zero_input(
                        actions, normalized_action, env.unwrapped._high_actions
                    )
                filter_obs, _, _, _ = env.step(actions)
            else:
                actions = (
                    lower + torch.rand(args_cli.num_envs, 3, device=env.unwrapped.device) * (upper - lower)
                    if args_cli.random_actions
                    else normalized_action
                )
                env.step(actions)
            if step % publish_interval_steps == 0:
                safe_velocity = env.unwrapped._high_actions[0].detach().cpu().tolist()
                velocity_publisher.publish(safe_velocity)
            measured_body_velocity_sum += env.unwrapped._robot.data.root_lin_vel_b[0]
            if occupancy_viewer is not None:
                occupancy_viewer.render()
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
    if occupancy_viewer is not None:
        occupancy_viewer.close()
    if env.unwrapped._data_writer is not None:
        env.unwrapped._data_writer.close()
    if navila_subscriber is not None:
        navila_subscriber.close()
    velocity_publisher.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
