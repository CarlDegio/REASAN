"""G1 port of the REASEN safety-filter task."""

import isaaclab.sim as sim_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from go2_lidar.sensor.lidar_pattern import (
    RayG1BodyEnvelope1xPatternCfg,
    RayG1BodyEnvelope3xPatternCfg,
    RayG1BodyEnvelope5xPatternCfg,
    RayG1BodyEnvelope11xPatternCfg,
)
from go2_lidar.tasks.g1_loco_env_cfg import UNITREE_G1_29DOF_CFG
from go2_lidar.tasks.go2_filter_env_cfg import Go2FilterEnvCfg


@configclass
class G1FilterEventCfg:
    """Unitree RL Lab 2.1 dynamics randomization used under the filter."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class G1FilterEnvCfg(Go2FilterEnvCfg):
    """REASEN ray/obstacle filter with a frozen Unitree G1 29-DoF loco actor."""

    num_loco_actions = 29
    action_scale_loco = 0.25
    loco_checkpoint = ""
    # Safe command envelope accepted by the frozen G1 loco actor.  Keep
    # lateral motion deliberately narrow and allow the newly-trained loco
    # policy to use its full yaw-command range.
    command_lower = (-0.5, -0.15, -1.0)
    command_upper = (1.0, 0.15, 1.0)
    random_command_ranges = (
        (1.5, 0.15, 1.0),
        (2.5, 0.15, 1.0),
        (0.5, 0.15, 1.0),
    )

    # Penalize first-order Filter action changes more strongly.  The
    # second-order smoothness term remains at the original REASEN weight.
    action_rate_reward_weight = -0.05
    action_smoothness_reward_weight = -0.01

    scene = InteractiveSceneCfg(num_envs=1024, env_spacing=10.0, replicate_physics=False)
    events: G1FilterEventCfg = G1FilterEventCfg()
    robot = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    static_friction_range = (0.3, 1.0)
    dynamic_friction_range = (0.3, 1.0)
    restitution_range = (0.0, 0.0)

    # G1 mounts Mid-360 upside down.  In torso/body coordinates the useful
    # optical FOV is approximately -52..+7 degrees; use 30 two-degree bins
    # centered on the downward-looking region.
    ray_grid_theta_range = (-55.0, 5.0)

    sim = SimulationCfg(
        dt=1 / 200,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
    )

    def __post_init__(self):
        # Keep the original Filter observation/ray contract, but never install
        # the Go2 RandomDCMotor actuator into the G1 articulation.
        self._build_observation_space()
        self.sim.render_interval = self.decimation
        self.raycaster.prim_path = "/World/envs/env_.*/Robot/torso_link"
        # Public G1 Mid360 URDF extrinsic: torso_link -> mid360_link.  The
        # near-pi pitch and yaw are essential: G1 mounts the Mid-360 upside
        # down so its +52 degree optical elevation looks toward the ground.
        # xyz=(0.0002835, 0.00003, 0.41618), rpy=(0, 3.101, 3.1415).
        self.raycaster.offset.pos = (0.0002835, 0.00003, 0.41618)
        self.raycaster.offset.rot = (9.401992e-7, -0.99979404, 4.631725e-5, 0.02029493)
        self.raycaster_measure.prim_path = "/World/envs/env_.*/Robot/torso_link"
        self.raycaster_measure.offset.pos = (0.0, 0.0, 0.0)
        self.raycaster.update_period = self.sim.dt * self.decimation
        self.raycaster_measure.update_period = self.sim.dt * self.decimation

    def set_raycaster_measure_pattern(self, pattern_name: str):
        """Select the G1-sized body-envelope ground-truth ray pattern."""
        patterns = {
            "1x": (RayG1BodyEnvelope1xPatternCfg, 1),
            "3x": (RayG1BodyEnvelope3xPatternCfg, 3),
            "5x": (RayG1BodyEnvelope5xPatternCfg, 5),
            "11x": (RayG1BodyEnvelope11xPatternCfg, 11),
        }
        try:
            pattern_cfg, self.num_ray_centers = patterns[pattern_name]
        except KeyError as exc:
            raise ValueError(f"Invalid G1 ray pattern: {pattern_name}") from exc
        self.raycaster_measure.pattern_cfg = pattern_cfg()
