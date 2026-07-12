"""Configuration for the standalone G1 safety-filter environment skeleton."""

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from go2_lidar.tasks.g1_loco_env_cfg import G1LocoEnvCfg


@configclass
class G1FilterEventCfg:
    """Unitree RL Lab 2.1 G1 training randomization, without REASEN COM changes."""

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
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
class G1FilterEnvCfg(G1LocoEnvCfg):
    """Stage-one skeleton: G1 physics with a three-dimensional filter interface.

    The frozen locomotion policy is intentionally not part of this configuration yet.
    Until that adapter is added, the environment holds the G1 at its default pose.
    """

    episode_length_s = 20.0

    num_high_actions = 3
    num_actions = None
    action_space = num_high_actions
    observation_space = 9
    state_space = 0

    # Unitree RL Lab 2.1 command envelope used by the future locomotion adapter.
    command_lower = (-0.5, -0.3, -0.2)
    command_upper = (1.0, 0.3, 0.2)

    # Empty keeps the step-one pose-hold behavior. A model_*.pt path enables
    # the frozen, batch-capable Unitree RL Lab 2.1 locomotion actor.
    loco_checkpoint = ""
    loco_action_scale = 0.25

    scene = InteractiveSceneCfg(num_envs=64, env_spacing=2.5, replicate_physics=False)
    events: G1FilterEventCfg = G1FilterEventCfg()
