"""Configuration for the standalone G1 safety-filter environment skeleton."""

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg

from go2_lidar.tasks.g1_loco_env_cfg import G1LocoEnvCfg


@configclass
class G1FilterEnvCfg(G1LocoEnvCfg):
    """Stage-one skeleton: G1 physics with a three-dimensional filter interface.

    The frozen locomotion policy is intentionally not part of this configuration yet.
    Until that adapter is added, the environment holds the G1 at its default pose.
    """

    episode_length_s = 9.0

    num_high_actions = 3
    num_actions = num_high_actions
    action_space = num_high_actions
    observation_space = 9
    state_space = 0

    # Unitree RL Lab 2.1 command envelope used by the future locomotion adapter.
    command_lower = (-0.5, -0.3, -0.2)
    command_upper = (1.0, 0.3, 0.2)

    scene = InteractiveSceneCfg(num_envs=64, env_spacing=4.0, replicate_physics=False)
