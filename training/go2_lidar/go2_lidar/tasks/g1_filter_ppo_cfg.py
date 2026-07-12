"""PPO configuration for the migrated G1 safety filter."""

from isaaclab.utils import configclass

from go2_lidar.tasks.go2_filter_ppo_cfg import Go2FilterPPOCfg


@configclass
class G1FilterPPOCfg(Go2FilterPPOCfg):
    experiment_name = "g1_filter"
