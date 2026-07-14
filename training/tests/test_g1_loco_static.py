from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_g1_training_checkpoint_intervals_are_2000_iterations():
    tasks_dir = ROOT / "go2_lidar" / "go2_lidar" / "tasks"

    loco_cfg = (tasks_dir / "g1_loco_ppo_cfg.py").read_text()
    filter_cfg = (tasks_dir / "g1_filter_ppo_cfg.py").read_text()

    assert "save_interval = 2000" in loco_cfg
    assert "save_interval = 2000" in filter_cfg


def test_g1_loco_files_and_registration_are_present():
    tasks_dir = ROOT / "go2_lidar" / "go2_lidar" / "tasks"

    env_cfg = (tasks_dir / "g1_loco_env_cfg.py").read_text()
    env = (tasks_dir / "g1_loco_env.py").read_text()
    task_init = (tasks_dir / "__init__.py").read_text()
    train_script = (ROOT / "scripts" / "train_g1_loco.py").read_text()
    assets_script = (ROOT / "assets" / "download_reasan_assets.py").read_text()
    manual_script = (ROOT / "tests" / "manual_g1_loco_random_actions.py").read_text()
    unitree_play_script = (ROOT / "tests" / "manual_g1_loco_unitree_onnx.py").read_text()

    assert "class G1LocoEnvCfg" in env_cfg
    assert "UNITREE_G1_29DOF_CFG" in env_cfg
    assert "G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd" in env_cfg
    assert "action_space = 29" in env_cfg
    assert "observation_space = 480" in env_cfg
    assert "state_space = 495" in env_cfg
    assert "G1_LOCO_FLAT_TERRAIN_CFG" in env_cfg
    assert '"flat": terrain_gen.MeshPlaneTerrainCfg' in env_cfg
    assert "GO2_LOCO_TERRAIN_CFG" not in env_cfg
    assert "num_rows=9" in env_cfg
    assert "num_cols=21" in env_cfg
    assert "class G1LocoEnv" in env
    assert "_OBS_HISTORY_LENGTH = 5" in env
    assert 'class_name="ActorCritic"' in (tasks_dir / "g1_loco_ppo_cfg.py").read_text()
    assert "r_track_lin_vel_xy" in env
    assert "r_feet_gait" in env
    assert "r_joint_deviation_fingers" not in env
    assert "r_undesired_contacts = torch.sum(" in env
    assert "r_undesired_contacts = r_undesired_contacts.float()" in env
    assert "Unitree-G1-Locomotion" in task_init
    assert 'task_name = "Unitree-G1-Locomotion"' in train_script
    assert "DEFAULT_UNITREE_MODEL_ASSETS" in assets_script
    assert "G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd" in assets_script
    assert "DEFAULT_UNITREE_RL_LAB_ASSETS" in assets_script
    assert "deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx" in assets_script
    assert "deploy/robots/g1_29dof/config/policy/velocity/v0/params/deploy.yaml" in assets_script
    assert "--policy" not in manual_script
    assert "unitree_onnx" not in manual_script
    assert "--checkpoint" not in manual_script
    assert "onnxruntime" not in manual_script
    assert "UNITREE_G1_VELOCITY_ONNX" not in manual_script
    assert "UNITREE_G1_VELOCITY_ONNX" in unitree_play_script
    assert "UNITREE_OBS_HISTORY_LENGTH = 5" in unitree_play_script
    assert "UNITREE_SINGLE_FRAME_OBS_DIM = 96" in unitree_play_script
    assert "base_ang_vel" in unitree_play_script
    assert "projected_gravity" in unitree_play_script
    assert "velocity_commands" in unitree_play_script
    assert "joint_pos_rel" in unitree_play_script
    assert "joint_vel_rel" in unitree_play_script
    assert "last_action" in unitree_play_script
    assert "_build_unitree_history_obs" in unitree_play_script
    assert "SessionOptions" in unitree_play_script
    assert "intra_op_num_threads" in unitree_play_script
    assert "inter_op_num_threads" in unitree_play_script


def test_g1_loco_push_and_command_curriculum_matches_humanoid_ranges():
    tasks_dir = ROOT / "go2_lidar" / "go2_lidar" / "tasks"
    env_cfg = (tasks_dir / "g1_loco_env_cfg.py").read_text()
    env = (tasks_dir / "g1_loco_env.py").read_text()

    assert 'interval_range_s=(5.0, 5.0)' in env_cfg
    assert '"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}' in env_cfg
    assert "cmd_lin_vel_x_range = (-0.5, 0.5)" in env_cfg
    assert "cmd_lin_vel_y_range = (-0.1, 0.1)" in env_cfg
    assert "cmd_ang_vel_z_range = (-0.4, 0.4)" in env_cfg

    assert "self.events.push_robot.interval_range_s = (3.0, 3.0)" in env_cfg
    assert 'self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}' in env_cfg
    assert "self.cmd_lin_vel_x_range = (-1.0, 1.0)" in env_cfg
    assert "self.cmd_lin_vel_y_range = (-0.3, 0.3)" in env_cfg
    assert "self.cmd_ang_vel_z_range = (-1.0, 1.0)" in env_cfg
    assert "self.cmd_resample_interval = (2.0, 3.0)" in env_cfg

    assert "self._cmd_lower" in env
    assert "self._cmd_upper" in env
    assert "self.cfg.standing_command_probability" in env
    assert "new_commands[:, 0] = new_commands[:, 0].abs()" not in env
    assert "new_commands[:, 1] = 0.0" not in env
