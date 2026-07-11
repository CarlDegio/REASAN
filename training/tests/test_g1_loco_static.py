from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
    assert "action_scale = 0.25" in env_cfg
    assert "class G1LocoEnv" in env
    assert "r_track_lin_vel_xy" in env
    assert "r_feet_gait" in env
    assert "r_joint_deviation_fingers" not in env
    assert "r_undesired_contacts = torch.sum(" in env
    assert "r_undesired_contacts = r_undesired_contacts.float()" in env
    assert "self._energy_ids_jt = self._controlled_joint_ids.copy()" in env
    assert 'self._energy_ids_jt = self._find_joints(".*_hip_.*|.*_knee_joint|.*_ankle_.*")' not in env
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
    assert "push_enabled = False" in env_cfg
    assert "if not self.push_enabled:" in env_cfg
    assert "self.events.push_robot = None" in env_cfg
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


def test_g1_loco_observation_noise_matches_unitree_effective_scale():
    env = (ROOT / "go2_lidar" / "go2_lidar" / "tasks" / "g1_loco_env.py").read_text()

    # Unitree adds noise before observation scaling. This direct environment
    # therefore stores the already-scaled, effective noise amplitudes.
    assert "torch.ones(3) * 0.2 * 0.2" in env
    assert "torch.ones(self._num_actions) * 1.5 * 0.05" in env
    assert "torch.ones(self._num_actions) * 1.5 * 0.5" not in env


def test_g1_loco_reset_does_not_apply_persistent_external_wrench():
    env_cfg = (ROOT / "go2_lidar" / "go2_lidar" / "tasks" / "g1_loco_env_cfg.py").read_text()

    assert '"force_range": (0.0, 0.0)' in env_cfg
    assert '"torque_range": (0.0, 0.0)' in env_cfg
    assert '"force_range": (-1.0, 1.0)' not in env_cfg
    assert '"torque_range": (-1.0, 1.0)' not in env_cfg


def test_g1_loco_rewards_match_unitree_except_for_retained_termination_penalty():
    env = (ROOT / "go2_lidar" / "go2_lidar" / "tasks" / "g1_loco_env.py").read_text()

    assert 'self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies("(?!.*ankle.*).*")' in env
    assert "self._robot.data.joint_pos[:, self._controlled_joint_ids]" in env
    assert "self._robot.data.soft_joint_pos_limits[:, self._controlled_joint_ids]" in env
    assert "self._contact_sensor.data.current_contact_time[:, self._feet_ids_cs] > 0.0" in env
    assert "return torch.sum((contact == desired_contact).float(), dim=1)" in env
    assert "torch.tanh(2.0 * foot_xy_speed)" in env
    assert "r_feet_clearance[cmd_norm < 0.1] = 0.0" not in env

    assert "r_termination = self._reset_buf.float()" in env
    assert "r_termination *= -200.0" in env
