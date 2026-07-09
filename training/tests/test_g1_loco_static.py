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
    assert "class G1LocoEnv" in env
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
