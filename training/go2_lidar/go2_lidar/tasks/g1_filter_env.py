"""REASEN safety-filter task migrated to Unitree G1 29-DoF locomotion."""

from __future__ import annotations

import math

import isaaclab.utils.math as math_utils
import torch

from go2_lidar.policies import UnitreeG1LocoAdapter
from go2_lidar.tasks.g1_filter_env_cfg import G1FilterEnvCfg
from go2_lidar.tasks.go2_filter_env import Go2FilterEnv


class G1FilterEnv(Go2FilterEnv):
    """Original REASEN filter pipeline driving a frozen Unitree G1 actor."""

    cfg: G1FilterEnvCfg
    _OBS_HISTORY_LENGTH = 5
    _OBS_TERM_ORDER = (
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )

    def __init__(self, cfg: G1FilterEnvCfg, render_mode: str | None = None, **kwargs):
        if not cfg.loco_checkpoint:
            raise ValueError("G1 Filter requires cfg.loco_checkpoint pointing to a Unitree model_*.pt")
        super().__init__(cfg, render_mode, **kwargs)

    def _initialize_loco_policy(self):
        self._loco_policy = UnitreeG1LocoAdapter(self.cfg.loco_checkpoint, self.device)
        self._loco_obs_history = None
        print(f"[INFO] Loaded frozen Unitree G1 loco actor: {self._loco_policy.checkpoint}")
        print(f"[INFO] Loco batch interface: [N, 480] -> [N, 29], N={self.num_envs}")

    def _configure_robot_indices(self):
        self._base_id_cs, _ = self._contact_sensor.find_bodies("torso_link")
        self._feet_ids_cs, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link")
        self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies(
            "torso_link|.*_hip_.*|.*_knee.*|.*_shoulder.*|.*_elbow.*"
        )
        self._feet_ids_bd, _ = self._robot.find_bodies(".*_ankle_roll_link")
        self._hip_ids_jt = []
        if len(self._robot.joint_names) != 29:
            raise ValueError(f"G1 Filter requires 29 joints, got {len(self._robot.joint_names)}")
        if len(self._base_id_cs) != 1 or len(self._feet_ids_cs) != 2:
            raise ValueError(
                f"Unexpected G1 body mapping: torso={len(self._base_id_cs)}, feet={len(self._feet_ids_cs)}"
            )

    def _randomize_mass(self):
        # The official add_base_mass startup event already applies [-1, +3] kg.
        return

    def _configure_actuator_gains(self):
        # Preserve the exact Unitree RL Lab actuator stiffness and damping.
        return

    def _compute_loco_actions(self):
        safe_commands = self._high_actions.clamp(self._command_lower, self._command_upper)
        terms = {
            "base_ang_vel": self._robot.data.root_ang_vel_b * 0.2,
            "projected_gravity": self._robot.data.projected_gravity_b,
            "velocity_commands": safe_commands,
            "joint_pos_rel": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel_rel": self._robot.data.joint_vel * 0.05,
            "last_action": self._prev_loco_actions[-1],
        }
        if self._loco_obs_history is None:
            self._loco_obs_history = {
                name: value.unsqueeze(1).repeat(1, self._OBS_HISTORY_LENGTH, 1)
                for name, value in terms.items()
            }
        else:
            for name, value in terms.items():
                self._loco_obs_history[name] = torch.cat(
                    (self._loco_obs_history[name][:, 1:], value.unsqueeze(1)), dim=1
                )
        observations = torch.cat(
            [self._loco_obs_history[name].flatten(start_dim=1) for name in self._OBS_TERM_ORDER], dim=-1
        )
        if observations.shape != (self.num_envs, 480):
            raise RuntimeError(f"Invalid Unitree loco observation shape: {tuple(observations.shape)}")
        self._loco_actions.copy_(self._loco_policy(observations))

    def _apply_action(self):
        self._robot.set_joint_position_target(
            self._robot.data.default_joint_pos + self._loco_actions * self.cfg.action_scale_loco
        )

    def _reset_loco_policy(self, env_ids: torch.Tensor, env_mask: torch.Tensor):
        self._loco_actions[env_ids] = 0.0
        for actions in self._prev_loco_actions:
            actions[env_ids] = 0.0
        if self._loco_obs_history is None:
            return
        safe_commands = self._high_actions.clamp(self._command_lower, self._command_upper)
        terms = {
            "base_ang_vel": self._robot.data.root_ang_vel_b * 0.2,
            "projected_gravity": self._robot.data.projected_gravity_b,
            "velocity_commands": safe_commands,
            "joint_pos_rel": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel_rel": self._robot.data.joint_vel * 0.05,
            "last_action": self._loco_actions,
        }
        for name, value in terms.items():
            self._loco_obs_history[name][env_ids] = value[env_ids].unsqueeze(1).repeat(
                1, self._OBS_HISTORY_LENGTH, 1
            )

    def _reset_joint_state(self, env_ids: torch.Tensor):
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos.clamp_(limits[..., 0], limits[..., 1])
        joint_vel = torch.empty_like(self._robot.data.default_joint_vel[env_ids]).uniform_(-1.0, 1.0)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def _reset_robot_and_cmd(self, env_ids: torch.Tensor):
        env_ids = env_ids.flatten()
        if env_ids.numel() == 0:
            return
        patches = torch.randint(0, self._flat_patches.shape[0], (len(env_ids),), device=self.device)
        self._robot_start_pos[env_ids] = self._flat_patches[patches]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = self._robot_start_pos[env_ids]
        root_state[:, 2] += 0.8
        zeros = torch.zeros(len(env_ids), device=self.device)
        yaw = torch.rand_like(zeros) * (2.0 * math.pi) - math.pi
        root_state[:, 3:7] = math_utils.quat_mul(
            root_state[:, 3:7], math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        )
        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        if not self._use_keyboard_control:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_ids] = True
            self.randomly_sample_commands(mask & self._use_random_cmd)
            self.randomly_sample_speed_and_heading(mask & (~self._use_random_cmd))
