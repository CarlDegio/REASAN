"""Standalone G1 safety-filter environment skeleton.

This first implementation validates the G1 asset, contacts, resets, terminations,
and the three-dimensional filter action boundary.  It deliberately does not load
or emulate a locomotion policy; that integration is the next implementation step.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from go2_lidar.tasks.g1_filter_env_cfg import G1FilterEnvCfg


class G1FilterEnv(DirectRLEnv):
    """G1 simulation shell exposing a normalized three-dimensional filter action."""

    cfg: G1FilterEnvCfg

    def __init__(self, cfg: G1FilterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._filter_actions = torch.zeros(self.num_envs, self.cfg.num_high_actions, device=self.device)
        self._safe_commands = torch.zeros_like(self._filter_actions)
        self._command_lower = torch.tensor(self.cfg.command_lower, device=self.device)
        self._command_upper = torch.tensor(self.cfg.command_upper, device=self.device)

        self._base_ids, _ = self._contact_sensor.find_bodies("torso_link")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link")
        if len(self._robot.joint_names) != 29:
            raise ValueError(f"G1 filter requires the 29-DoF asset, got {len(self._robot.joint_names)} joints")
        if len(self._base_ids) != 1 or len(self._feet_ids) != 2:
            raise ValueError(
                f"Unexpected G1 body mapping: torso={len(self._base_ids)}, feet={len(self._feet_ids)}"
            )

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)
        self._terrain.terrain_levels[:] = 0

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ).func("/World/skyLight")

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != self._filter_actions.shape:
            raise ValueError(f"Expected filter actions {tuple(self._filter_actions.shape)}, got {tuple(actions.shape)}")
        self._filter_actions.copy_(actions)
        normalized = actions.clamp(-1.0, 1.0)
        self._safe_commands.copy_(
            self._command_lower + 0.5 * (normalized + 1.0) * (self._command_upper - self._command_lower)
        )

    def _apply_action(self):
        # Step one intentionally has no locomotion adapter. Holding the nominal
        # pose exercises G1 physics and resets without inventing a control policy.
        self._robot.set_joint_position_target(self._robot.data.default_joint_pos)

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._safe_commands,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        # A neutral placeholder prevents accidental interpretation as a usable
        # filter objective before the G1-specific reward port is reviewed.
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        contact_history = self._contact_sensor.data.net_forces_w_history
        torso_contact = torch.any(
            torch.max(torch.norm(contact_history[:, :, self._base_ids], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        tipped = -self._robot.data.projected_gravity_b[:, 2] < 0.25
        too_low = self._robot.data.root_pos_w[:, 2] < 0.3
        return torso_contact | tipped | too_low, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._filter_actions[env_ids] = 0.0
        self._safe_commands[env_ids] = 0.0

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        zeros = torch.zeros(len(env_ids), device=self.device)
        yaw = torch.rand_like(zeros) * (2.0 * math.pi) - math.pi
        root_state[:, 3:7] = math_utils.quat_mul(
            root_state[:, 3:7], math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        )
        self._robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])
        self._robot.write_joint_state_to_sim(
            joint_pos, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids
        )

