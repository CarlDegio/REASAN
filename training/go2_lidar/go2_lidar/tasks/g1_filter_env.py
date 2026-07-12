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

from go2_lidar.policies import UnitreeG1LocoAdapter
from go2_lidar.tasks.g1_filter_env_cfg import G1FilterEnvCfg


class G1FilterEnv(DirectRLEnv):
    """G1 simulation shell exposing a normalized three-dimensional filter action."""

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
        super().__init__(cfg, render_mode, **kwargs)

        self._filter_actions = torch.zeros(self.num_envs, self.cfg.num_high_actions, device=self.device)
        self._safe_commands = torch.zeros_like(self._filter_actions)
        self._command_lower = torch.tensor(self.cfg.command_lower, device=self.device)
        self._command_upper = torch.tensor(self.cfg.command_upper, device=self.device)

        self._loco_policy = None
        self._loco_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self._loco_obs_history = None
        if self.cfg.loco_checkpoint:
            self._loco_policy = UnitreeG1LocoAdapter(self.cfg.loco_checkpoint, self.device)
            print(f"[INFO] Loaded frozen Unitree G1 loco actor: {self._loco_policy.checkpoint}")
            print(f"[INFO] Loco batch interface: [N, 480] -> [N, 29], N={self.num_envs}")

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

        sky_light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != self._filter_actions.shape:
            raise ValueError(f"Expected filter actions {tuple(self._filter_actions.shape)}, got {tuple(actions.shape)}")
        self._filter_actions.copy_(actions)
        normalized = actions.clamp(-1.0, 1.0)
        self._safe_commands.copy_(
            self._command_lower + 0.5 * (normalized + 1.0) * (self._command_upper - self._command_lower)
        )
        if self._loco_policy is not None:
            terms = self._build_loco_obs_terms()
            self._append_loco_history(terms)
            history_obs = self._flatten_loco_history()
            self._loco_actions.copy_(self._loco_policy(history_obs))

    def _apply_action(self):
        targets = self._robot.data.default_joint_pos.clone()
        if self._loco_policy is not None:
            targets += self._loco_actions * self.cfg.loco_action_scale
        self._robot.set_joint_position_target(targets)

    def _build_loco_obs_terms(self) -> dict[str, torch.Tensor]:
        return {
            "base_ang_vel": self._robot.data.root_ang_vel_b * 0.2,
            "projected_gravity": self._robot.data.projected_gravity_b,
            "velocity_commands": self._safe_commands,
            "joint_pos_rel": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel_rel": self._robot.data.joint_vel * 0.05,
            "last_action": self._loco_actions,
        }

    def _append_loco_history(self, terms: dict[str, torch.Tensor]):
        if self._loco_obs_history is None:
            self._loco_obs_history = {
                name: value.unsqueeze(1).repeat(1, self._OBS_HISTORY_LENGTH, 1)
                for name, value in terms.items()
            }
            return
        for name, value in terms.items():
            self._loco_obs_history[name] = torch.cat(
                (self._loco_obs_history[name][:, 1:], value.unsqueeze(1)), dim=1
            )

    def _flatten_loco_history(self) -> torch.Tensor:
        observations = torch.cat(
            [self._loco_obs_history[name].flatten(start_dim=1) for name in self._OBS_TERM_ORDER], dim=-1
        )
        if observations.shape != (self.num_envs, UnitreeG1LocoAdapter.observation_dim):
            raise RuntimeError(f"Invalid Unitree observation shape: {tuple(observations.shape)}")
        return observations

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
        tipped = -self._robot.data.projected_gravity_b[:, 2] < math.cos(0.8)
        too_low = self._robot.data.root_pos_w[:, 2] < 0.2
        return tipped | too_low, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._filter_actions[env_ids] = 0.0
        self._safe_commands[env_ids] = 0.0
        self._loco_actions[env_ids] = 0.0

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
        joint_vel = torch.empty_like(self._robot.data.default_joint_vel[env_ids]).uniform_(-1.0, 1.0)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        if self._loco_policy is not None and self._loco_obs_history is not None:
            terms = self._build_loco_obs_terms()
            for name, value in terms.items():
                self._loco_obs_history[name][env_ids] = value[env_ids].unsqueeze(1).repeat(
                    1, self._OBS_HISTORY_LENGTH, 1
                )
