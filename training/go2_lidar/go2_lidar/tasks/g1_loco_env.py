# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from go2_lidar.tasks.g1_loco_env_cfg import G1LocoEnvCfg


class G1LocoEnv(DirectRLEnv):
    cfg: G1LocoEnvCfg

    def __init__(self, cfg: G1LocoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.is_second_stage:
            print("second stage training.")
        else:
            print("first stage training.")

        if not self.cfg.is_second_stage:
            self._randomize_mass(-1.0, 2.0)
        else:
            self._randomize_mass(-2.0, 5.0)

        self._reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._init_physx_material_buffer()
        self._reset_physx_materials(torch.ones(self.num_envs, device="cpu", dtype=torch.bool))

        self._first_reset = True
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = [self._actions.clone()] * 5
        self._num_actions = self._actions.shape[1]
        self._controlled_joint_ids = list(range(self._num_actions))
        if len(self._robot.joint_names) != self._num_actions:
            raise ValueError(
                "G1 locomotion expects the no-finger 29dof asset."
                f" Got {len(self._robot.joint_names)} joints, but action space is {self._num_actions}."
            )

        self._base_id_cs, _ = self._contact_sensor.find_bodies("torso_link")
        self._feet_ids_cs, _ = self._contact_sensor.find_bodies(".*_ankle_roll_link")
        self._undesired_contact_body_ids_cs, _ = self._contact_sensor.find_bodies(
            "torso_link|.*_hip_.*|.*_knee.*|.*_shoulder.*|.*_elbow.*"
        )
        self._feet_ids_bd, _ = self._robot.find_bodies(".*_ankle_roll_link")
        self._ankle_ids_jt, _ = self._robot.find_joints(".*_ankle_.*")
        self._hip_ids_jt = self._find_joints(".*_hip_(yaw|roll)_joint")
        self._leg_ids_jt = self._find_joints(".*_hip_.*|.*_knee_joint")
        self._energy_ids_jt = self._find_joints(".*_hip_.*|.*_knee_joint|.*_ankle_.*")
        self._arm_ids_jt = self._find_joints(
            ".*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|"
            ".*_elbow_joint|.*_elbow_pitch_joint|.*_elbow_roll_joint|.*_wrist_roll_joint|"
            ".*_wrist_pitch_joint|.*_wrist_yaw_joint"
        )
        self._torso_ids_jt = self._find_joints("torso_joint|waist_.*_joint")

        self._episode_sums = {}
        self._step_counter = 0

        self._cmd_lin_vel = torch.zeros(self.num_envs, 2, device=self.device)
        self._cmd_ang_vel = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_intervals = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_resample_accums = torch.zeros(self.num_envs, 1, device=self.device)
        self._cmd_lower = torch.tensor(
            [
                self.cfg.cmd_lin_vel_x_range[0],
                self.cfg.cmd_lin_vel_y_range[0],
                self.cfg.cmd_ang_vel_z_range[0],
            ],
            device=self.device,
        )
        self._cmd_upper = torch.tensor(
            [
                self.cfg.cmd_lin_vel_x_range[1],
                self.cfg.cmd_lin_vel_y_range[1],
                self.cfg.cmd_ang_vel_z_range[1],
            ],
            device=self.device,
        )
        self._reset_commands()

        self._show_debug_viz = False
        if self.cfg.is_play_env:
            print("\n********************************************")
            print("           **Enabling debug draw.**")
            print("********************************************\n")
            self._show_debug_viz = True
            # fmt: off
            import isaacsim
            from isaacsim.core.utils.extensions import enable_extension
            enable_extension("omni.isaac.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw
            # fmt: on
            self._debug_draw = _debug_draw.acquire_debug_draw_interface()

            goal_vel_viz_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/velocity_goal")
            goal_vel_viz_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
            self._goal_vel_viz = VisualizationMarkers(cfg=goal_vel_viz_cfg)
            self._goal_vel_viz.set_visibility(True)

        self._update_debug_draw()

        self._track_env_id = 0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.num_terrain_rows = 10
        self.num_terrain_cols = 20

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.cfg.terrain.terrain_generator.num_rows = self.num_terrain_rows
        self.cfg.terrain.terrain_generator.num_cols = self.num_terrain_cols
        self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)

        rand_cols = torch.randint(0, self.num_terrain_cols, size=(self.num_envs,), device=self.device)
        self._env_terrain_cols = rand_cols
        self._terrain.env_origins[:] = self._terrain.terrain_origins[0, rand_cols]
        self._terrain.terrain_levels[:] = 0

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        sky_light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        )
        sky_light_cfg.func("/World/skyLight", sky_light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._step_counter += 1

        self._update_debug_draw()

        if self.cfg.is_play_env and self.viewport_camera_controller is not None:
            lookat_pos = self._robot.data.root_pos_w[self._track_env_id].cpu()
            eye_pos = lookat_pos.clone()
            eye_pos[0] += 2.0
            eye_pos[1] += 2.0
            eye_pos[2] += 1.0
            self.viewport_camera_controller.update_view_location(eye=eye_pos, lookat=lookat_pos)

        self._previous_actions.append(self._actions.clone())
        self._previous_actions.pop(0)
        self._actions = actions.clone()

        self._cmd_resample_accums += self.step_dt
        self._reset_commands(self._cmd_resample_accums >= self._cmd_resample_intervals)

    def _apply_action(self):
        action_scaled = self._actions * self.cfg.action_scale
        joint_pos_target = self._robot.data.default_joint_pos.clone()
        joint_pos_target[:, self._controlled_joint_ids] += action_scaled
        self._robot.set_joint_position_target(joint_pos_target)

    def _get_observations(self) -> dict:
        contact_indicators = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 1.0
        contact_indicators = contact_indicators.float()

        obs_buf = torch.cat(
            [
                self._robot.data.root_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self._cmd_lin_vel * 2.0,
                self._cmd_ang_vel * 0.25,
                self._robot.data.joint_pos[:, self._controlled_joint_ids]
                - self._robot.data.default_joint_pos[:, self._controlled_joint_ids],
                self._robot.data.joint_vel[:, self._controlled_joint_ids] * 0.05,
                self._actions,
            ],
            dim=-1,
        )

        critic_obs_buf = torch.cat(
            [
                obs_buf,
                self._robot.data.root_lin_vel_b * 2.0,
                contact_indicators,
                self._robot.data.applied_torque[:, self._controlled_joint_ids],
                self._robot.data.body_lin_vel_w[:, self._feet_ids_bd, :].reshape(self.num_envs, -1),
                self._robot.data.body_pos_w[:, self._feet_ids_bd, :].reshape(self.num_envs, -1),
            ],
            dim=-1,
        )

        noise_buf = torch.cat(
            [
                torch.ones(3) * 0.2,
                torch.ones(3) * 0.05,
                torch.zeros(3),
                torch.ones(self._num_actions) * 0.01,
                torch.ones(self._num_actions) * 1.5 * 0.5,
                torch.zeros(self._num_actions),
            ],
            dim=0,
        )
        obs_buf += (torch.rand_like(obs_buf) * 2.0 - 1.0) * noise_buf.to(self.device)

        return {"policy": obs_buf, "critic": critic_obs_buf}

    def _get_rewards(self) -> torch.Tensor:
        cmd = torch.cat([self._cmd_lin_vel, self._cmd_ang_vel], dim=-1)
        cmd_norm = torch.linalg.norm(cmd, dim=1)

        vel_yaw = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self._robot.data.root_quat_w),
            self._robot.data.root_lin_vel_w[:, :3],
        )
        lin_vel_error = torch.sum(torch.square(self._cmd_lin_vel - vel_yaw[:, :2]), dim=1)
        r_track_lin_vel_xy = torch.exp(-lin_vel_error / 0.25)
        r_track_lin_vel_xy *= 1.0

        ang_vel_error = torch.square(self._cmd_ang_vel[:, 0] - self._robot.data.root_ang_vel_b[:, 2])
        r_track_ang_vel_z = torch.exp(-ang_vel_error / 0.25)
        r_track_ang_vel_z *= 0.5

        r_alive = (~self.reset_terminated).float()
        r_alive *= 0.15

        r_base_linear_velocity = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        r_base_linear_velocity *= -2.0

        r_base_angular_velocity = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        r_base_angular_velocity *= -0.05

        r_joint_vel = torch.sum(torch.square(self._robot.data.joint_vel[:, self._controlled_joint_ids]), dim=1)
        r_joint_vel *= -0.001

        r_joint_acc = torch.sum(torch.square(self._robot.data.joint_acc[:, self._controlled_joint_ids]), dim=1)
        r_joint_acc *= -2.5e-7

        r_action_rate = torch.sum(torch.square(self._actions - self._previous_actions[-1]), dim=1)
        r_action_rate *= -0.05

        ankle_pos = self._robot.data.joint_pos[:, self._ankle_ids_jt]
        ankle_pos_limits = self._robot.data.soft_joint_pos_limits[:, self._ankle_ids_jt]
        out_of_limits = -(ankle_pos - ankle_pos_limits[..., 0]).clip(max=0.0)
        out_of_limits += (ankle_pos - ankle_pos_limits[..., 1]).clip(min=0.0)
        r_dof_pos_limits = torch.sum(out_of_limits, dim=1)
        r_dof_pos_limits *= -5.0

        r_energy = torch.sum(
            torch.abs(self._robot.data.applied_torque[:, self._energy_ids_jt])
            * torch.abs(self._robot.data.joint_vel[:, self._energy_ids_jt]),
            dim=1,
        )
        r_energy *= -2.0e-5

        r_joint_deviation_arms = self._joint_deviation_l1(self._arm_ids_jt)
        r_joint_deviation_arms *= -0.1

        r_joint_deviation_waists = self._joint_deviation_l1(self._torso_ids_jt)
        r_joint_deviation_waists *= -1.0

        r_joint_deviation_legs = self._joint_deviation_l1(self._hip_ids_jt)
        r_joint_deviation_legs *= -1.0

        r_flat_orientation_l2 = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        r_flat_orientation_l2 *= -5.0

        r_base_height = torch.square(self._robot.data.root_pos_w[:, 2] - 0.78)
        r_base_height *= -10.0

        contact = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids_cs, :], dim=-1) > 1.0
        r_feet_gait = self._feet_gait_reward(contact, period=0.8, threshold=0.55)
        r_feet_gait *= 0.5
        r_feet_gait[cmd_norm < 0.1] = 0.0

        feet_vel = self._robot.data.body_lin_vel_w[:, self._feet_ids_bd, :]
        r_feet_slide = torch.sum(torch.norm(feet_vel[..., :2], dim=-1) * contact, dim=1)
        r_feet_slide *= -0.2

        foot_z = self._robot.data.body_pos_w[:, self._feet_ids_bd, 2] - self._terrain.env_origins[:, 2].unsqueeze(-1)
        foot_xy_speed = torch.norm(feet_vel[..., :2], dim=-1)
        clearance_error = torch.square(foot_z - 0.1) * torch.tanh(foot_xy_speed)
        r_feet_clearance = torch.exp(-torch.sum(clearance_error, dim=1) / 0.05)
        r_feet_clearance *= 1.0
        r_feet_clearance[cmd_norm < 0.1] = 0.0

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        r_undesired_contacts = torch.sum(
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids_cs], dim=-1), dim=1)[0]
            > 1.0,
            dim=1,
        )
        r_undesired_contacts = r_undesired_contacts.float()
        r_undesired_contacts *= -1.0

        r_termination = self._reset_buf.float()
        r_termination *= -200.0

        rewards = {k: v * self.step_dt for k, v in locals().items() if k.startswith("r_")}

        for k, v in rewards.items():
            if k not in self._episode_sums:
                self._episode_sums[k] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            if v.shape != self._episode_sums[k].shape:
                raise ValueError(f"reward {k} has wrong shape: {v.shape}, expected: {self._episode_sums[k].shape}")
            self._episode_sums[k] += v

        return torch.sum(torch.stack(list(rewards.values())), dim=0)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id_cs], dim=-1), dim=1)[0] > 1.0,
            dim=1,
        )
        died |= -self._robot.data.projected_gravity_b[:, 2] < 0.25
        self._reset_buf[:] = died

        if torch.any(self._robot.data.root_pos_w[:, 2] < -5.0):
            raise RuntimeError("robot is falling down!")

        return died, time_out

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        env_mask[env_ids] = True

        if not self._first_reset:
            distance = torch.norm(
                self._robot.data.root_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2], dim=1
            )
            move_up = distance > self.cfg.terrain.terrain_generator.size[0] * 0.5
            move_down = distance < 1.0
            move_down *= ~move_up
            self._terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
            self._terrain.terrain_levels[env_ids] = torch.where(
                self._terrain.terrain_levels[env_ids] >= self.num_terrain_rows,
                torch.randint_like(self._terrain.terrain_levels[env_ids], self.num_terrain_rows),
                torch.clip(self._terrain.terrain_levels[env_ids], 0),
            )
            self._terrain.env_origins[env_ids, :] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
            ]
        else:
            self._first_reset = False

        if self.cfg.is_play_env:
            self._terrain.terrain_levels[0] = self.num_terrain_rows // 2
            self._env_terrain_cols[0] = self.num_terrain_cols // 2
            self._terrain.env_origins[env_ids, :] = self._terrain.terrain_origins[
                self._terrain.terrain_levels[env_ids], self._env_terrain_cols[env_ids], :
            ]

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        for actions in self._previous_actions:
            actions[env_ids] = 0.0

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        zeros = torch.zeros(default_root_state.shape[0], device=self.device)
        rand_yaw = torch.rand_like(zeros) * math.pi * 2.0 - math.pi
        rand_quats = math_utils.quat_from_euler_xyz(zeros, zeros, rand_yaw)
        default_root_state[:, 3:7] = math_utils.quat_mul(default_root_state[:, 3:7], rand_quats)
        self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_pos *= math_utils.sample_uniform(0.8, 1.2, default_joint_pos.shape, default_joint_pos.device)
        joint_pos_limits = self._robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = default_joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
        self._robot.write_joint_state_to_sim(joint_pos, self._robot.data.default_joint_vel[env_ids], env_ids=env_ids)

        self._reset_commands(env_mask)

        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Episode/average_speed"] = torch.mean(self._robot.data.root_lin_vel_b.norm(dim=1))
        extras["Episode/average_terrain_level"] = torch.mean(self._terrain.terrain_levels.to(torch.float))
        extras["Episode/max_terrain_level"] = torch.max(self._terrain.terrain_levels.to(torch.float))
        self.extras["log"] = extras

    def _reset_commands(self, masks: torch.Tensor | None = None):
        if masks is None:
            masks = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        masks = masks.squeeze()

        num_resets = torch.count_nonzero(masks).item()
        if num_resets == 0:
            return

        new_commands = math_utils.sample_uniform(
            self._cmd_lower, self._cmd_upper, (num_resets, 3), device=self.device
        )
        standing_commands = torch.rand(num_resets, device=self.device) < self.cfg.standing_command_probability
        new_commands[standing_commands] = 0.0
        self._cmd_lin_vel[masks, :2] = new_commands[:, :2]
        self._cmd_ang_vel[masks, 0] = new_commands[:, 2]

        self._cmd_resample_intervals[masks] = math_utils.sample_uniform(
            self.cfg.cmd_resample_interval[0], self.cfg.cmd_resample_interval[1], (num_resets, 1), self.device
        )
        self._cmd_resample_accums[masks] = torch.zeros(num_resets, 1, device=self.device)

    def _joint_deviation_l1(self, joint_ids: list[int]) -> torch.Tensor:
        if len(joint_ids) == 0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.sum(
            torch.abs(
                self._robot.data.joint_pos[:, joint_ids] - self._robot.data.default_joint_pos[:, joint_ids]
            ),
            dim=1,
        )

    def _find_joints(self, pattern: str) -> list[int]:
        try:
            joint_ids, _ = self._robot.find_joints(pattern)
        except ValueError:
            return []
        return joint_ids

    def _feet_gait_reward(self, contact: torch.Tensor, period: float, threshold: float) -> torch.Tensor:
        phase = torch.remainder(self.episode_length_buf.float() * self.step_dt / period, 1.0)
        desired_left = phase < threshold
        desired_right = torch.remainder(phase + 0.5, 1.0) < threshold
        desired_contact = torch.stack([desired_left, desired_right], dim=1)
        if contact.shape[1] != 2:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.mean((contact == desired_contact).float(), dim=1)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self._goal_vel_viz.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        base_quat_w = self._robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        return arrow_scale, arrow_quat

    def _update_debug_draw(self):
        if not self._show_debug_viz:
            return

        self._debug_draw.clear_lines()
        self._debug_draw.clear_points()

        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._cmd_lin_vel[:, :2])
        self._goal_vel_viz.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)

    def _init_physx_material_buffer(self):
        self._num_shapes_per_body = []
        for link_path in self._robot.root_physx_view.link_paths[0]:
            link_physx_view = self._robot._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
            self._num_shapes_per_body.append(link_physx_view.max_shapes)
        num_shapes = sum(self._num_shapes_per_body)
        expected_shapes = self._robot.root_physx_view.max_shapes
        if num_shapes != expected_shapes:
            raise ValueError(
                "Failed to parse the number of shapes per body."
                f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
            )

        self._num_physx_mat_buckets = 64
        range_list = [self.cfg.static_friction_range, self.cfg.dynamic_friction_range, self.cfg.restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self._material_buckets = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (self._num_physx_mat_buckets, 3), device="cpu"
        )

        self._physics_parameters = torch.zeros(self.num_envs, 3, device=self.device)

    def _reset_physx_materials(self, env_ids):
        bucket_ids = torch.randint(0, self._num_physx_mat_buckets, (len(env_ids),), device="cpu")
        material_samples = self._material_buckets[bucket_ids]
        self._physics_parameters[env_ids, :] = material_samples.to(self.device)
        materials = self._robot.root_physx_view.get_material_properties()
        materials[env_ids, :] = material_samples.reshape(-1, 1, 3)
        self._robot.root_physx_view.set_material_properties(materials, env_ids)

    def _randomize_mass(self, min_delta, max_delta):
        asset = self._robot

        env_ids = torch.arange(self.num_envs, device="cpu")

        body_ids, _ = self._robot.find_bodies("torso_link")
        body_ids = torch.tensor(body_ids, dtype=torch.int, device="cpu")
        assert body_ids.shape == (1,), "Mass randomization is only supported for the torso body."

        masses = asset.root_physx_view.get_masses()
        print("*" * 50)
        print(f"masses before randomization: {masses[env_ids, body_ids]}")
        print("*" * 50)

        rand_samples = math_utils.sample_uniform(min_delta, max_delta, (len(env_ids), len(body_ids)), device="cpu")
        masses[env_ids[:, None], body_ids] += rand_samples

        print("*" * 50)
        print(f"min mass: {masses[env_ids, body_ids].min()}")
        print(f"max mass: {masses[env_ids, body_ids].max()}")
        print("*" * 50)

        asset.root_physx_view.set_masses(masses, env_ids)
