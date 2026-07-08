# Go2 goal-reaching ("local navigation") task, in the spirit of Rudin et al. 2022
# "Advanced Skills by Learning Locomotion and Local Navigation End-to-End" (arXiv 2209.12827).
#
# Instead of tracking velocity commands, each robot gets a TARGET POINT (e.g. a spot that
# requires climbing the stairs). Rewards pay for progress toward / arrival at the point;
# heading, gait and approach angle are left entirely to the policy ("any posture").
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import torch_rand_float, quat_apply
import torch
from legged_gym.envs.go2.go2 import Go2


class Go2Nav(Go2):
    def _init_buffers(self):
        super()._init_buffers()
        self.goal_pos = torch.zeros(self.num_envs, 2, device=self.device)         # world-frame goal xy
        self.goal_dist = torch.zeros(self.num_envs, device=self.device)           # current distance to goal
        self.last_goal_dist = torch.zeros(self.num_envs, device=self.device)      # distance at previous step
        self.goal_progress = torch.zeros(self.num_envs, device=self.device)       # per-step progress [m]
        self.at_goal_time = torch.zeros(self.num_envs, device=self.device)        # [s] spent at goals this episode

    def _resample_commands(self, env_ids):
        """Sample a new goal point for these envs: random direction, random distance from
        the robot's CURRENT position (called right after teleport on resets)."""
        if len(env_ids) == 0:
            return
        if self.cfg.goal.fixed_offset is not None:
            # demo mode: goal pinned at a fixed offset from the terrain-tile origin ("point B")
            self.goal_pos[env_ids, 0] = self.env_origins[env_ids, 0] + self.cfg.goal.fixed_offset[0]
            self.goal_pos[env_ids, 1] = self.env_origins[env_ids, 1] + self.cfg.goal.fixed_offset[1]
        else:
            n = len(env_ids)
            theta = torch_rand_float(-3.1416, 3.1416, (n, 1), device=self.device).squeeze(1)
            d = torch_rand_float(self.cfg.goal.dist_range[0], self.cfg.goal.dist_range[1],
                                 (n, 1), device=self.device).squeeze(1)
            self.goal_pos[env_ids, 0] = self.root_states[env_ids, 0] + d * torch.cos(theta)
            self.goal_pos[env_ids, 1] = self.root_states[env_ids, 1] + d * torch.sin(theta)
        # commands buffer stays zero (obs slot is replaced by the goal vector)
        self.commands[env_ids, :] = 0.
        dist = torch.norm(self.goal_pos[env_ids] - self.root_states[env_ids, :2], dim=1)
        self.last_goal_dist[env_ids] = dist
        self.goal_dist[env_ids] = dist

    def _post_physics_step_callback(self):
        # periodic goal resampling (a robot standing at its goal gets a fresh one)
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0)\
            .nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        # goal bookkeeping: progress made this step (used by the reward)
        cur = torch.norm(self.goal_pos - self.root_states[:, :2], dim=1)
        self.goal_progress = self.last_goal_dist - cur
        self.last_goal_dist = cur
        self.goal_dist = cur
        self.at_goal_time += (cur < self.cfg.goal.reach_radius) * self.dt   # success signal for curriculum

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)   # curriculum (inside) reads at_goal_time BEFORE we zero it
        self.at_goal_time[env_ids] = 0.

    def compute_observations(self):
        """Same layout as the base env, but the 3 command slots become the goal vector:
        [unit direction to goal in heading frame (2), clipped distance (1)]."""
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        dx = self.goal_pos[:, 0] - self.root_states[:, 0]
        dy = self.goal_pos[:, 1] - self.root_states[:, 1]
        gx = torch.cos(heading) * dx + torch.sin(heading) * dy    # goal in robot heading frame
        gy = -torch.sin(heading) * dx + torch.cos(heading) * dy
        dist = torch.sqrt(gx**2 + gy**2).clamp(min=1e-5)
        goal_obs = torch.stack([gx / dist, gy / dist, torch.clip(dist, 0., 5.) / 5.], dim=1)

        self.obs_buf = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                                  self.base_ang_vel * self.obs_scales.ang_vel,
                                  self.projected_gravity,
                                  goal_obs,
                                  (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                  self.dof_vel * self.obs_scales.dof_vel,
                                  self.actions
                                  ), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _update_terrain_curriculum(self, env_ids):
        """Success-based curriculum: promote robots that actually reached their goals
        (spent time inside reach_radius), demote those that never arrived. Distance-walked
        is a bad proxy here -- a robot can visit 2 goals and still end near its spawn."""
        if not self.init_done:
            return
        move_up = self.at_goal_time[env_ids] > 1.0                # reliably reached goal(s)
        move_down = (self.at_goal_time[env_ids] < 0.2) * ~move_up # never really arrived
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids] >= self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0))
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _draw_debug_vis(self):
        """Draw each robot's goal as a red wireframe sphere (enable with env.debug_viz=True)."""
        self.gym.clear_lines(self.viewer)
        sphere = gymutil.WireframeSphereGeometry(self.cfg.goal.reach_radius, 12, 12, None, color=(1, 0, 0))
        goal_xyz = torch.cat([self.goal_pos, torch.zeros_like(self.goal_pos[:, :1])], dim=1).unsqueeze(1)
        ground_z = self._foot_terrain_heights(goal_xyz).squeeze(1)   # terrain height under each goal
        for i in range(self.num_envs):
            pose = gymapi.Transform(gymapi.Vec3(self.goal_pos[i, 0].item(),
                                                self.goal_pos[i, 1].item(),
                                                ground_z[i].item() + 0.3), r=None)
            gymutil.draw_lines(sphere, self.gym, self.viewer, self.envs[i], pose)

    # ---------------- navigation rewards ----------------
    def _reward_goal_progress(self):
        # speed toward the goal [m/s], clipped (progress/dt); replaces velocity tracking
        return torch.clip(self.goal_progress / self.dt, -2., 2.)

    def _reward_goal_reached(self):
        # continuous bonus while within reach_radius -> arrive AND stay
        return (self.goal_dist < self.cfg.goal.reach_radius).float()

    def _reward_time_penalty(self):
        # the clock runs while NOT at the goal: every step away from it costs.
        # (scale is negative) -> arriving faster = fewer penalized steps.
        return (self.goal_dist >= self.cfg.goal.reach_radius).float()

    def _reward_feet_air_time(self):
        # Base version gates this on the VELOCITY-COMMAND norm -- which is always zero in
        # nav, silently killing the term. Same computation, but gate on "traveling to a
        # goal" instead: reward steps while away from the goal, none while standing at it.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= self.goal_dist > self.cfg.goal.reach_radius   # traveling, not at goal
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # Base version penalizes motion at zero COMMAND (always true in nav -> would fire
        # even mid-walk). Nav semantics: penalize fidgeting only while AT the goal.
        # Inert until a scale is set in the config.
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) \
            * (self.goal_dist < self.cfg.goal.reach_radius)
