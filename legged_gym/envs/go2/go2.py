# Go2 environment subclass. Adds a foot-clearance reward so the robot lifts its
# swing feet over step edges -> lets it climb past the terrain_level plateau on stairs.
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_angle_axis
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2(LeggedRobot):
    def _reset_root_states(self, env_ids):
        super()._reset_root_states(env_ids)
        # Base env always spawns robots facing +x. On pyramid-stair tiles that forces a
        # head-on ascent attempt every reset. Random yaw spreads the start directions,
        # so robots begin at feasible (diagonal) approach angles.
        if getattr(self.cfg.init_state, 'randomize_yaw', False):
            yaw = torch_rand_float(-3.1416, 3.1416, (len(env_ids), 1), device=self.device).squeeze(1)
            z_axis = torch.tensor([0., 0., 1.], device=self.device)
            self.root_states[env_ids, 3:7] = quat_from_angle_axis(yaw, z_axis)
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _init_buffers(self):
        super()._init_buffers()
        # rigid-body states aren't acquired by the base env; we need them for foot heights
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, -1, 13)

    def _foot_terrain_heights(self, feet_xyz):
        """Terrain surface height directly under each foot (same sampling as _get_heights)."""
        if self.cfg.terrain.mesh_type in ['plane', 'none']:
            return torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)
        points = feet_xyz.clone()
        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()
        px = torch.clip(points[:, :, 0].view(-1), 0, self.height_samples.shape[0] - 2)
        py = torch.clip(points[:, :, 1].view(-1), 0, self.height_samples.shape[1] - 2)
        h = torch.min(torch.min(self.height_samples[px, py], self.height_samples[px + 1, py]),
                      self.height_samples[px, py + 1])
        return h.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _reward_feet_clearance(self):
        # foot world position + velocity from the rigid-body state tensor
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        feet_xyz = self.rigid_body_states[:, self.feet_indices, 0:3]
        feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        clearance = feet_xyz[:, :, 2] - self._foot_terrain_heights(feet_xyz)   # height above ground
        target = self.cfg.rewards.foot_clearance_target
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.            # in stance?
        foot_speed = torch.norm(feet_vel[:, :, :2], dim=2)                     # horizontal speed
        # penalize a SWING foot that is below target height while it's moving (i.e. not clearing steps)
        below = torch.clip(target - clearance, min=0.)
        return torch.sum(below * foot_speed * ~contact, dim=1)
