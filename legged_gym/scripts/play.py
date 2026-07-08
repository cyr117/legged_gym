# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, PLAY_NUM_ENVS)
    # Fewer sub-terrains = a SMALLER trimesh = fewer triangles to upload = far fewer
    # viewer segfaults on driver 595. 3x3 (9 tiles) still crashes sometimes; 2x2 (4 tiles)
    # is much safer. Drop to 1x1 if it still crashes. Flat/plane tasks ignore this.
    env_cfg.terrain.num_rows = PLAY_TERRAIN_ROWS
    env_cfg.terrain.num_cols = PLAY_TERRAIN_COLS
    env_cfg.terrain.curriculum = False
    # STAIRS-ONLY demo: cut every other terrain type (slopes, rough, discrete) so the demo
    # shows nothing but pyramid staircases -> smaller/simpler mesh AND a focused stair demo.
    # [smooth slope, rough slope, stairs up, stairs down, discrete]; harmless on flat tasks.
    if PLAY_STAIRS_ONLY and hasattr(env_cfg.terrain, 'terrain_proportions'):
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.6, 0.4, 0.0]
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # NAV tasks (go2_nav): robots follow internal GOAL POINTS, not velocity commands.
    if hasattr(env_cfg, 'goal'):
        if NAV_FIXED_GOAL is not None:
            env_cfg.goal.fixed_offset = NAV_FIXED_GOAL   # pin "point B" at env_origin+offset
            env_cfg.commands.resampling_time = 1e6        # one goal per episode: go there, stand
    # DEMO mode (velocity tasks): command every robot to walk forward instead of random commands.
    # Random commands are right for training, but for a demo they make the robot try
    # impossible maneuvers (e.g. strafe sideways on a staircase) -> looks like failure.
    elif FIXED_COMMAND:
        env_cfg.commands.resampling_time = 1e6      # never re-randomize during the demo
        env_cfg.commands.ranges.lin_vel_x = [COMMAND[0], COMMAND[0]]   # forward [m/s]
        env_cfg.commands.ranges.lin_vel_y = [COMMAND[1], COMMAND[1]]   # lateral [m/s]
        if HEADING_LOCK:
            # hold a WORLD-frame heading: yaw command is recomputed every step to steer
            # back to HEADING, so the robot can't satisfy "forward" by turning around.
            env_cfg.commands.heading_command = True
            env_cfg.commands.ranges.heading = [HEADING, HEADING]
        else:
            # body-frame demo: "walk forward wherever you face" (robot may re-orient freely)
            env_cfg.commands.heading_command = False
            env_cfg.commands.ranges.ang_vel_yaw = [COMMAND[2], COMMAND[2]] # turn [rad/s]

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if hasattr(env_cfg, 'goal'):
        env.debug_viz = DRAW_GOALS   # draw goal markers (can crash the RT renderer on newer drivers)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIXED_COMMAND = True          # True = demo: all robots walk with COMMAND below; False = random commands
    COMMAND = [1.0, 0.0, 0.0]     # [forward m/s, lateral m/s, yaw rad/s] -- edit to drive the demo
    # NOTE: HEADING_LOCK only changes the yaw COMMAND (an observation). The trained policy
    # learned (reward: lin_vel 1.0 > ang_vel 0.5) to override yaw commands when stuck on a
    # riser and turn away to regain forward velocity — that reflex is baked into the weights,
    # so on steep pits the lock has little visible effect. Changing this needs retraining
    # (e.g. stronger heading/ang_vel reward), not a demo-time command.
    HEADING_LOCK = False          # True = command a world-frame heading (policy may still override)
    HEADING = 0.0                 # [rad] world heading to hold (0 = +x, the spawn-facing direction)
    # nav tasks (go2_nav) only: pin every robot's goal at env_origin + this offset [m].
    # On stair-pit tiles the origin is the pit bottom -> [4,0] puts "point B" up/over the stairs.
    # Set to None for random goals (the training distribution).
    NAV_FIXED_GOAL = [4.0, 0.0]
    PLAY_NUM_ENVS = 8             # robots to show. >36 overflows the RT renderer's acceleration
                                  # structure and segfaults on driver 595 (50 crashes, 36 ok).
    # Trimesh terrain size for the demo. Smaller = smaller mesh = fewer viewer segfaults on
    # driver 595. 2x2 is the safe default; try 1x1 if the viewer still crashes. Ignored by
    # flat/plane tasks (go2_flat, a1_flat).
    PLAY_TERRAIN_ROWS = 1
    PLAY_TERRAIN_COLS = 1
    PLAY_STAIRS_ONLY = True        # True = show only pyramid staircases (cuts slopes/rough/discrete).
                                  # Ideal for go2_stairs / go2_nav stair demos. Harmless on flat tasks.
    DRAW_GOALS = False            # nav: draw goal markers. Per-frame line redraw forces the RT
                                  # renderer to rebuild its acceleration structure every frame,
                                  # which segfaults on driver 595. Keep False on this machine.
    args = get_args()
    play(args)
