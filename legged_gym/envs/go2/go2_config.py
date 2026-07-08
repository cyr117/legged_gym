# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Unitree Go2 config for legged_gym (rough terrain). Mirrors the A1 setup
# (same FL/FR/RL/RR hip/thigh/calf joint layout). Asset from unitree_ros
# go2_description, mesh paths converted from package:// to relative.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2RoughCfg( LeggedRobotCfg ):
    class terrain( LeggedRobotCfg.terrain ):
        # Laptop-friendly rough terrain. Full default (10x20 sub-terrains, 25 m border)
        # allocates ~5.5 GB just for the collision mesh and won't fit a 6 GB GPU, and its
        # huge mesh crashes the viewer. Shrinking keeps rough-terrain learning while fitting.
        mesh_type = 'trimesh'   # trimesh/heightfield = rough; still uses the curriculum
        num_rows = 5            # difficulty levels (curriculum) -- was 10
        num_cols = 5            # terrain types -- was 20
        border_size = 5         # [m] -- was 25
        curriculum = True       # start easy, auto-advance to harder rows as robots succeed
        max_init_terrain_level = 0  # START on the easiest row so a from-scratch policy can bootstrap
        measure_heights = True  # robot senses terrain -> +187 numbers in observation (obs=235)
        # Stage 2 (after walking is learned): rebalance toward stairs to drill stair-climbing.
        # Stage 1 used [0.4,0.4,0.1,0.05,0.05] (80% slopes) to bootstrap basic walking.
        # [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.35, 0.15, 0.1]   # now 50% stairs

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        randomize_yaw = True   # spawn at random heading (not always +x head-on into stairs)
        default_joint_angles = { # target angles [rad] when action = 0.0
            # symmetric front/rear pose (A1's asymmetric rear thigh pitched Go2 forward)
            'FL_hip_joint': 0.1,
            'RL_hip_joint': 0.1,
            'FR_hip_joint': -0.1,
            'RR_hip_joint': -0.1,

            'FL_thigh_joint': 0.8,
            'RL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,
            'RR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'RL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,
            'RR_calf_joint': -1.5,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'P'
        # Go2 base is ~6.9 kg (heavier than A1); stiffness 20 sagged the stance
        # to 0.226 m. 40 holds it near its ~0.27 m nominal height.
        stiffness = {'joint': 40.}   # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        action_scale = 0.25
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_description.urdf'
        name = "go2"
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.28   # Go2 nominal stance height (measured ~0.27 m)
        foot_clearance_target = 0.11  # [m] raised from 0.08: clear the taller risers when climbing UP
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.00005      # softened from -0.0002: harsh movement penalty caused stand-still
            dof_pos_limits = -10.0
            orientation = -5.0      # keep body level (base default 0 -> could sprawl/not commit to walking)
            feet_air_time = 2.0     # reward taking steps -> breaks the "stand still, don't fall" optimum
            feet_clearance = -2.5   # raised from -1.0: stronger foot-lift so it clears ascending step edges
            lin_vel_z = -1.0        # softened from -2.0: going UP stairs REQUIRES upward body velocity

class Go2RoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2'
