# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Flat-terrain variant of the A1 task.
# Created for machines whose (newer) NVIDIA OpenGL driver crashes Isaac Gym's
# viewer when a large trimesh terrain is rendered. Flat 'plane' terrain avoids
# the crash, so this task can be trained AND watched in the viewer.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO

class A1FlatCfg( A1RoughCfg ):
    class env( A1RoughCfg.env ):
        num_observations = 48   # rough (235) minus the 187 terrain height samples

    class terrain( A1RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class rewards( A1RoughCfg.rewards ):
        max_contact_force = 350.
        class scales( A1RoughCfg.rewards.scales ):
            orientation = -5.0
            torques = -0.000025
            feet_air_time = 2.

    class commands( A1RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( A1RoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( A1RoughCfg.domain_rand ):
        friction_range = [0., 1.5]

class A1FlatCfgPPO( A1RoughCfgPPO ):
    class policy( A1RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm( A1RoughCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( A1RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_a1'
        load_run = -1
        max_iterations = 300
