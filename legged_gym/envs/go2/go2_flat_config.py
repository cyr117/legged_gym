# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Unitree Go2 flat-terrain config. Flat 'plane' terrain so the Isaac Gym viewer
# works on newer NVIDIA drivers (trimesh terrain crashes the old GL renderer).
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO

class Go2FlatCfg( Go2RoughCfg ):
    class env( Go2RoughCfg.env ):
        num_observations = 48   # rough (235) minus the 187 terrain height samples

    class terrain( Go2RoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False

    class rewards( Go2RoughCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.28
        class scales( Go2RoughCfg.rewards.scales ):
            orientation = -5.0      # penalize tilted back
            base_height = -10.0     # penalize crouching below 0.28 m -> stops the prone sprawl
            torques = -0.000025
            feet_air_time = 2.
            feet_clearance = 0.     # flat ground needs no high foot-lift (that's for stairs) -> keep natural gait

    class commands( Go2RoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        class ranges( Go2RoughCfg.commands.ranges ):
            ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( Go2RoughCfg.domain_rand ):
        friction_range = [0., 1.5]

class Go2FlatCfgPPO( Go2RoughCfgPPO ):
    class policy( Go2RoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu'

    class algorithm( Go2RoughCfgPPO.algorithm ):
        entropy_coef = 0.01

    class runner( Go2RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'flat_go2'
        load_run = -1
        max_iterations = 300
