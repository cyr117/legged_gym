# Config for the Go2 goal-reaching task (see go2_nav.py).
# Velocity-tracking rewards are OFF; goal progress/arrival rewards are ON.
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO


class Go2NavCfg( Go2RoughCfg ):
    class goal:
        dist_range = [2.0, 6.0]   # [m] goal sampled this far from the robot, random direction
        reach_radius = 0.5        # [m] counts as "arrived" within this radius
        fixed_offset = None       # e.g. [4., 0.]: goal pinned at env_origin+offset (demo); None = random

    class commands( Go2RoughCfg.commands ):
        heading_command = False   # no heading constraint -- "any posture"
        resampling_time = 8.      # [s] new goal every 8 s (2-3 goals per 20 s episode)

    class rewards( Go2RoughCfg.rewards ):
        class scales( Go2RoughCfg.rewards.scales ):
            # navigation objective replaces velocity tracking
            tracking_lin_vel = 0.
            tracking_ang_vel = 0.
            goal_progress = 1.5   # ~speed toward goal [m/s], the main driver
            goal_reached = 1.0    # continuous while standing within reach_radius
            time_penalty = -0.6   # per-step cost while NOT at the goal -> rewards arriving fast.
                                  # raised from -0.25: total progress payout is path-independent,
                                  # so this is the ONLY term that penalizes slow/spiral routes.
                                  # if success rate drops (rushing -> falls), back off to ~-0.4
            # locomotion regularizers inherited from Go2RoughCfg
            # (orientation -5, feet_air_time 2, feet_clearance -2.5, torques, etc.)


class Go2NavCfgPPO( Go2RoughCfgPPO ):
    class runner( Go2RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'nav_go2'
