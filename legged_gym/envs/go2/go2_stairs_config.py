# Go2 STAIRS task, tuned to climb UP with a STRAIGHT posture (level torso, facing
# forward) instead of the "corkscrew / sideways" shortcut.
#
# WHY the robot corkscrews if you don't do this: climbing a tall step straight-on is
# hard, so if the command distribution ALLOWS sideways velocity (lin_vel_y) and any
# heading, the policy discovers it can satisfy "move" by strafing/spinning up the stairs
# diagonally -- easier, but ugly and not what we want. The cure is to REMOVE that freedom:
# command forward-only, no strafe, hold a fixed heading. Then the only way to earn the
# tracking reward is to walk straight up with a level body -> it must find the leg motion
# (front feet reaching high onto each step) that makes straight ascent possible.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go2.go2_config import Go2RoughCfg, Go2RoughCfgPPO


class Go2StairsCfg( Go2RoughCfg ):
    class terrain( Go2RoughCfg.terrain ):
        # Focus on GOING UP: 100% pyramid stairs-UP (center is a pit, robot climbs outward).
        # [smooth slope, rough slope, stairs up, stairs down, discrete]. Add some 'down'
        # back later (e.g. [0,0,0.8,0.2,0]) once straight ascent is solid.
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        max_init_terrain_level = 0   # start on the smallest steps; curriculum raises height

    class init_state( Go2RoughCfg.init_state ):
        # Spawn facing +x (NOT random yaw). Every robot then climbs straight up the same
        # face, perpendicular to the step edges -- a clean, consistent "straight up" signal.
        # (Random yaw + forward command = robot hits stairs at a random angle, which is what
        #  invited the diagonal/corkscrew climb.) Re-enable for robustness once it's learned.
        randomize_yaw = False

    class commands( Go2RoughCfg.commands ):
        heading_command = True        # ang_vel_yaw is auto-computed to hold the heading below
        class ranges( Go2RoughCfg.commands.ranges ):
            lin_vel_x = [0.3, 1.0]    # ALWAYS walk forward (no standing, no backing up)
            lin_vel_y = [0.0, 0.0]    # NO sideways strafing -> kills the "climb sideways" shortcut
            ang_vel_yaw = [0.0, 0.0]  # (overwritten by heading controller; kept 0 for safety)
            heading = [-0.2, 0.2]     # hold ~forward heading (+-11 deg) -> kills the corkscrew spin

    class rewards( Go2RoughCfg.rewards ):
        # THE key change for straight ascent: lift the swing feet HIGH ENOUGH to land ON the
        # next step. Steps reach ~0.19 m (step_height = 0.05 + 0.18*row/num_rows), but the old
        # target was only 0.11 m -> feet stubbed into tall risers -> straight ascent impossible
        # -> the policy corkscrewed (a diagonal path has a smaller effective rise per stride).
        # 0.18 m lets the front feet clear and reach onto steps up to the high curriculum rows,
        # so the LEGS (not a spiral) can carry the body straight up.
        foot_clearance_target = 0.18   # was 0.11
        class scales( Go2RoughCfg.rewards.scales ):
            feet_clearance = -3.0     # raised from -2.5: push harder for the high foot lift
            # --- posture / anti-spin ---
            tracking_ang_vel = 1.0    # raised from 0.5: strongly HOLD heading -> no spinning up
            ang_vel_xy = -0.1         # raised from -0.05: steadier body, less roll/pitch wobble
            # orientation -5.0 (inherited): keep the TORSO LEVEL = the "straight/upright posture".
            #   Body can't pitch over + feet lift high -> the LEGS must carry it straight up.
            # tracking_lin_vel 1.0 (inherited): reward forward progress up the stairs.


class Go2StairsCfgPPO( Go2RoughCfgPPO ):
    class runner( Go2RoughCfgPPO.runner ):
        run_name = ''
        experiment_name = 'stairs_go2'   # logs/stairs_go2/
