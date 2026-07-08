# Experiment: identical to go2_flat EXCEPT base_height_target 0.28 -> 0.20.
# Demonstrates that one reward parameter changes the learned behavior:
# this Go2 learns to walk in a low crouch. Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.go2.go2_flat_config import Go2FlatCfg, Go2FlatCfgPPO

class Go2CrouchCfg( Go2FlatCfg ):
    class rewards( Go2FlatCfg.rewards ):
        base_height_target = 0.20   # <-- walk low instead of 0.28
        class scales( Go2FlatCfg.rewards.scales ):
            base_height = -10.0     # firm weight so the target actually bites (see note below)

class Go2CrouchCfgPPO( Go2FlatCfgPPO ):
    class runner( Go2FlatCfgPPO.runner ):
        experiment_name = 'go2_crouch'   # separate log dir so it won't touch go2_flat
