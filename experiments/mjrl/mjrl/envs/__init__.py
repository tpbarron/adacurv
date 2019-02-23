from gym.envs.registration import register

# ----------------------------------------
# mjrl environments
# ----------------------------------------

register(
    id='mjrl_point_mass-v0',
    entry_point='mjrl.envs:PointMassEnv',
    max_episode_steps=25,
)

register(
    id='mjrl_swimmer-v0',
    entry_point='mjrl.envs:SwimmerEnv',
    max_episode_steps=500,
)

register(
    id='BasketballEnv-v0',
    entry_point='bbbot.basketball_env:BasketballEnv',
    max_episode_steps=50,
)

register(
    id='BasketballEnvRendered-v0',
    entry_point='bbbot.basketball_env:BasketballEnvRendered',
    max_episode_steps=50,
)


register(
    id='BasketballEnvRandomHoop-v0',
    entry_point='bbbot.basketball_env:BasketballEnvRandomHoop',
    max_episode_steps=50,
)

register(
    id='BasketballEnvRandomHoopRendered-v0',
    entry_point='bbbot.basketball_env:BasketballEnvRandomHoopRendered',
    max_episode_steps=50
)


### -- unused

register(
    id='BasketballEnvHard-v0',
    entry_point='bbbot.basketball_env_hard:BasketballEnvHard',
    max_episode_steps=250,
)

register(
    id='BasketballEnvHardRendered-v0',
    entry_point='bbbot.basketball_env_hard:BasketballEnvHardRendered',
    max_episode_steps=250,
)

# from mjrl.envs.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
# from mjrl.envs.point_mass import PointMassEnv
# from mjrl.envs.swimmer import SwimmerEnv
