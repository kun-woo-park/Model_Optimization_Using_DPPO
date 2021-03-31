from gym.envs.registration import register

register(
    id='acav-v0',
    entry_point='gym_Aircraft.envs:AircraftEnv',
    max_episode_steps=200,
    reward_threshold=195.0,

)

