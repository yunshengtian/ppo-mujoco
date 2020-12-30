from gym.envs.registration import register

register(
    id = 'My-HalfCheetah-v2',
    entry_point = 'envs.half_cheetah:HalfCheetahEnv',
    max_episode_steps=500,
)
