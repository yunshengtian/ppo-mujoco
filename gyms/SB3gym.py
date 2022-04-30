import gym
import numpy as np
from gym import spaces


class SB3gym(gym.Env):
    def __init__(self, cfg: dict):
        super(SB3gym, self).__init__()
        self.width = 100 # TODO!!! cfg["screen_width"]
        self.height = 100 # TODO!!! cfg["screen_height"]
        self.env = gym.make('HalfCheetah-v2') # gym.make(cfg["task"])
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.height, self.width), dtype=np.uint8
        )

    def step(self, action):
        obs, reward, done, info = self.env.step(action=action)
        rgb_obs = self.env.render(
            mode="rgb_array", width=self.width, height=self.height
        )
        rgb_obs = np.transpose(rgb_obs, (2, 0, 1))
        return rgb_obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        rgb_obs = self.env.render(
            mode="rgb_array", width=self.width, height=self.height
        )
        return np.transpose(rgb_obs, (2, 0, 1))

    def render(self, mode="human"):
        return self.env.render(mode=mode)

    def seed(self, seed: int):
        self.env.seed(seed)
        self.seed = seed

    def close(self):
        return self.env.close()
