#这个环境实现一个离散动作的swimmer
import numpy as np
from xuance.environment import RawEnvironment
from gym.spaces import Box,Discrete,MultiDiscrete
import gym
import time

class DiscreteSwimmerRLModel(RawEnvironment):
    def __init__(self,env_config):
        super(DiscreteSwimmerRLModel,self).__init__()
        self.model = DiscreteSwimmer()
        self.env_id = env_config.env_id
        self.observation_space = self.model.env.observation_space
        self.actNumbers = 17
        self.action_space = MultiDiscrete([self.actNumbers,self.actNumbers,self.actNumbers])
    def reset(self):
        return self.model.reset()
    def step(self, action):
        #action的范围是[-1,1]^2
        #因此这里返回
        real_action = np.array(
            [2*action[0]/(self.actNumbers-1)-1,
            2*action[1]/(self.actNumbers-1)-1,
            2*action[2]/(self.actNumbers-1)-1]
            )
        observation, reward, terminated, truncated, info = self.model.step(real_action)
        return observation, reward, terminated, truncated, info
    def render(self, *args, **kwargs):  # Render your environment and return an image if the render_mode is "rgb_array".
        return np.ones([64, 64, 64])
    def close(self):  # Close your environment.
        return  

class DiscreteSwimmer():
    def __init__(self):
        self.env = gym.make("Hopper-v4")
    def step(self,action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.observation = observation
        return observation, reward, terminated, truncated, info
    def reset(self):
        obs,info = self.env.reset()
        return obs, info
D = DiscreteSwimmer()
A = D.env.action_space.sample()
action_space = MultiDiscrete([5,5])
AA = action_space.sample()
























