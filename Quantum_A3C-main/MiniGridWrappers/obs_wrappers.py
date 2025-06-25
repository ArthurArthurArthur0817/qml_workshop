# 2020 10 21
# MiniGridWrappers for the Obs

import math
import operator
from functools import reduce

import numpy as np

# import gym
import gymnasium as gym
# from gym_minigrid.wrappers import *

import torch


class ImgObsFlatWrapper(gym.core.ObservationWrapper):
	"""
	Use the image as the only observation output, no language/mission.
	The image tensor is flattened.
	"""

	def __init__(self, env):
		super().__init__(env)
		# self.observation_space = env.observation_space.spaces['image']
		imgSpace = env.observation_space.spaces['image']
		imgSize = reduce(operator.mul, imgSpace.shape, 1)
		self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(imgSize,),
            dtype=np.float32
        )

	def observation(self, obs):
		return obs['image'].flatten()



def main():
	env = gym.make('MiniGrid-Empty-5x5-v0')
	env = ImgObsFlatWrapper(env)
	init_obs = env.reset()
	print(init_obs)
	print(init_obs.shape)

	obs = torch.from_numpy(init_obs).unsqueeze(0).unsqueeze(0)

	print(obs)
	print(obs.shape)


	print("OBS SPACE: ", env.observation_space.shape[0])
	print("ACT SPACE: ", env.action_space.n)
	
	return



if __name__ == '__main__':
	main()