# torch == 1.12.1
# pennylane == 0.35.1
# numpy == 1.26.4

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import v_wrap, set_init, push_and_pull, record
from plot_functions import full_plotting


from shared_adam import SharedAdam

import gymnasium as gym
# import gym_cartpolemod
# import gym_twoqubit
import os
from os.path import abspath, dirname, join
import pickle

import pennylane as qml

from custom_torch_vqc import TorchVQC


import minigrid
from MiniGridWrappers import ImgObsFlatWrapper

os.environ["OMP_NUM_THREADS"] = "1"

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 10000


ENV_NAME = 'MiniGrid-Empty-5x5-v0'
env = gym.make(ENV_NAME)
env = ImgObsFlatWrapper(env)

# Define VQC

latent_dim = 8
input_dim = latent_dim
output_dim = latent_dim
num_qubit = latent_dim
num_vqc_layers = 2



N_S = env.observation_space.shape[0]
N_A = env.action_space.n


n_qubits = latent_dim
q_depth = num_vqc_layers
n_class = latent_dim

###

class Net(nn.Module):
	def __init__(self, s_dim, a_dim):
		super(Net, self).__init__()
		self.s_dim = s_dim
		self.a_dim = a_dim
		self.pi1 = nn.Linear(s_dim, latent_dim)
		
		# self.pi_vqc = qml.qnn.TorchLayer(q_node_func, weight_shapes_1)
		self.pi_vqc = TorchVQC(vqc_depth = q_depth, n_qubits = n_qubits, n_class = n_class)

		self.pi2 = nn.Linear(latent_dim, a_dim)
		self.v1 = nn.Linear(s_dim, latent_dim)

		# self.v_vqc = qml.qnn.TorchLayer(q_node_func, weight_shapes_1)
		self.v_vqc = TorchVQC(vqc_depth = q_depth, n_qubits = n_qubits, n_class = n_class)
		self.v2 = nn.Linear(latent_dim, 1)
		set_init([self.pi1, self.pi2, self.v1, self.v2])
		self.distribution = torch.distributions.Categorical

	def forward(self, x):
		pi1 = torch.tanh(self.pi1(x))
		pi1 = torch.tanh(self.pi_vqc(pi1))
		logits = self.pi2(pi1)

		v1 = torch.tanh(self.v1(x))
		v1 = torch.tanh(self.v_vqc(v1))
		values = self.v2(v1)
		return logits, values

	def choose_action(self, s):
		self.eval()
		logits, _ = self.forward(s)
		prob = F.softmax(logits, dim=1).data
		m = self.distribution(prob)
		return m.sample().numpy()[0]

	def loss_func(self, s, a, v_t):
		self.train()
		logits, values = self.forward(s)
		td = v_t - values
		c_loss = td.pow(2)
		
		probs = F.softmax(logits, dim=1)
		m = self.distribution(probs)
		exp_v = m.log_prob(a) * td.detach().squeeze()
		a_loss = -exp_v
		total_loss = (c_loss + a_loss).mean()
		return total_loss


class Worker(mp.Process):
	def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
		super(Worker, self).__init__()
		self.name = 'w%02i' % name
		self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
		self.gnet, self.opt = gnet, opt
		self.lnet = Net(N_S, N_A)           # local network
		# self.env = gym.make('CartPole-v1')
		# self.env = gym.make('CartPole-v0').unwrapped
		# self.env = gym.make('BasicTwoQubit-v0', target = target)
		# self.env = gym.make('MiniGrid-SimpleCrossingS9N1-v0')
		# self.env = ImgObsFlatWrapper(gym.make('MiniGrid-SimpleCrossingS9N1-v0'))
		self.env = ImgObsFlatWrapper(gym.make(ENV_NAME))

	def run(self):
		total_step = 1
		while self.g_ep.value < MAX_EP:
			# s = self.env.reset()
			s, _ = self.env.reset() # New Gymnasium
			buffer_s, buffer_a, buffer_r = [], [], []
			ep_r = 0.
			epi_step = 1
			while True:


				a = self.lnet.choose_action(v_wrap(s[None, :])) 


				s_, r, terminated, truncated, info = self.env.step(a)
				done = terminated or truncated  # Combine both flags into a single `done` flag

				epi_step += 1

				# 2022 09 11: not good for QAS (originally for CartPole)
				# if done: r = -1

				ep_r += r
				buffer_a.append(a)
				buffer_s.append(s)
				buffer_r.append(r)

				if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
					# sync
					push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
					buffer_s, buffer_a, buffer_r = [], [], []

					if done or epi_step >= 500:  # done and print information
						record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
						break
				s = s_
				total_step += 1
		# return 
		self.res_queue.put(None)


if __name__ == "__main__":

	gnet = Net(N_S, N_A)        # global network
	gnet.share_memory()         # share the global parameters in multiprocessing
	opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
	global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

	# parallel training
	workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(min(mp.cpu_count(), 80))]
	[w.start() for w in workers]
	
	# [w.join() for w in workers]

	res = []                    # record episode reward to plot

	# print("FINISHED join()")

	# while not res_queue.empty():
	# 	print("in the while loop")
	# 	r = res_queue.get()
	# 	print(r)

	# 	res.append(r)



	while True:
		r = res_queue.get()
		# if r is not None:
		# 	res.append(r)
		# else:
		# 	break

		if len(res) < MAX_EP and (r is not None):
			res.append(r)

		elif res_queue.empty():
			break
			
		else:
			pass

	[w.join() for w in workers]

	print(len(res))

	

	# Plot and save
	full_plotting(_fileTitle = "VQC_A3C_{}".format(ENV_NAME), _trainingLength = len(res), _currentRewardList = res)
	path_model = join(dirname(abspath(__file__)), "trained_model_VQC_A3C_{}.pth".format(ENV_NAME))
	torch.save(gnet.state_dict(), path_model)

	# Save the raw data of res
	path_episode_reward = join(dirname(abspath(__file__)), "episode_returns_VQC_A3C_{}".format(ENV_NAME))
	with open(path_episode_reward, "wb") as fp:
		pickle.dump(res, fp)





