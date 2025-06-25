import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from utils import v_wrap, set_init, push_and_pull, record
from plot_functions import full_plotting


from shared_adam import SharedAdam

import gym
# import gym_cartpolemod
# import gym_twoqubit
import os
from os.path import abspath, dirname, join
import pickle

import pennylane as qml

### Basic building blocks of a VQC

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RX_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RX(element, wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def RZ_layer(w):
	"""Layer of parametrized qubit rotations around the y axis.
	"""
	for idx, element in enumerate(w):
		qml.RZ(element, wires=idx)




def entangling_layer(nqubits):
	"""Layer of CNOTs followed by another shifted layer of CNOT.
	"""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])
	

def cycle_entangling_layer(nqubits):
	for i in range(0, nqubits):  
		qml.CNOT(wires=[i, (i + 1) % nqubits])


###

# dev = qml.device("default.qubit", wires=n_qubits)

# @qml.qnode(dev, interface="torch")
# def quantum_net(inputs, q_weights):
# 	n_dep = q_weights.shape[0]
# 	n_qub = q_weights.shape[1]

# 	H_layer(n_qub)

# 	RY_layer(inputs)

# 	for k in range(n_dep):
# 		entangling_layer(n_qub)
# 		RY_layer(q_weights[k])

# 	# Expectation values in the Z basis
# 	# exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
# 	# return tuple(exp_vals)
# 	return [qml.expval(qml.PauliZ(position)) for position in range(n_qub)]

###

class TorchQNNLayer(nn.Module):
	def __init__(self, q_depth, n_qubits):
		super().__init__()

		dev = qml.device("default.qubit", wires=n_qubits)

		@qml.qnode(dev, interface="torch")
		def quantum_net(inputs, q_weights):
			n_dep = q_weights.shape[0]
			n_qub = q_weights.shape[1]

			H_layer(n_qub)

			RY_layer(inputs)

			for k in range(n_dep):
				entangling_layer(n_qub)
				RY_layer(q_weights[k])

			# Expectation values in the Z basis
			# exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
			# return tuple(exp_vals)
			return [qml.expval(qml.PauliZ(position)) for position in range(n_qub)]


		weight_shapes = {"q_weights": (q_depth, n_qubits)}

		self.q_func_torch = qml.qnn.TorchLayer(quantum_net, weight_shapes)



	def forward(self, inputs):
		return self.q_func_torch(inputs)