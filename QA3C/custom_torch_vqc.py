# 2024 10 09
# Custom Torch VQC for use
# PennyLane 0.35.1

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


##

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

##

# Define actual circuit architecture
def q_function(x, q_weights, n_class):
	""" The variational quantum circuit. """

	# Reshape weights
	# θ = θ.reshape(vqc_depth, n_qubits)

	# Start from state |+> , unbiased w.r.t. |0> and |1>

	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	# Embed features in the quantum node
	RY_layer(x)

	# Sequence of trainable variational layers
	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])

	# Expectation values in the Z basis
	exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]  # only measure first "n_class" of qubits and discard the rest
	return exp_vals

##

# Wrapped previous model as a PyTorch Module
class TorchVQC(nn.Module):
	def __init__(self, vqc_depth, n_qubits, n_class):
		super().__init__()
		self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # g rotation params
		self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
		self.VQC = qml.QNode(q_function, self.dev, interface = "torch")

		self.n_class = n_class


	def forward(self, X):
		y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)).float() for x in X]) # PennyLane 0.35.1
		return y_preds





def main():

	BATCH_SIZE = 4
	INPUT_DIM = 8

	VQC_DEPTH = 2
	N_QUBITS = 8
	N_CLASS = 3

	trial_input = torch.randn(BATCH_SIZE, INPUT_DIM)

	model = TorchVQC(vqc_depth = VQC_DEPTH, n_qubits = N_QUBITS, n_class = N_CLASS)

	res = model.forward(trial_input)

	print("res: {}".format(res))

	res[0][0].backward()

	for name, param in model.named_parameters():
		if param.requires_grad:
			print(f"Parameter name: {name}")
			print(f"Parameter shape: {param.shape}")
			print(f"Parameter grad: {param.grad}")
			print(f"Parameter value: {param.data}\n")



	return

if __name__ == '__main__':
	main()






