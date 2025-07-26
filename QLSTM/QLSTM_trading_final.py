from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import DataFrame
import pennylane as qml
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import time
import os

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ================== Model Definition =====================

def H_layer(nqubits):
	"""Layer of single-qubit Hadamard gates.
	"""
	for idx in range(nqubits):
		qml.Hadamard(wires=idx)

def RY_layer(w):
	"""Layer of parametrized qubit rotations around the y_tilde axis."""
	for idx, element in enumerate(w):
		qml.RY(element, wires=idx)

def entangling_layer(nqubits):
	""" Layer of CNOTs followed by another shifted layer of CNOT."""
	# In other words it should apply something like :
	# CNOT  CNOT  CNOT  CNOT...  CNOT
	#   CNOT  CNOT  CNOT...  CNOT
	for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
		qml.CNOT(wires=[i, i + 1])
	for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
		qml.CNOT(wires=[i, i + 1])

def q_function(x, q_weights, n_class):
	""" The variational quantum circuit. """
	n_dep = q_weights.shape[0]
	n_qub = q_weights.shape[1]

	H_layer(n_qub)

	# ------ origin -------
	# RY_layer(x)  # Embed features in the quantum node
	# ------ modified -------
	qml.AngleEmbedding(x, wires=range(x.shape[1]), rotation='Y')

	# Sequence of trainable variational layers
	for k in range(n_dep):
		entangling_layer(n_qub)
		RY_layer(q_weights[k])

	# Expectation values in the Z basis
	exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]  # only measure first "n_class" of qubits and discard the rest
	return exp_vals

class VQC(nn.Module):
	def __init__(self, vqc_depth, n_qubits, n_class):
		super().__init__()
		self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))  # g rotation params
		# self.dev = qml.device("default.qubit", wires=n_qubits)  # Can use different simulation backend or quantum computers.
		self.dev = qml.device("lightning.qubit", wires=n_qubits)  # ! 
		# self.dev = qml.device("lightning.gpu", wires=n_qubits)  # ! 
		self.VQC = qml.QNode(q_function, self.dev, interface="torch")
		self.n_class = n_class

	def forward(self, X):
		# y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)) for x in X]) # PennyLane 0.35.1
		# return y_preds

		y_preds = self.VQC(X, self.weights, self.n_class)
		return torch.stack(y_preds, dim=1)

class CustomQLSTMCell(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, vqc_depth):
		super(CustomQLSTMCell, self).__init__()
		self.hidden_size = hidden_size

		# Linear layers for gates and cell update
		# Change here to use PEennyLane Quantum VQCs.
		self.input_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.forget_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.cell_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)
		self.output_gate = VQC(vqc_depth = vqc_depth, n_qubits = input_size + hidden_size, n_class = hidden_size)

		self.output_post_processing = nn.Linear(hidden_size, output_size)

	def forward(self, x, hidden):
		h_prev, c_prev = hidden

		# Concatenate input and hidden state
		combined = torch.cat((x, h_prev), dim=1)

		# Compute gates
		i_t = torch.sigmoid(self.input_gate(combined))  # Input gate
		f_t = torch.sigmoid(self.forget_gate(combined))  # Forget gate
		g_t = torch.tanh(self.cell_gate(combined))      # Cell gate
		o_t = torch.sigmoid(self.output_gate(combined)) # Output gate

		# Update cell state
		c_t = f_t * c_prev + i_t * g_t

		# Update hidden state
		h_t = o_t * torch.tanh(c_t)

		# Actual outputs
		out = self.output_post_processing(h_t)

		return out, h_t, c_t

class CustomLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, lstm_cell_QT):
		super(CustomLSTM, self).__init__()
		self.hidden_size = hidden_size

		# Single LSTM cell
		self.cell = lstm_cell_QT

	def forward(self, x, hidden=None):
		batch_size, seq_len, _ = x.size()

		# Initialize hidden and cell states if not provided
		if hidden is None:
			h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
			c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
		else:
			h_t, c_t = hidden

		outputs = []

		# Process sequence one time step at a time
		for t in range(seq_len):
			x_t = x[:, t, :]  # Extract the t-th time step
			# print("x_t.shape: {}".format(x_t.shape))
			out, h_t, c_t = self.cell(x_t, (h_t, c_t))  # Update hidden and cell states
			# print("out: {}".format(out))
			outputs.append(out.unsqueeze(1))  # Collect output for this time step

		outputs = torch.cat(outputs, dim=1)  # Concatenate outputs across all time steps
		# print("outputs: {}".format(outputs))
		return outputs, (h_t, c_t)
	
# =========================================================

def calculate_accuracy(y_pred, y_true):
    # y_pred 的 shape 是 (batch_size, num_classes)
    # 我們需要找到分數最高的那個類別作為預測結果
    predicted_class = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
    correct = (predicted_class == y_true).sum().item()
    accuracy = correct / len(y_true)
    return accuracy

def create_features_and_labels(file_path, num_rows=1000, horizon=5, threshold_pct=3.0):
    df = pd.read_csv(file_path).tail(num_rows)
    df = df[::-1].reset_index(drop=True)
    df.reset_index(inplace=True, drop=True)

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()

    # Calculate future price change percentage
    future_prices = df['close'].shift(-horizon)
    price_change_pct = (future_prices - df['close']) / df['close'] * 100

    # Label based on threshold
    df['label'] = 0  # Default to sideways (0)
    df.loc[price_change_pct > threshold_pct, 'label'] = 1    # Up > 3%
    df.loc[price_change_pct < -threshold_pct, 'label'] = 2   # Down > 3%

    del df['volume']

    # Drop rows where we can't calculate MA or future price
    df.dropna(subset=['ma10'], inplace=True)
    df = df[:-horizon]  # Remove last 'horizon' rows without future prices
    df.reset_index(inplace=True, drop=True)

    return df

def create_sequences(features_df, labels_series, sequence_length=16):
    x_list, y_list = [], []
    features_np = features_df.values
    labels_np = labels_series.values

    # 從 sequence_length-1 的位置開始遍歷，確保每個點都能回溯到足夠長的序列
    for i in range(sequence_length - 1, len(features_np)):
        start_index = i - sequence_length + 1
        end_index = i + 1
        x_list.append(features_np[start_index:end_index])
        y_list.append(labels_np[i])
        
    return torch.tensor(np.array(x_list), dtype=torch.float32), torch.tensor(np.array(y_list), dtype=torch.long)

def normalize_sequences(x_tensor):
    normalized_tensor = x_tensor.clone()
    for i in range(x_tensor.shape[0]):
        sequence = x_tensor[i] # Shape: (seq_len, features)
        
        min_val = sequence.min()
        max_val = sequence.max()
        
        # 為了避免分母為零 (當序列中所有值都相同時)
        denominator = max_val - min_val
        if denominator > 0:
            normalized_tensor[i] = (sequence - min_val) / denominator
        else:
            # 如果所有值都相同，標準化後就都是 0
            normalized_tensor[i] = torch.zeros_like(sequence)
            
    return normalized_tensor

def train_epoch_full(opt, model, X, Y, batch_size):
	losses, accuracies = [], []
	model.train()

	for beg_i in range(0, X.shape[0], batch_size):
		X_train_batch = X[beg_i:beg_i + batch_size]
		Y_train_batch = Y[beg_i:beg_i + batch_size]

		opt.zero_grad()
		model_res, _ = model(X_train_batch)
		predictions = model_res[:, -1, :]
		loss = nn.CrossEntropyLoss()(predictions, Y_train_batch)

		loss.backward()
		opt.step()

		losses.append(loss.item())
		acc = calculate_accuracy(predictions.detach(), Y_train_batch.detach())
		accuracies.append(acc)
	return np.mean(losses), np.mean(accuracies)

def saving(params, iteration_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list, model, y_true_test, y_pred_test):
    exp_name = params['exp_name']
    file_name = f"{exp_name}_NO_{params['exp_index']}_Epoch_{iteration_list[-1]}"
    os.makedirs(exp_name, exist_ok=True)
    
    # 儲存 loss, accuracy 等指標
    with open(f"{exp_name}/{file_name}_metrics.pkl", "wb") as fp:
        pickle.dump({
            "train_loss": train_loss_list, "test_loss": test_loss_list,
            "train_acc": train_acc_list, "test_acc": test_acc_list
        }, fp)
    
    # 儲存模型參數到 models 目錄
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_filename = f"qlstm_model_epochs_{iteration_list[-1]}.pth"
    torch.save(model.state_dict(), os.path.join(models_dir, model_filename))
    
    # 呼叫新的繪圖函數 (您需要先定義好 plotting_curves)
    plotting_curves(exp_name, file_name, iteration_list, train_loss_list, test_loss_list, 'Loss')
    plotting_curves(exp_name, file_name, iteration_list, train_acc_list, test_acc_list, 'Accuracy')
    
    # 呼叫混淆矩陣繪圖函數
    plotting_confusion_matrix(exp_name, file_name, y_true_test, y_pred_test)

def plotting_curves(exp_name, file_name, iteration_list, train_vals, test_vals, curve_type='Loss'):
    fig, ax = plt.subplots()
    ax.plot(iteration_list, train_vals, '-b', label=f'Training {curve_type}')
    ax.plot(iteration_list, test_vals, '-r', label=f'Testing {curve_type}')
    ax.legend()
    ax.set(xlabel='Epoch', ylabel=curve_type, title=f"{exp_name} - {curve_type} Curve")
    ax.grid(True)
    fig.savefig(f"{exp_name}/{file_name}_{curve_type.lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf", format='pdf')
    plt.close(fig)

def plotting_confusion_matrix(exp_name, file_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    class_names = ['Up (>3%)', 'Down (>3%)']
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'{exp_name} - Confusion Matrix')
    plt.tight_layout()
    fig.savefig(f"{exp_name}/{file_name}_confusion_matrix_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf", format='pdf')
    plt.close(fig)

def main(params):
	total_start_time = time.time()
	torch.manual_seed(params['seed'])
	np.random.seed(params['seed'])

	# Step 1: Create features and labels with future price changes
	full_data_df = create_features_and_labels(
		file_path='USD_TWD_Historical Data.csv', 
		num_rows=10000,
		horizon=5,
		threshold_pct=1.2
	)

	# Step 2: Split DataFrame into train and test sets
	split_point = int(len(full_data_df) * 0.8)
	train_df = full_data_df[:split_point]
	test_df = full_data_df[split_point:]
	test_df.reset_index(inplace=True, drop=True)

	# Step 3: Convert to LSTM format
	feature_columns = params['feature_columns']
	X_train, y_train = create_sequences(train_df[feature_columns], train_df['label'], params['sequence_length'])
	X_test, y_test = create_sequences(test_df[feature_columns], test_df['label'], params['sequence_length'])

	# Step 4: Apply Instance-wise Normalization
	X_train = normalize_sequences(X_train)
	X_test = normalize_sequences(X_test)

	print(f"\nOriginal data size: Training={len(X_train)}, Test={len(X_test)}")
	print(f"Label distribution before filtering:")
	print(f"  Train - Sideways: {(y_train == 0).sum().item()}, Up: {(y_train == 1).sum().item()}, Down: {(y_train == 2).sum().item()}")
	print(f"  Test - Sideways: {(y_test == 0).sum().item()}, Up: {(y_test == 1).sum().item()}, Down: {(y_test == 2).sum().item()}")

	# Filter out sideways (label=0), keep only up (1) and down (2)
	train_mask = (y_train == 1) | (y_train == 2)
	test_mask = (y_test == 1) | (y_test == 2)
	
	X_train = X_train[train_mask]
	y_train = y_train[train_mask]
	X_test = X_test[test_mask]
	y_test = y_test[test_mask]

	# Check if we have any data left
	if len(X_train) == 0 or len(X_test) == 0:
		print("\nERROR: No samples remaining after filtering!")
		print("All samples are in the sideways category (price change within ±3%).")
		print("Suggestions:")
		print("  1. Lower the threshold_pct value (current: 3.0)")
		print("  2. Try threshold_pct=1.0 or 0.5 for more sensitive detection")
		print("  3. Check if the data has enough price volatility")
		return

	# Remap labels: 1->0 (up), 2->1 (down)
	y_train = y_train - 1
	y_test = y_test - 1

	print(f"\nAfter filtering: Training={len(X_train)}, Test={len(X_test)}")
	print(f"Label distribution after remapping:")
	print(f"  Train - Up: {(y_train == 0).sum().item()}, Down: {(y_train == 1).sum().item()}")
	print(f"  Test - Up: {(y_test == 0).sum().item()}, Down: {(y_test == 1).sum().item()}")

	# Update input_size based on actual features
	params['input_size'] = len(feature_columns)

	# Convert to appropriate dtype
	dtype = torch.FloatTensor
	x_train = X_train.type(dtype)
	x_test = X_test.type(dtype)

	# Step 3. model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Current device: {device}")
	qlstm_cell = CustomQLSTMCell(params['input_size'], params['hidden_size'], params['output_size'], params['qnn_depth']).float().to(device)
	model = CustomLSTM(params['input_size'], params['hidden_size'], qlstm_cell).float().to(device)
	# opt = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
	opt = torch.optim.RMSprop(model.parameters(), lr=params['learning_rate'])
	print(model)

	# Step 4. main loop
	train_loss_list, test_loss_list = [], []
	train_acc_list, test_acc_list = [], []
	iteration_list = []
	print("\n--- start training ---")
	for i in range(params['num_epochs']):
		epoch_start_time = time.time()

		# training
		train_loss, train_acc = train_epoch_full(opt=opt, model=model, X=x_train.to(device), Y=y_train.to(device), batch_size=params['batch_size'])
		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		iteration_list.append(i + 1)

		# validation
		model.eval()
		with torch.no_grad():
			test_res, _ = model(x_test.to(device))
			test_preds = test_res[:, -1, :]
			test_loss = nn.CrossEntropyLoss()(test_preds, y_test.to(device)).item()
			test_acc = calculate_accuracy(test_preds, y_test.to(device))
			test_loss_list.append(test_loss); test_acc_list.append(test_acc)

		print(f"Epoch [{i+1:03d}/{params['num_epochs']}] | "
			  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
			  f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2%} | "
			  f"Time: {time.time() - epoch_start_time:.2f}s")
        
		if (i + 1) % 5 == 0 or (i + 1) == params['num_epochs']:
			y_pred_on_test = torch.argmax(test_preds.cpu(), dim=1).numpy()
			y_true_on_test = y_test.cpu().numpy()
			saving(params, iteration_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list, model, y_true_on_test, y_pred_on_test)
			print(f"--- Results saved at Epoch {i+1} ---")

	print(f"\nTraining complete! Total runtime: {time.time() - total_start_time:.2f} seconds")
	print(f"Final Testing Accuracy: {test_acc:.2%}")


if __name__ == '__main__':
	PARAMS = {
		'seed': 0,
		'feature_columns': ['open', 'high', 'low', 'close', 'ma5', 'ma10'],
		'sequence_length': 4,
		'input_size': 6,   # will be updated in main() based on feature_columns
		'hidden_size': 2,  # 4
		'output_size': 2,  # (上漲>1%, 下跌>1%)
		'qnn_depth': 1,    # 3
		'batch_size': 4,
		'learning_rate': 5e-3,
		'num_epochs': 50,
		'exp_name': "QLSTM_Trading_Classification",
		'exp_index': 1,
	}

	main(PARAMS)
