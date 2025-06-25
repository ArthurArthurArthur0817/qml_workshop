from sklearn.preprocessing import StandardScaler
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

def get_ohlcv_data(num_datapoints=1000, sequence_length=20, price_change_threshold=0.005):
    """
    產生模擬的 OHLCV 金融時間序列數據，並生成分類標籤。
    
    Args:
        num_datapoints (int): 總共要生成的資料點數量。
        sequence_length (int): 用於預測的序列長度。
        price_change_threshold (float): 定義漲跌盤整的百分比閾值。

    Returns:
        torch.Tensor: 特徵資料 (x)。
        torch.Tensor: 分類目標 (y)，0:下跌, 1:盤整, 2:上漲。
    """
    # ... (前段生成 ohlcv_data 的部分不變) ...
    # 1. 產生收盤價 (Close)
    mu = 0.0001
    sigma = 0.01
    prices = [100]
    for _ in range(num_datapoints):
        dt = 1
        Z = np.random.normal(0, 1)
        new_price = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        prices.append(new_price)
    
    close_prices = np.array(prices[1:])

    # 2. 基於收盤價產生 Open, High, Low
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0] * (1 - (np.random.rand() - 0.5) * 0.01)
    high_prices = np.maximum(open_prices, close_prices) * (1 + np.random.rand(num_datapoints) * 0.01)
    low_prices = np.minimum(open_prices, close_prices) * (1 - np.random.rand(num_datapoints) * 0.01)

    # 3. 產生交易量 (Volume)
    volume = np.random.poisson(1000, num_datapoints) + 100

    # 4. 組合成 OHLCV 數據
    ohlcv_data = np.stack([open_prices, high_prices, low_prices, close_prices, volume], axis=1)

    # 5. 數據標準化
    scaler = StandardScaler()
    ohlcv_data_scaled = scaler.fit_transform(ohlcv_data)

    # 6. 建立序列資料 (x) 和分類目標 (y)
    x_list, y_labels = [], []
    for i in range(len(ohlcv_data) - sequence_length -1):
        x_list.append(ohlcv_data_scaled[i : i + sequence_length])
        
        # 計算未來價格變動百分比
        current_price = ohlcv_data[i + sequence_length -1, 3] # 當前序列的最後一個收盤價
        future_price = ohlcv_data[i + sequence_length, 3]     # 下一個真實收盤價
        price_change = (future_price - current_price) / current_price
        
        # 根據閾值生成標籤
        if price_change > price_change_threshold:
            y_labels.append(2)  # 上漲
        elif price_change < -price_change_threshold:
            y_labels.append(0)  # 下跌
        else:
            y_labels.append(1)  # 盤整

    # 注意：目標 y 的 dtype 必須是 torch.long
    return torch.tensor(np.array(x_list)), torch.tensor(np.array(y_labels), dtype=torch.long)

def process_data_with_vt_labels(file_path, 
                                sequence_length=20, 
                                num_rows=1000,
                                horizon=10, 
                                vol_window=20, 
                                k=1.0):
	"""
	Args:
		file_path (str): CSV 檔案路徑
		sequence_length (int): 用於預測的序列長度
		num_rows (int, optional): 讀取的資料筆數，None 為全部讀取
		horizon (int): 計算未來報酬率的時間窗口（向前看 N 根 K 棒）
		vol_window (int): 計算滾動波動度的時間窗口
		k (float): 波動度閾值的乘數

	Returns:
		torch.Tensor: 特徵資料 (x)
		torch.Tensor: 分類目標 (y). 0:下跌, 1:盤整, 2:上漲
	"""
	df = pd.read_csv(file_path, nrows=num_rows)
	close_prices = df['close']

	# --- 標註策略 (Volatility-Scaled Labeling) ---
	# Step 1. 計算未來 N 個時間點的 log 報酬率 (shift負數為未來)
	fwd_ret = np.log(close_prices.shift(-horizon) / close_prices)
    
    # Step 2. 計算滾動波動度 (rolling volatility)
    # 使用 pct_change() 計算每一步的價格變動百分比，再計算其滾動標準差
	vol = close_prices.pct_change().rolling(vol_window).std()

    # 計算動態閾值，即波動度的 k 倍
	dynamic_thresh = k * vol
    
    # Step 3. Labelling based on dynamic threshold
    # 下跌 (Down) = 0, 盤整 (Sideways) = 1, 上漲 (Up) = 2
	labels = pd.Series(np.nan, index=close_prices.index)  # container
	labels.loc[fwd_ret >  dynamic_thresh] = 2  # 上漲
	labels.loc[fwd_ret < -dynamic_thresh] = 0  # 下跌
	labels.fillna(1, inplace=True)  # 剩下的 NaN 皆為盤整
	# ---------------------------------------------

    # Normalization
	ohlcv_data = df[['open', 'high', 'low', 'close', 'volume']].values
	scaler = StandardScaler()
	ohlcv_data_scaled = scaler.fit_transform(ohlcv_data)

    # 3. 建立序列資料 (x) 和對齊標籤 (y)
	x_list, y_labels = [], []
    
    # ** 注意迴圈範圍 **：
    # 必須確保迴圈內的所有索引都是有效的。
    # - 起始點：必須大於等於 vol_window，以避開波動度計算產生的 NaN。
    # - 結束點：必須讓 sequence 和 label 的未來 horizon 都有值。
    #   最後一個 sequence 的起始點 i，必須滿足 i + sequence_length + horizon <= len(df)
	start_index = vol_window
	end_index = len(df) - sequence_length - horizon
	for i in range(start_index, end_index):
		x_list.append(ohlcv_data_scaled[i: i + sequence_length])
		label_for_sequence = labels.iloc[i + sequence_length - 1]
		y_labels.append(label_for_sequence)

    # 轉為 PyTorch Tensors
	x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
	y_tensor = torch.tensor(np.array(y_labels), dtype=torch.long)
    
	return x_tensor, y_tensor

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
    
    # 儲存模型參數
    torch.save(model.state_dict(), f"{exp_name}/{file_name}_torch_model.pth")
    
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
    class_names = ['Down', 'Sideways', 'Up']
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

	dtype = torch.FloatTensor

	# Step 1. load data
	# x, y = get_ohlcv_data(
	# 	num_datapoints=1500,
	# 	sequence_length=params['seq_length'],
	# 	price_change_threshold=params['price_change_threshold']
	# )

	x, y = process_data_with_vt_labels(
		file_path='EURUSD_test.csv', 
		num_rows=1000,
		sequence_length=params['seq_length'],
		horizon=10,
		vol_window=20,
		k=1.0
	)

	params['input_size'] = x.shape[2]

	# Step 2. split train/test set
	num_for_train_set = int(0.8 * len(x))
	num_for_train_set = int(0.8 * len(x))
	x_train = x[:num_for_train_set].type(dtype)
	y_train = y[:num_for_train_set]

	x_test = x[num_for_train_set:].type(dtype)
	y_test = y[num_for_train_set:]
	print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
	print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

	# Step 3. model
	qlstm_cell = CustomQLSTMCell(params['input_size'], params['hidden_size'], params['output_size'], params['qnn_depth']).float()
	model = CustomLSTM(params['input_size'], params['hidden_size'], qlstm_cell).float()
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
		train_loss, train_acc = train_epoch_full(opt=opt, model=model, X=x_train, Y=y_train, batch_size=params['batch_size'])
		train_loss_list.append(train_loss)
		train_acc_list.append(train_acc)
		iteration_list.append(i + 1)

		# validation
		model.eval()
		with torch.no_grad():
			test_res, _ = model(x_test)
			test_preds = test_res[:, -1, :]
			test_loss = nn.CrossEntropyLoss()(test_preds, y_test).item()
			test_acc = calculate_accuracy(test_preds, y_test)
			test_loss_list.append(test_loss); test_acc_list.append(test_acc)

		print(f"Epoch [{i+1:03d}/{params['num_epochs']}] | "
			  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
			  f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2%} | "
			  f"Time: {time.time() - epoch_start_time:.2f}s")
        
		if (i + 1) % 5 == 0 or (i + 1) == params['num_epochs']:
			y_pred_on_test = torch.argmax(test_preds, dim=1).cpu().numpy()
			y_true_on_test = y_test.cpu().numpy()
			saving(params, iteration_list, train_loss_list, test_loss_list, train_acc_list, test_acc_list, model, y_true_on_test, y_pred_on_test)
			print(f"--- 結果已於 Epoch {i+1} 儲存 ---")

	print(f"\n訓練完成！總執行時間: {time.time() - total_start_time:.2f} 秒")


if __name__ == '__main__':
	PARAMS = {
		'seed': 0,
		'seq_length': 4,
		'input_size': 5,   # OHLCV
		'hidden_size': 5,
		'output_size': 3,  # (下跌, 盤整, 上漲)
		'qnn_depth': 5,
		'batch_size': 16,
		'learning_rate': 0.005,
		'num_epochs': 10,
		'exp_name': "QLSTM_OHLCV_Classification",
		'exp_index': 1,
		'price_change_threshold': 0.005,  # 定義漲跌的閾值 (0.5%)
	}

	main(PARAMS)
