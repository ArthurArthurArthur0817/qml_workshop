import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import confusion_matrix
import mplfinance as mpf
import seaborn as sns
import matplotlib.pyplot as plt

def create_features_and_labels(file_path, num_rows=1000):
    df = pd.read_csv(file_path).tail(num_rows)
    df.reset_index(inplace=True, drop=True)

    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()

    df['label'] = np.nan
    golden_cross = (df['ma5'].shift(1) <= df['ma10'].shift(1)) & (df['ma5'] > df['ma10'])
    df.loc[golden_cross, 'label'] = 1  # 上漲為 1

    death_cross = (df['ma5'].shift(1) >= df['ma10'].shift(1)) & (df['ma5'] < df['ma10'])
    df.loc[death_cross, 'label'] = 0  # 下跌為 0
    del df['volume']

    df.dropna(subset=['ma10'], inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df

def create_sequences(features_df, labels_series, sequence_length=16):
    x_list, y_list = [], []
    features_np = features_df.values
    labels_np = labels_series.values

    # 從 sequence_length-1 的位置開始遍歷，確保每個點都能回溯到足夠長的序列
    for i in range(sequence_length - 1, len(features_np)):
        current_label = labels_np[i]
        
        # 只要有 label 0 or 1
        if not np.isnan(current_label):
            start_index = i - sequence_length + 1
            end_index = i + 1
            x_list.append(features_np[start_index:end_index])
            y_list.append(current_label)
        
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

def visualize_training_samples(X_train, y_train):
    print(X_train.shape, y_train.shape)

    for i in range(X_train.shape[0]):
        ohlc_data = X_train[i, :, :4]
        label = y_train[i]
        
        # 準備 K 線圖所需的核心 OHLC DataFrame
        ohlc_data = X_train[i, :, :4].numpy()
        dates = pd.date_range(end=pd.Timestamp.now(), periods=ohlc_data.shape[0], freq='D')
        plot_df = pd.DataFrame(
            ohlc_data,
            columns=['Open', 'High', 'Low', 'Close'],
            index=dates
        )
        
        # 準備要疊加的 MA 數據
        ma5_data = X_train[i, :, 4].numpy()
        ma10_data = X_train[i, :, 5].numpy()
        
        # 使用 mplfinance 的 make_addplot 功能來準備疊加圖層
        additional_plots = [
            mpf.make_addplot(ma5_data, color='blue', panel=0, width=0.7, secondary_y=False),
            mpf.make_addplot(ma10_data, color='orange', panel=0, width=0.7, secondary_y=False)
        ]
        
        label_text = "Rise, Label=1" if label == 1 else "Fall, Label=0"
        
        mpf.plot(
            plot_df,
            type='candle',
            style='yahoo',
            title=f"Sample #{i} - Label: {label_text}",
            ylabel='Price',
            addplot=additional_plots,
        )

    raise ValueError()

class StandardLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(StandardLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        last_time_step_out = out[:, -1, :]
        final_out = self.fc(last_time_step_out)
        return final_out

feature_columns = ['open', 'high', 'low', 'close', 'ma5', 'ma10']
sequence_length = 8
params = {
    'input_size': len(feature_columns),  # 6
    'hidden_size': 16,
    'output_size': 2,
    'num_layers': 1,
    'learning_rate': 0.001,
    'batch_size': 16,
    'epochs': 50,
}

# 步驟 A: 建立包含所有連續特徵和稀疏標籤的 DataFrame
full_data_df = create_features_and_labels(file_path='EURUSD_test.csv', num_rows=10000)

# 步驟 B: 分割 DataFrame 為訓練集和測試集
split_point = int(len(full_data_df) * 0.8)
train_df = full_data_df[:split_point]
test_df = full_data_df[split_point:]
test_df.reset_index(inplace=True, drop=True)

# 步驟 C: 在標準化後的連續資料上，建立有效的序列
X_train, y_train = create_sequences(train_df[feature_columns], train_df['label'], sequence_length)
X_test, y_test = create_sequences(test_df[feature_columns], test_df['label'], sequence_length)

# 步驟 D: 進行新的實例標準化 (Instance-wise Normalization)
X_train = normalize_sequences(X_train)
X_test = normalize_sequences(X_test)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=params['batch_size'], shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=params['batch_size'], shuffle=False)
print(f"資料準備完成。訓練集大小: {len(X_train)}, 測試集大小: {len(X_test)}")

# visualize_training_samples(X_train, y_train)

# 步驟 E: 定義 model & opt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"current device: {device}")
model = StandardLSTM(
    input_size=params['input_size'],
    hidden_size=params['hidden_size'],
    output_size=params['output_size'],
    num_layers=params['num_layers']
).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

# 步驟 F: 訓練模型
start_time = time.time()
for epoch in range(params['epochs']):
    # --- 訓練階段 ---
    model.train()
    train_loss = 0
    for i, (sequences, labels) in enumerate(train_loader):
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- 驗證階段 ---
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()

    # 計算平均損失
    avg_train_loss = train_loss / len(train_loader)
    avg_valid_loss = valid_loss / len(test_loader)

    print(f"Epoch [{epoch+1}/{params['epochs']}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_valid_loss:.6f}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} sec")

# 步驟 G: 評估模型 + 混淆矩陣 
model.eval()
all_labels, all_preds = [], []
with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

correct = (np.array(all_preds) == np.array(all_labels)).sum()
total = len(all_labels)
print(f'Testing Accuracy: {100 * correct / total:.2f} %')

cm = confusion_matrix(all_labels, all_preds)
class_names = ['Fall (0)', 'Rise (1)']
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
# plt.savefig("confusion_matrix_final.png")
plt.show()