import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
import pennylane as qml
import time
import threading

# --- 專案路徑設定與模組引用 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 引用 QLSTM 中的資料處理函式和模型類別
from QLSTM.QLSTM_trading_final import (
    create_sequences, normalize_sequences,
    H_layer, RY_layer, entangling_layer, q_function,
    VQC, CustomQLSTMCell, CustomLSTM
)
# 引用 A3C 的輔助工具
from QA3C.utils import v_wrap, set_init, push_and_pull, record
from QA3C.plot_functions import full_plotting
from QA3C.shared_adam import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"

def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    印出一個 PyTorch 模型的分層參數摘要，包含總參數數量。

    Args:
        model (nn.Module): 要分析的 PyTorch 模型。
        model_name (str): 顯示在摘要標題中的模型名稱。
    """
    print("=" * 70)
    print(f"{model_name} Parameters Summary")
    print("-" * 70)
    print(f"{'層名稱 (Layer Name)':<35} {'形狀 (Shape)':<20} {'參數數量':>12}")
    print("-" * 70)
    
    total_params = 0
    
    # 遍歷模型中所有命名的參數
    for name, param in model.named_parameters():
        # 只計算需要計算梯度的參數（可訓練的參數）
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            shape_str = str(list(param.shape))
            print(f"{name:<35} {shape_str:<20} {num_params:>12,}")
            
    print("-" * 70)
    print(f"總可訓練參數 (Total Trainable Parameters): {total_params:>15,}")
    print("=" * 70)

# ===== Quantum Layer Functions for VQC =====
def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)

def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT."""
    for i in range(0, nqubits - 1, 2):  # Loop over even indices: i=0,2,...N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # Loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

def q_function(x, q_weights, n_class):
    """The variational quantum circuit."""
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
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
    return exp_vals

class TorchVQC(nn.Module):
    """Quantum Variational Circuit wrapped as PyTorch Module"""
    def __init__(self, vqc_depth, n_qubits, n_class):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.VQC = qml.QNode(q_function, self.dev, interface="torch")
        self.n_class = n_class
    
    def forward(self, X):
        y_preds = torch.stack([torch.stack(self.VQC(x, self.weights, self.n_class)).float() for x in X])
        return y_preds
# ===== End of Quantum Layer Functions =====

# QLSTM 參數
qlstm_params = {
    'feature_columns': ['open', 'high', 'low', 'close', 'ma5', 'ma10'],
    'sequence_length': 4, # 8
    'input_size': 6,
    'hidden_size': 2, # 4
    'output_size': 2,
    'qnn_depth': 1, # 3
}

def load_qlstm_model(model_path, device):
    """Load QLSTM model within each process to avoid pickling issues"""
    qlstm_cell = CustomQLSTMCell(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        output_size=qlstm_params['output_size'],
        vqc_depth=qlstm_params['qnn_depth']
    ).float().to(device)
    
    lstm_model = CustomLSTM(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        lstm_cell_QT=qlstm_cell
    ).float().to(device)
    
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()
    return lstm_model

def prepare_trading_data(file_path, num_rows=10000):
    """
    專為交易環境設計的資料準備函式.
    包含 ma5, ma10 技術指標都先算好, 並移除 NaN 值, 以求簡化.
    """
    df = pd.read_csv(file_path)
    df = df[::-1].reset_index(drop=True)
    print(df.head())
    
    # 預先計算好所有特徵
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 檢查是否存在價格為 0 的異常資料
    zero_price_rows = df[df['close'] == 0]
    if not zero_price_rows.empty:
        print("錯誤：在資料中發現 'close' 價格為 0 的異常行：")
        print(zero_price_rows)
        raise ValueError("資料驗證失敗：'close' 價格不應為 0。請檢查原始資料檔案。")

    if df.empty:
        raise ValueError("錯誤：經過資料清理後，沒有剩餘的有效資料可供訓練。請檢查原始資料檔案。")

    return df

class TradingEnv:
    """
    經過驗證的最佳交易環境 (The Proven Best Environment) - State 已修正版

    這個版本修正了 state representation，移除了所有無效的佔位符，
    並使用 8 個完整、經過正規化的金融特徵來建構觀測值，
    為 Agent 提供豐富且穩定的決策依據。
    """
    def __init__(self, df, lstm_model, device, time_penalty=0.02):
        self.df = df.copy()
        self.lstm_model = lstm_model
        self.device = device
        self.sequence_length = qlstm_params['sequence_length']
        self.feature_columns = qlstm_params['feature_columns']
        self.time_penalty = time_penalty
        self.volatility_window = 20  # 新增：用於計算波動率的窗口大小

        # 預先計算所有需要的技術指標
        for window in [5, 20, 60]:
            if f'ma{window}' not in self.df.columns:
                self.df[f'ma{window}'] = self.df['close'].rolling(window=window).mean()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # [已修正] 觀測空間維度 = LSTM 輸出維度 + 8 個額外特徵
        self.observation_space_shape = (qlstm_params['output_size'] + 8,)
        self.action_space_n = 3

        self.initial_cash = 50000 # TODO
        self.trade_amount_per_time = 45000 # TODO
        self.reset()

    def _get_state(self):
        """
        [已修正] 產生一個由 8 個有意義的特徵組成的 state 向量。
        """
        # 1. LSTM 輸出 (市場趨勢預測)
        start = self.current_step - self.sequence_length + 1
        end = self.current_step + 1
        sequence_df = self.df.iloc[start:end]
        
        x = torch.tensor(sequence_df[self.feature_columns].values, dtype=torch.float32).unsqueeze(0)
        x_normalized = normalize_sequences(x).to(self.device)
        with torch.no_grad():
            outputs, _ = self.lstm_model(x_normalized)
            logits = outputs[:, -1, :]  # Get the last timestep output
            lstm_state = F.softmax(logits, dim=1).squeeze(0).cpu()
            # print('lstm_state', lstm_state)

        # 2. 計算 8 個額外的金融特徵
        current_price = self.df['close'].iloc[self.current_step]
        ma5 = self.df['ma5'].iloc[self.current_step]
        ma20 = self.df['ma20'].iloc[self.current_step]
        ma60 = self.df['ma60'].iloc[self.current_step]

        # 特徵 1: 現金比例
        cash_ratio = self.cash / self.initial_cash
        
        # 特徵 2: 持倉價值比例
        holdings_value = sum(self.active_trades_shares) * current_price
        holdings_ratio = holdings_value / self.initial_cash
        
        # 特徵 3 & 4: 未實現損益 和 成本現價比
        unrealized_pnl_pct = 0.0
        avg_price_ratio = 0.0
        if sum(self.active_trades_shares) > 0:
            avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
            if avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if current_price > 0:
                avg_price_ratio = (current_price - avg_buy_price) / current_price
        
        # 特徵 5 & 6: 價格與中/長期均線的乖離率
        price_ma20_ratio = (current_price - ma20) / ma20 if ma20 > 0 else 0.0
        price_ma60_ratio = (current_price - ma60) / ma60 if ma60 > 0 else 0.0

        # 特徵 7: 短期與中期均線的乖離率 (判斷趨勢動能)
        ma5_ma20_ratio = (ma5 - ma20) / ma20 if ma20 > 0 else 0.0
        
        # 特徵 8: 近期價格波動率
        if self.current_step >= self.volatility_window:
            recent_prices = self.df['close'].iloc[self.current_step - self.volatility_window : self.current_step]
            mean_price = np.clip(np.mean(recent_prices), 1e-9, np.inf)
            price_volatility = np.std(recent_prices) / mean_price
        else:
            price_volatility = 0.0

        # 組合所有特徵成一個 tensor
        additional_state = torch.tensor([
            cash_ratio,
            holdings_ratio,
            np.clip(unrealized_pnl_pct, -1.0, 1.0),
            np.clip(avg_price_ratio, -1.0, 1.0),
            np.clip(price_ma20_ratio, -0.2, 0.2),
            np.clip(price_ma60_ratio, -0.2, 0.2),
            np.clip(ma5_ma20_ratio, -0.1, 0.1),
            np.clip(price_volatility, 0.0, 1.0)
        ], dtype=torch.float32)

        return torch.cat((lstm_state, additional_state))
        
    def _reset_trade_state(self):
        self.active_trades_buy_prices = []
        self.active_trades_shares = []

    def reset(self):
        # 從第 60 步開始，確保所有 MA 指標都有值
        self.current_step = 60 
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.portfolio_history = [self.initial_cash] * self.current_step
        self.historical_trades = []
        self._reset_trade_state()
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df['close'].iloc[self.current_step]
        info = {'action_taken': 'hold'}
        # 預設獎勵為時間懲罰，鼓勵 Agent 採取行動
        reward = -self.time_penalty

        ma5 = self.df['ma5'].iloc[self.current_step]
        ma20 = self.df['ma20'].iloc[self.current_step]
        is_uptrend = ma5 > ma20

        if action == 1:  # Buy
            if self.cash >= self.trade_amount_per_time and is_uptrend:
                shares_bought = self.trade_amount_per_time / current_price
                self.cash -= self.trade_amount_per_time
                self.active_trades_buy_prices.append(current_price)
                self.active_trades_shares.append(shares_bought)
                info['action_taken'] = 'buy'
                reward += 0.5  # 順勢交易的即時獎勵
            elif self.cash >= self.trade_amount_per_time and not is_uptrend:
                reward -= 2.0  # 對逆勢買入的懲罰
            else:
                info['action_taken'] = 'invalid_buy'
                reward -= 0.5  # 對無效買入的懲罰

        elif action == 2:  # Sell
            total_shares = sum(self.active_trades_shares)
            if total_shares > 0:
                avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
                realized_profit = (current_price - avg_buy_price) * total_shares
                cost_basis = avg_buy_price * total_shares
                pnl_pct = realized_profit / cost_basis if cost_basis > 0 else 0

                if realized_profit > 0:
                    # if pnl_pct >= 0.015:
                    #     # 達到利潤目標，給予巨大獎勵
                    #     reward += 10.0 + pnl_pct * 50
                    # else:
                    #     # 未達目標但仍獲利，給予較小的獎勵，鼓勵再等等
                    #     reward += 10.0 + pnl_pct * 10
                    reward += 10.0 + pnl_pct * 50
                else:
                    cost_basis = avg_buy_price * total_shares
                    pnl_pct = realized_profit / cost_basis if cost_basis > 0 else 0
                    reward -= 2.0 + pnl_pct * 10 # 對虧損賣出的懲罰
                    
                self.cash += total_shares * current_price
                info['action_taken'] = 'sell'
                self._reset_trade_state()
            else:
                info['action_taken'] = 'invalid_sell'
                reward -= 0.5 # 懲罰無倉可賣的行為
        
        # 對未實現虧損的持續懲罰
        total_shares = sum(self.active_trades_shares)
        if total_shares > 0 and info['action_taken'] != 'sell':
            avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
            unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if unrealized_pnl_pct < 0:
                # 虧損越大，懲罰越重 (二次方懲罰)
                reward += unrealized_pnl_pct * abs(unrealized_pnl_pct) * 5.0
        
        current_holdings_value = sum(self.active_trades_shares) * current_price
        self.portfolio_value = self.cash + current_holdings_value
        self.portfolio_history.append(self.portfolio_value)
        final_reward = np.clip(reward, -15.0, 30.0)  # [-15, 15]
        next_state = self._get_state()
        
        if info['action_taken'] == 'sell':
            self.historical_trades.append({'profit': realized_profit})

        return next_state, final_reward, done, False, info

# --- Quantum A3C 模型與 Worker ---
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        # Quantum hyperparameters
        latent_dim = 8
        q_depth = 2
        n_qubits = 8
        
        self.s_dim = s_dim
        self.a_dim = a_dim
        
        # Actor network with quantum layer
        self.pi1 = nn.Linear(s_dim, latent_dim)
        self.pi_vqc = TorchVQC(vqc_depth=q_depth, n_qubits=n_qubits, n_class=latent_dim)
        self.pi2 = nn.Linear(latent_dim, a_dim)
        
        # Critic network with quantum layer
        self.v1 = nn.Linear(s_dim, latent_dim)
        self.v_vqc = TorchVQC(vqc_depth=q_depth, n_qubits=n_qubits, n_class=latent_dim)
        self.v2 = nn.Linear(latent_dim, 1)
        
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        # Actor with quantum layer
        pi1 = torch.tanh(self.pi1(x))
        pi1 = torch.tanh(self.pi_vqc(pi1))
        logits = self.pi2(pi1)
        
        # Critic with quantum layer
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

    def choose_best_action(self, s):
        """[新增] 在評估時，選擇機率最高的動作"""
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1)
        return torch.argmax(prob, dim=1).item()

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)

        entropy = m.entropy()

        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss - A3C_PARAMS['entropy_beta'] * entropy).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, df, model_path, device):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        
        # Store parameters for later use in run()
        self.df = df
        self.model_path = model_path
        self.device = device
        
        # Calculate observation space shape without creating TradingEnv
        # observation_space_shape = (qlstm_params['output_size'] + 8,)
        N_S = qlstm_params['output_size'] + 8
        N_A = 3  # action_space_n
        self.lnet = Net(N_S, N_A)

    def run(self):
        # Create QLSTM model and TradingEnv in the child process
        lstm_model = load_qlstm_model(self.model_path, self.device)
        self.env = TradingEnv(self.df, lstm_model, self.device)
        
        total_step = 1
        while self.g_ep.value < A3C_PARAMS['max_ep']:
            episode_start_time = time.time()  # 記錄 episode 開始時間
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            episode_steps = 0  # 記錄 episode 步數
            
            while True:
                episode_steps += 1
                a = self.lnet.choose_action(v_wrap(s.unsqueeze(0)))
                s_, r, done, _, _ = self.env.step(a)

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % A3C_PARAMS['update_global_iter'] == 0 or done:
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, A3C_PARAMS['gamma'])
                    buffer_s, buffer_a, buffer_r = [], [], []

                if done:
                    episode_time = time.time() - episode_start_time  # 計算 episode 執行時間
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    print(f"{self.name} | Episode {self.g_ep.value} | Time: {episode_time:.2f}s | Steps: {episode_steps} | Reward: {ep_r:.2f}")
                    break
                
                s = s_
                total_step += 1
        self.res_queue.put(None)

def plot_trade_history(df, trade_log, portfolio_history, initial_cash, filename, title):
    """
    [修改] 繪製收盤價、買賣點以及累積損益的可視化圖表。
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # === 第一張圖：原始的兩個子圖（價格 + PnL）===
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(20, 15), 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f'Agent Trading Performance on {title}', fontsize=22)

    # --- 上方圖表: 價格與交易訊號 ---
    ax1.plot(df.index, df['close'], label='Close Price', color='dodgerblue', alpha=0.8, linewidth=1.5)
    
    buy_indices = [item['step'] for item in trade_log if item['action'] == 'buy']
    buy_prices = [item['price'] for item in trade_log if item['action'] == 'buy']
    
    sell_indices = [item['step'] for item in trade_log if item['action'] == 'sell']
    sell_prices = [item['price'] for item in trade_log if item['action'] == 'sell']
    
    if buy_indices:
        ax1.scatter(buy_indices, buy_prices, marker='^', color='lime', s=120, label='Buy Signal', edgecolors='black', zorder=5)
    if sell_indices:
        ax1.scatter(sell_indices, sell_prices, marker='v', color='red', s=120, label='Sell Signal', edgecolors='black', zorder=5)
    
    ax1.set_ylabel('Price (USD/TWD)', fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.tick_params(axis='y', labelsize=12)

    # --- 下方圖表: 累積損益 (Cumulative PnL) ---
    if portfolio_history:
        portfolio_values = np.array(portfolio_history)
        cumulative_pnl = portfolio_values - initial_cash
        
        ax2.plot(df.index, cumulative_pnl, label='Cumulative PnL', color='purple', linewidth=2)
        
        # 為正負損益區域上色
        ax2.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl >= 0), color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl < 0), color='red', alpha=0.3, interpolate=True)
        
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax2.set_ylabel('Cumulative PnL (USD)', fontsize=15)
        ax2.grid(True)
    
    ax2.set_xlabel(f'Time Steps in {title}', fontsize=15)
    ax2.tick_params(axis='both', labelsize=12)
    
    # 調整佈局
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.1) # 為 suptitle 留出空間，並減少子圖間距
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"交易可視化圖表已儲存至：{filename}")
    
    # === 第二張圖：單獨的 PnL 圖 ===
    if portfolio_history:
        # 創建新的圖形
        fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(12, 8))
        fig_pnl.suptitle(f'Agent Trading Performance on {title}', fontsize=24)
        
        # 計算累積損益
        portfolio_values = np.array(portfolio_history)
        cumulative_pnl = portfolio_values - initial_cash
        
        # 繪製累積損益線
        ax_pnl.plot(df.index, cumulative_pnl, color='purple', linewidth=2)
        
        # 為正負損益區域上色
        ax_pnl.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl >= 0), 
                           color='green', alpha=0.3, interpolate=True)
        ax_pnl.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl < 0), 
                           color='red', alpha=0.3, interpolate=True)
        
        # 添加零線
        ax_pnl.axhline(0, color='grey', linestyle='--', linewidth=1)
        
        # 設置標籤和格式
        ax_pnl.set_xlabel(f'Time Steps in {title}', fontsize=22)
        ax_pnl.set_ylabel('Cumulative PnL (USD)', fontsize=22)
        ax_pnl.grid(True, alpha=0.3)
        ax_pnl.tick_params(axis='both', labelsize=20)
        
        # 調整佈局
        plt.tight_layout()
        plt.show()
        plt.close()

def run_evaluation(df, gnet, lstm_model, device, title, filename):
    """
    [修改] 執行評估、繪圖，並返回包含所有結果的字典。
    """
    print(f"\n在 {title} 上執行評估...")
    eval_df = df.reset_index(drop=True)
    eval_env = TradingEnv(eval_df, lstm_model, device)
    
    s = eval_env.reset()
    done = False
    trade_log_for_plot = []
    
    while not done:
        a = gnet.choose_best_action(v_wrap(s.unsqueeze(0)))
        s_next, r, done, _, info = eval_env.step(a)
        
        action_taken = info.get('action_taken', 'hold')
        if action_taken in ['buy', 'sell']:
            price = eval_df['close'].iloc[eval_env.current_step-1]
            trade_log_for_plot.append({
                'step': eval_env.current_step-1,
                'action': action_taken,
                'price': price
            })
        s = s_next

    portfolio_history = eval_env.portfolio_history
    padding_size = len(eval_df) - len(portfolio_history)
    padded_history = [eval_env.initial_cash] * padding_size + portfolio_history

    # 繪圖函式保持不變
    plot_trade_history(
        df=eval_df, 
        trade_log=trade_log_for_plot, 
        portfolio_history=padded_history,
        initial_cash=eval_env.initial_cash,
        filename=filename, 
        title=title
    )

    # 計算並取得績效指標
    performance_metrics = calculate_and_print_metrics(eval_env, title, trade_log_for_plot)
    print(f"{title} 評估與繪圖完成。")

    # 將所有結果打包成一個字典並返回
    results = {
        'trade_log': trade_log_for_plot,
        'portfolio_history': portfolio_history,
        'initial_cash': eval_env.initial_cash,
        'final_portfolio_value': eval_env.portfolio_value,
        'performance_metrics': performance_metrics
    }
    return results

def calculate_and_print_metrics(eval_env, title, trade_log_for_plot):
    """
    [修改] 計算、印出並返回詳細的交易績效指標字典。
    """
    print(f"\n--- {title} 績效指標 ---")
    metrics = {} # 新增：用來儲存指標的字典

    # 1. 總回報率 (Total Return)
    initial_value = eval_env.initial_cash
    final_value = eval_env.portfolio_value
    total_return_pct = ((final_value - initial_value) / initial_value) * 100
    metrics['total_return_pct'] = total_return_pct
    print(f"1. 總回報率 (Total Return): {total_return_pct:.2f}%")

    # 2. 最大回撤 (Max Drawdown)
    portfolio_history = np.array(eval_env.portfolio_history)
    max_drawdown_pct = 0.0
    if len(portfolio_history) >= 2:
        peaks = np.maximum.accumulate(portfolio_history)
        drawdowns = (peaks - portfolio_history) / peaks
        if np.any(drawdowns > 0):
            max_drawdown_pct = np.max(drawdowns) * 100
    metrics['max_drawdown_pct'] = max_drawdown_pct
    print(f"2. 最大回撤 (Max Drawdown): {max_drawdown_pct:.2f}%")

    # --- 從交易日誌計算勝率等指標 ---
    completed_trades = []
    active_buy_prices = []
    trade_amount = eval_env.trade_amount_per_time

    for trade in trade_log_for_plot:
        if trade['action'] == 'buy':
            active_buy_prices.append(trade['price'])
        elif trade['action'] == 'sell' and len(active_buy_prices) > 0:
            sell_price = trade['price']
            avg_buy_price = np.mean(active_buy_prices)
            cost_basis = len(active_buy_prices) * trade_amount
            if cost_basis > 0:
                total_shares_bought = sum([trade_amount / p for p in active_buy_prices if p > 0])
                profit = (sell_price - avg_buy_price) * total_shares_bought
                trade_return = (profit / cost_basis) * 100
                completed_trades.append({'profit': profit, 'return_pct': trade_return})
            active_buy_prices = []

    total_trades = len(completed_trades)
    metrics['total_trades'] = total_trades
    print(f"3. 總交易次數 (Total Trades): {total_trades}")

    if total_trades > 0:
        winning_trades = sum(1 for t in completed_trades if t['profit'] > 0)
        win_rate_pct = (winning_trades / total_trades) * 100
        trade_returns = [t['return_pct'] for t in completed_trades]
        best_trade_pct = max(trade_returns)
        worst_trade_pct = min(trade_returns)
        metrics.update({
            'win_rate_pct': win_rate_pct,
            'best_trade_pct': best_trade_pct,
            'worst_trade_pct': worst_trade_pct
        })
        print(f"4. 勝率 (Win Rate): {win_rate_pct:.2f}%")
        print(f"5. 最佳交易 (Best Trade): {best_trade_pct:.2f}%")
        print(f"6. 最差交易 (Worst Trade): {worst_trade_pct:.2f}%")
    else:
        metrics.update({'win_rate_pct': 0, 'best_trade_pct': 0, 'worst_trade_pct': 0})
        # ... 省略印出 N/A ...

    print("-" * 25)
    
    return metrics # 新增：返回指標字典

# --- 主程式 ---
A3C_PARAMS = {
    'update_global_iter': 30,
    'gamma': 0.995,
    'max_ep': 2000,  # 500
    'lr': 1e-5,  # 1e-5
    'entropy_beta': 0.05,  # 0.01
    'train_split': 0.8,
    'weight_path': None,  # Path to checkpoint model to resume training from
}

if __name__ == "__main__":
    # --- 步驟 1: 載入預訓練 LSTM 模型 ---
    print("載入預訓練的 QLSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化 QLSTM 模型
    qlstm_cell = CustomQLSTMCell(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        output_size=qlstm_params['output_size'],
        vqc_depth=qlstm_params['qnn_depth']
    ).float().to(device)
    
    lstm_model = CustomLSTM(
        input_size=qlstm_params['input_size'],
        hidden_size=qlstm_params['hidden_size'],
        lstm_cell_QT=qlstm_cell
    ).float().to(device)
    
    model_path = os.path.join(project_root, 'QLSTM', 'models', 'qlstm_model_epochs_50.pth')
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()  # Set to evaluation mode
    # lstm_model.share_memory()  # Removed - can't pickle quantum devices
    print("QLSTM 模型載入完成。")

    # --- 步驟 2: 準備資料 ---
    print("準備交易資料...")
    data_path = os.path.join(project_root, 'QLSTM', 'USD_TWD_Historical Data.csv')
    # 使用全新的、穩健的函式來準備資料
    full_data_df = prepare_trading_data(file_path=data_path, num_rows=10000)

    # [新增資料驗證程序] 檢查是否存在價格為 0 的異常資料
    zero_price_rows = full_data_df[full_data_df['close'] == 0]
    if not zero_price_rows.empty:
        print("錯誤：在資料中發現 'close' 價格為 0 的異常行：")
        print(zero_price_rows)
        raise ValueError("資料驗證失敗：'close' 價格不應為 0。請檢查原始資料檔案。")

    if full_data_df.empty:
        raise ValueError("錯誤：經過資料清理後，沒有剩餘的有效資料可供訓練。請檢查原始資料檔案。")

    split_point = int(len(full_data_df) * A3C_PARAMS['train_split'])
    train_df = full_data_df[:split_point]
    print(f"資料準備完成，共 {len(train_df)} 筆訓練資料。")
    print('*** training info ***')
    print(train_df.head(2), train_df.tail(2))

    # --- 步驟 3: 初始化 A3C ---
    dummy_env = TradingEnv(train_df, lstm_model, device)
    N_S = dummy_env.observation_space_shape[0]
    N_A = dummy_env.action_space_n
    del dummy_env

    gnet = Net(N_S, N_A)
    print_model_summary(gnet, model_name="QA3C Agent")
    raise ValueError()
    gnet.share_memory()
    
    # Load checkpoint if weight_path is provided
    if A3C_PARAMS['weight_path'] is not None:
        # Build full path using project_root if path is relative
        if not os.path.isabs(A3C_PARAMS['weight_path']):
            checkpoint_path = os.path.join(project_root, A3C_PARAMS['weight_path'])
        else:
            checkpoint_path = A3C_PARAMS['weight_path']
            
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                gnet.load_state_dict(checkpoint['model_state_dict'])
                start_episode = checkpoint.get('episode', 0)
                print(f"Resumed from episode {start_episode}")
            else:
                # Handle old format where only state_dict is saved
                gnet.load_state_dict(checkpoint)
                start_episode = 0
                print("Loaded checkpoint (old format, starting from episode 0)")
        else:
            start_episode = 0
            print(f"Warning: Checkpoint path '{checkpoint_path}' not found, starting fresh")
    else:
        start_episode = 0
    
    opt = SharedAdam(gnet.parameters(), lr=A3C_PARAMS['lr'], betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', start_episode), mp.Value('d', 0.), mp.Queue()

    # --- 步驟 4: 啟動並行訓練 ---
    num_workers = min(mp.cpu_count(), 10)
    print(f'[cpu_count] {mp.cpu_count()}')
    print(f"啟動 {num_workers} 個 Worker 進行並行訓練...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(project_root, 'QA3C', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Define checkpoint saving function
    def save_checkpoint_periodically(gnet, global_ep, interval=14):
        last_saved_ep = 0
        while global_ep.value < A3C_PARAMS['max_ep']:
            current_ep = global_ep.value
            if current_ep > 0 and current_ep % interval == 0 and current_ep > last_saved_ep:
                checkpoint = {
                    'model_state_dict': gnet.state_dict(),
                    'episode': current_ep,
                    'timestamp': time.strftime('%Y%m%d_%H%M%S')
                }
                checkpoint_path = os.path.join(models_dir, f'qa3c_model_ep_{current_ep}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"\n[Checkpoint] Saved model at episode {current_ep} to {checkpoint_path}")
                last_saved_ep = current_ep
            time.sleep(2)  # Check every 2 seconds
    
    # Start checkpoint saving thread
    checkpoint_thread = threading.Thread(
        target=save_checkpoint_periodically,
        args=(gnet, global_ep),
        daemon=True
    )
    checkpoint_thread.start()
    
    training_start_time = time.time()  # 記錄整體訓練開始時間
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, train_df, model_path, device) for i in range(num_workers)]
    [w.start() for w in workers]
    
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    training_total_time = time.time() - training_start_time  # 計算總訓練時間
    print(f"\n訓練完成。總訓練時間: {training_total_time:.2f} 秒")
    print(f"總共執行 {len(res)} 個 episodes")
    print(f"平均每個 episode 執行時間: {training_total_time/len(res):.2f} 秒")

    # --- 步驟 5: 儲存與繪圖 ---
    print("儲存模型與繪製結果...")
    full_plotting(_fileTitle="A3C_Trading_Agent", _trainingLength=len(res), _currentRewardList=res)
    
    # Save final model with episode number
    final_episode = global_ep.value
    final_checkpoint = {
        'model_state_dict': gnet.state_dict(),
        'episode': final_episode,
        'timestamp': time.strftime('%Y%m%d_%H%M%S')
    }
    
    # Save to models directory with episode number
    final_model_path = os.path.join(models_dir, f'qa3c_model_ep_{final_episode}_final.pth')
    torch.save(final_checkpoint, final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    # Also save with original filename for backward compatibility
    torch.save(gnet.state_dict(), "A3C_trading_model.pth")
    print("結果已儲存。")
    
    # --- [修改] 步驟 6: 在訓練集與測試集上評估並繪製交易點位圖 ---
    all_results = {}
    print("\n--- 開始執行評估 ---")
    
    # 評估訓練集
    train_results = run_evaluation(
        df=train_df, 
        gnet=gnet, 
        lstm_model=lstm_model, 
        device=device,
        title="Training Data",
        filename="A3C_trade_visual_train.png"
    )
    all_results['training'] = train_results
    
    # 評估測試集
    test_df = full_data_df[split_point:]
    print('*** testing info ***')
    print(train_df.head(2), train_df.tail(2))
    test_results = run_evaluation(
        df=test_df,
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Testing Data",
        filename="A3C_trade_visual_test.png"
    )
    all_results['testing'] = test_results

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    pickle_filepath = os.path.join(results_dir, "all_trading_results.pkl")
    
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n所有 train 和 test 的交易結果已成功儲存至: {pickle_filepath}")
    print("\n所有評估與繪圖完成。")
