import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# --- 專案路徑設定與模組引用 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 引用 QLSTM 中的資料處理函式、模型架構和參數
from QLSTM.LSTM_trading import (
    create_sequences, normalize_sequences,
    StandardLSTM, params as lstm_params
)
# 引用 A3C 的輔助工具
from QA3C.utils import v_wrap, set_init, push_and_pull, record
from QA3C.plot_functions import full_plotting
from QA3C.shared_adam import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"

def prepare_trading_data(file_path, num_rows=10000):
    """
    專為交易環境設計的資料準備函式.
    包含 ma5, ma10 技術指標都先算好, 並移除 NaN 值, 以求簡化.
    """
    df = pd.read_csv(file_path)
    
    # 預先計算好所有特徵
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # [新增資料驗證程序] 檢查是否存在價格為 0 的異常資料
    zero_price_rows = df[df['close'] == 0]
    if not zero_price_rows.empty:
        print("錯誤：在資料中發現 'close' 價格為 0 的異常行：")
        print(zero_price_rows)
        raise ValueError("資料驗證失敗：'close' 價格不應為 0。請檢查原始資料檔案。")

    if df.empty:
        raise ValueError("錯誤：經過資料清理後，沒有剩餘的有效資料可供訓練。請檢查原始資料檔案。")

    return df

# --- A3C 交易環境 ---
class TradingEnv:
    """一個模擬金融交易的自訂 Gym 環境"""
    def __init__(self, df, lstm_model, device):
        self.df = df
        self.lstm_model = lstm_model
        self.device = device
        self.sequence_length = lstm_params['sequence_length']
        self.n_features = len(lstm_params['feature_columns'])
        
        # State Vector: LSTM outputs + 4 個額外特徵
        #   - 現金比例: 目前現金 / 初始現金
        #   - 持倉價值比例: (持有股數 * 現價) / 初始現金
        #   - 未實現損益百分比: (現價 - 平均買入價) / 平均買入價
        #   - 價格與短期均線的相對位置: (現價 - 5日均線) / 5日均線
        self.observation_space_shape = (lstm_params['output_size'] + 4,)
        self.action_space_n = 3

        self.initial_cash = 100000
        self.trade_amount = 10000
        self.reset()

    def _get_state(self):
        """
        [修改] 根據當前 step，取得 LSTM 輸出並結合額外市場資訊作為 state。
        新的 state 包含：
        1. LSTM 的漲跌預測機率 (2-dim)
        2. 標準化後的現金比例 (1-dim)
        3. 持有部位價值佔初始資金的比例 (1-dim)
        4. 未實現損益百分比 (1-dim)
        5. 價格與短期均線的相對位置 (1-dim)
        """
        # 1. LSTM 輸出 (市場趨勢預測)
        if self.current_step < self.sequence_length - 1:
            lstm_state = torch.zeros(lstm_params['output_size'])
        else:
            start = self.current_step - self.sequence_length + 1
            end = self.current_step + 1
            sequence_df = self.df.iloc[start:end]
            
            x = torch.tensor(sequence_df[lstm_params['feature_columns']].values, dtype=torch.float32).unsqueeze(0)
            
            x_normalized = normalize_sequences(x)
            x_normalized = x_normalized.to(self.device)

            self.lstm_model.eval()
            with torch.no_grad():
                logits = self.lstm_model(x_normalized)
                lstm_state = F.softmax(logits, dim=1).squeeze(0).cpu()

        # 2. 新增的四個維度，以提供更完整的倉位和市場資訊
        current_price = self.df['close'].iloc[self.current_step]

        # 2.1 標準化後的現金比例
        cash_ratio = self.cash / self.initial_cash

        # 2.2 目前持有部位價值佔初始資金的比例 (標準化後的風險暴露)
        holdings_value = sum(self.active_trades_shares) * current_price
        holdings_ratio = holdings_value / self.initial_cash

        # 2.3 未實現損益百分比
        unrealized_pnl_percentage = 0.0
        total_shares = sum(self.active_trades_shares)
        if total_shares > 0:
            avg_buy_price = np.average(
                self.active_trades_buy_prices,
                weights=self.active_trades_shares
            )
            if avg_buy_price > 0:
                unrealized_pnl_percentage = (current_price - avg_buy_price) / avg_buy_price
        
        # 2.4 價格與短期均線的相對位置
        ma5 = self.df['ma5'].iloc[self.current_step]
        price_ma_ratio = 0.0
        if ma5 > 0:
            price_ma_ratio = (current_price - ma5) / ma5
            
        # 組合 state
        additional_state = torch.tensor([
            cash_ratio,
            holdings_ratio,
            unrealized_pnl_percentage,
            price_ma_ratio
        ], dtype=torch.float32)

        full_state = torch.cat((lstm_state, additional_state))
        return full_state

    def _reset_trade_state(self):
        """重置當前進行中的交易狀態"""
        self.active_trades_buy_prices = []
        self.active_trades_shares = []

    def reset(self):
        """重置整個環境狀態，包括新的一輪交易"""
        self.current_step = self.sequence_length - 1
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.historical_trades = []
        self._reset_trade_state()
        return self._get_state()

    def step(self, action):
        """執行一步動作，包含新的交易邏輯與詳細的日誌輸出"""
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        current_price = self.df['close'].iloc[self.current_step]
        reward = 0
        action_str = {0: "Hold", 1: "Buy", 2: "Sell"}.get(action, "Unknown")
        
        info = {'action_taken': 'hold'} # [修改] 初始化 info，預設為 hold

        # print(f"\n--- Step: {self.current_step}, Price: {current_price:.4f}, Action: {action_str} ---")

        # --- 執行動作 ---
        if action == 1: # Buy
            if self.cash >= self.trade_amount:
                shares_bought = self.trade_amount / current_price
                self.cash -= self.trade_amount
                self.active_trades_buy_prices.append(current_price)
                self.active_trades_shares.append(shares_bought)
                info['action_taken'] = 'buy' # [修改] 記錄實際執行的動作
                # print(f"  [Buy] Bought {shares_bought:.4f} shares. Cash left: {self.cash:.2f}")
            else:
                reward -= 0.1 # 懲罰想賣但沒倉位的行為
                # print(f"  [Buy] Insufficient cash. Wanted to buy, but did nothing.")
        
        elif action == 2: # Sell
            total_shares = sum(self.active_trades_shares)
            if total_shares > 0:
                avg_buy_price = np.average(
                    self.active_trades_buy_prices,
                    weights=self.active_trades_shares
                )
                realized_profit = (current_price - avg_buy_price) * total_shares
                reward = realized_profit # 已實現的利潤直接作為 reward
                
                self.cash += total_shares * current_price
                info['action_taken'] = 'sell' # [修改] 記錄實際執行的動作

                # print(f"  [Sell] Sold {total_shares:.4f} shares. Avg Buy Price: {avg_buy_price:.4f}, Sell Price: {current_price:.4f}")
                # print(f"  [Sell] Realized Profit for this trade: {realized_profit:.2f}")

                trade_log = {
                    'buy_prices': self.active_trades_buy_prices.copy(),
                    'buy_shares': self.active_trades_shares.copy(),
                    'sell_price': current_price,
                    'total_shares': total_shares,
                    'profit': realized_profit,
                    'step': self.current_step
                }
                self.historical_trades.append(trade_log)
                
                self._reset_trade_state()
            else:
                reward -= 0.1 # 懲罰想賣但沒倉位的行為
                # print(f"  [Sell] No shares to sell. Did nothing.")

        # --- 計算未實現損益作為密集獎勵 ---
        unrealized_pnl = 0
        if action != 2:
            current_holdings_value = sum(self.active_trades_shares) * current_price
            new_portfolio_value = self.cash + current_holdings_value
            unrealized_pnl = new_portfolio_value - self.portfolio_value
            self.portfolio_value = new_portfolio_value
        else:
            self.portfolio_value = self.cash

        reward += unrealized_pnl # 將未實現損益加入獎勵
        
        # print(f"  Portfolio: Cash {self.cash:.2f}, Holdings {sum(self.active_trades_shares):.4f} shares")
        # print(f"  Values: Unrealized PnL: {unrealized_pnl:.2f}, Step Reward: {reward:.2f}, Portfolio Value: {self.portfolio_value:.2f}")
        
        next_state = self._get_state()
        info['historical_trades'] = self.historical_trades # [修改] 將交易歷史加入 info
        
        return next_state, reward, done, False, info

# --- A3C 模型與 Worker ---
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Net, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.pi1 = nn.Linear(s_dim, 128)
        self.pi2 = nn.Linear(128, a_dim)
        self.v1 = nn.Linear(s_dim, 128)
        self.v2 = nn.Linear(128, 1)
        set_init([self.pi1, self.pi2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)
        v1 = torch.tanh(self.v1(x))
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
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()
        return total_loss

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, df, lstm_model, device):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        
        self.env = TradingEnv(df, lstm_model, device)
        self.lnet = Net(self.env.observation_space_shape[0], self.env.action_space_n)

    def run(self):
        total_step = 1
        while self.g_ep.value < A3C_PARAMS['max_ep']:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            
            while True:
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
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    break
                
                s = s_
                total_step += 1
        self.res_queue.put(None)

def plot_trade_history(df, trade_log, filename="A3C_trade_visual.png"):
    """
    [新增] 繪製收盤價與買賣點的可視化圖表。
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(20, 10))
    
    # 繪製收盤價
    plt.plot(df.index, df['close'], label='Close Price', color='dodgerblue', alpha=0.8, linewidth=1.5)
    
    # 整理買賣點
    buy_indices = [item['step'] for item in trade_log if item['action'] == 'buy']
    buy_prices = [item['price'] for item in trade_log if item['action'] == 'buy']
    
    sell_indices = [item['step'] for item in trade_log if item['action'] == 'sell']
    sell_prices = [item['price'] for item in trade_log if item['action'] == 'sell']
    
    # 繪製買賣點
    if buy_indices:
        plt.scatter(buy_indices, buy_prices, marker='^', color='lime', s=120, label='Buy Signal', edgecolors='black', zorder=5)
    if sell_indices:
        plt.scatter(sell_indices, sell_prices, marker='v', color='red', s=120, label='Sell Signal', edgecolors='black', zorder=5)
    
    plt.title('Agent Trading Actions on Test Data', fontsize=20)
    plt.xlabel('Time Steps in Test Set', fontsize=15)
    plt.ylabel('Price (USD/TWD)', fontsize=15)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    # plt.savefig(filename)
    plt.close()
    print(f"交易可視化圖表已儲存至：{filename}")

# --- 主程式 ---
A3C_PARAMS = {
    'update_global_iter': 50,
    'gamma': 0.9,
    'max_ep': 3000,
    'lr': 1e-4,
}

if __name__ == "__main__":
    # --- 步驟 1: 載入預訓練 LSTM 模型 ---
    print("載入預訓練的 LSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = StandardLSTM(
        input_size=len(lstm_params['feature_columns']),
        hidden_size=lstm_params['hidden_size'],
        output_size=lstm_params['output_size'],
        num_layers=lstm_params['num_layers']
    ).to(device)
    
    model_path = os.path.join(project_root, 'QLSTM', 'models', f"lstm_model_epochs_{lstm_params['epochs']}.pth")
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.share_memory()
    print("LSTM 模型載入完成。")

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

    split_point = int(len(full_data_df) * 0.8)
    train_df = full_data_df[:split_point]
    print(f"資料準備完成，共 {len(train_df)} 筆訓練資料。")

    # --- 步驟 3: 初始化 A3C ---
    dummy_env = TradingEnv(train_df, lstm_model, device)
    N_S = dummy_env.observation_space_shape[0]
    N_A = dummy_env.action_space_n
    del dummy_env

    gnet = Net(N_S, N_A)
    gnet.share_memory()
    opt = SharedAdam(gnet.parameters(), lr=A3C_PARAMS['lr'], betas=(0.92, 0.999))
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # --- 步驟 4: 啟動並行訓練 ---
    num_workers = min(mp.cpu_count(), 4) # 方便除錯，使用單一 worker
    print(f"啟動 {num_workers} 個 Worker 進行並行訓練...")
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, train_df, lstm_model, device) for i in range(num_workers)]
    [w.start() for w in workers]
    
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]
    print("訓練完成。")

    # --- 步驟 5: 儲存與繪圖 ---
    print("儲存模型與繪製結果...")
    full_plotting(_fileTitle="A3C_Trading_Agent", _trainingLength=len(res), _currentRewardList=res)
    torch.save(gnet.state_dict(), "A3C_trading_model.pth")
    print("結果已儲存。")
    
    # --- [新增] 步驟 6: 在測試集上評估並繪製交易點位圖 ---
    print("\n在測試集上執行評估並產生交易點位圖...")
    test_df = full_data_df[split_point:].reset_index(drop=True)
    eval_env = TradingEnv(test_df, lstm_model, device)
    
    s = eval_env.reset()
    done = False
    trade_log_for_plot = []
    
    while not done:
        a = gnet.choose_best_action(v_wrap(s.unsqueeze(0)))
        s_next, r, done, _, info = eval_env.step(a)
        
        action_taken = info.get('action_taken', 'hold')
        if action_taken in ['buy', 'sell']:
            trade_log_for_plot.append({
                'step': eval_env.current_step,
                'action': action_taken,
                'price': test_df['close'].iloc[eval_env.current_step]
            })
        
        s = s_next

    plot_trade_history(test_df, trade_log_for_plot)
    print("評估與繪圖完成。")
