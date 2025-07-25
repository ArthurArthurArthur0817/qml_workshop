import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import pickle
import vectorbt as vbt
from plotly.subplots import make_subplots

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
    [修改] 專為交易環境設計的資料準備函式.
    包含 ma5, ma10 技術指標, 並確保日期欄位被正確處理.
    """
    df = pd.read_csv(file_path)
    
    # [新增] 確保日期格式正確並設為索引
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

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
# class TradingEnv:
#     """一個模擬金融交易的自訂 Gym 環境"""
#     def __init__(self, df, lstm_model, device):
#         self.df = df
#         self.lstm_model = lstm_model
#         self.device = device
#         self.sequence_length = lstm_params['sequence_length']
#         self.n_features = len(lstm_params['feature_columns'])
        
#         # State Vector: LSTM outputs + 4 個額外特徵
#         #   - 現金比例: 目前現金 / 初始現金
#         #   - 持倉價值比例: (持有股數 * 現價) / 初始現金
#         #   - 未實現損益百分比: (現價 - 平均買入價) / 平均買入價
#         #   - 價格與短期均線的相對位置: (現價 - 5日均線) / 5日均線
#         self.observation_space_shape = (lstm_params['output_size'] + 5,)
#         self.action_space_n = 3

#         self.initial_cash = 100000
#         self.trade_amount = 50000
#         self.stop_loss_threshold = 0.03
#         self.drawdown_penalty = 1.0  # drawdown penalty 的權重
#         self.commission_rate = 0.0002  # 貼近大型外匯商

#         self.reset()

#     def _get_state(self):
#         """
#         [修改] 根據當前 step，取得 LSTM 輸出並結合額外市場資訊作為 state。
#         新的 state 包含：
#         1. LSTM 的漲跌預測機率 (2-dim)
#         2. 標準化後的現金比例 (1-dim)
#         3. 持有部位價值佔初始資金的比例 (1-dim)
#         4. 未實現損益百分比 (1-dim)
#         5. 價格與短期均線的相對位置 (1-dim)
#         6. 成本與現價的相對位置 (1-dim)
#         """
#         # 1. LSTM 輸出 (市場趨勢預測)
#         if self.current_step < self.sequence_length - 1:
#             lstm_state = torch.zeros(lstm_params['output_size'])
#         else:
#             start = self.current_step - self.sequence_length + 1
#             end = self.current_step + 1
#             sequence_df = self.df.iloc[start:end]
            
#             x = torch.tensor(sequence_df[lstm_params['feature_columns']].values, dtype=torch.float32).unsqueeze(0)
            
#             x_normalized = normalize_sequences(x)
#             x_normalized = x_normalized.to(self.device)

#             self.lstm_model.eval()
#             with torch.no_grad():
#                 logits = self.lstm_model(x_normalized)
#                 lstm_state = F.softmax(logits, dim=1).squeeze(0).cpu()

#         # 2. 新增的四個維度，以提供更完整的倉位和市場資訊
#         current_price = self.df['close'].iloc[self.current_step]

#         # 2.1 標準化後的現金比例
#         cash_ratio = self.cash / self.initial_cash

#         # 2.2 目前持有部位價值佔初始資金的比例 (標準化後的風險暴露)
#         holdings_value = sum(self.active_trades_shares) * current_price
#         holdings_ratio = holdings_value / self.initial_cash

#         # 2.3 未實現損益百分比
#         avg_buy_price = 0.0
#         unrealized_pnl_percentage = 0.0
#         avg_price_ratio = 0.0
#         total_shares = sum(self.active_trades_shares)
#         if total_shares > 0:
#             avg_buy_price = np.average(
#                 self.active_trades_buy_prices,
#                 weights=self.active_trades_shares
#             )
#             if avg_buy_price > 0:  # 未實現損益百分比
#                 unrealized_pnl_percentage = (current_price - avg_buy_price) / avg_buy_price
            
#             if current_price > 0:  # 成本與現價的相對位置
#                 avg_price_ratio = (current_price - avg_buy_price) / current_price


#         # 2.4 價格與短期均線的相對位置
#         ma5 = self.df['ma5'].iloc[self.current_step]
#         price_ma_ratio = 0.0
#         if ma5 > 0:
#             price_ma_ratio = (current_price - ma5) / ma5
            
#         # 組合 state
#         additional_state = torch.tensor([
#             cash_ratio,
#             holdings_ratio,
#             unrealized_pnl_percentage,
#             price_ma_ratio,
#             avg_price_ratio 
#         ], dtype=torch.float32)

#         full_state = torch.cat((lstm_state, additional_state))
#         return full_state

#     def _reset_trade_state(self):
#         """重置當前進行中的交易狀態"""
#         self.active_trades_buy_prices = []
#         self.active_trades_shares = []

#     def reset(self):
#         """重置整個環境狀態，包括新的一輪交易"""
#         self.current_step = self.sequence_length - 1
#         self.cash = self.initial_cash
#         self.portfolio_value = self.initial_cash
#         self.portfolio_history = []
#         self.historical_trades = []
#         self._reset_trade_state()
#         return self._get_state()

#     def step(self, action):
#         self.current_step += 1
#         done = self.current_step >= len(self.df) - 1
#         current_price = self.df['close'].iloc[self.current_step]
        
#         info = {'action_taken': 'hold'}
#         reward = 0
#         realized_profit = 0
        
#         # --- 步驟 0: 強制 stop loss (優先於 Agent 的決定) ---
#         stop_loss_triggered = False
#         total_shares = sum(self.active_trades_shares)
        
#         if total_shares > 0:
#             avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
#             unrealized_pnl_percentage = (current_price - avg_buy_price) / avg_buy_price
            
#             # 如果未實現虧損達到了我們的停損閾值
#             if unrealized_pnl_percentage <= -self.stop_loss_threshold:
#                 stop_loss_triggered = True
#                 info['action_taken'] = 'stop_loss_sell' # 標記為停損賣出
                
#                 # 執行賣出邏輯 (含手續費)
#                 realized_profit = (current_price - avg_buy_price) * total_shares
#                 revenue = total_shares * current_price
#                 self.cash += revenue * (1 - self.commission_rate)
                
#                 # 記錄交易歷史
#                 trade_log = {
#                     'buy_prices': self.active_trades_buy_prices.copy(),
#                     'buy_shares': self.active_trades_shares.copy(),
#                     'sell_price': current_price,
#                     'total_shares': total_shares,
#                     'profit': realized_profit,
#                     'step': self.current_step
#                 }
#                 self.historical_trades.append(trade_log)
                
#                 # 清空持倉
#                 self._reset_trade_state()
        
#         # --- 步驟 1: 只有在「未觸發停損」時，才執行 Agent 的決定 ---
#         if not stop_loss_triggered:
#             # 動作 1: 買入
#             if action == 1:
#                 # 計算手續費
#                 cost_with_fee = self.trade_amount * (1 + self.commission_rate)
#                 if self.cash >= cost_with_fee:
#                     shares_bought = self.trade_amount / current_price
#                     self.cash -= cost_with_fee  # 扣除包含手續費的總成本
#                     self.active_trades_buy_prices.append(current_price)
#                     self.active_trades_shares.append(shares_bought)
#                     info['action_taken'] = 'buy'
#                 else:
#                     info['action_taken'] = 'invalid_buy'

#             # 動作 2: 賣出 (由 Agent 決定)
#             elif action == 2:
#                 if total_shares > 0:
#                     avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
#                     realized_profit = (current_price - avg_buy_price) * total_shares
#                     # 計算手續費
#                     revenue = total_shares * current_price
#                     self.cash += revenue * (1 - self.commission_rate)
#                     info['action_taken'] = 'sell'
                    
#                     trade_log = {
#                         'buy_prices': self.active_trades_buy_prices.copy(),
#                         'buy_shares': self.active_trades_shares.copy(),
#                         'sell_price': current_price, 'total_shares': total_shares,
#                         'profit': realized_profit, 'step': self.current_step
#                     }
#                     self.historical_trades.append(trade_log)
#                     self._reset_trade_state()
#                 else:
#                     info['action_taken'] = 'invalid_sell'
                    
#         # --- 步驟 2: 計算獎勵 ---
#         # 計算此步驟開始前的 portfolio value，作為正規化的基準
#         previous_portfolio_value = self.portfolio_value
#         current_holdings_value = sum(self.active_trades_shares) * current_price
#         new_portfolio_value = self.cash + current_holdings_value

#         # 更新環境狀態，以供下一個時間步使用
#         self.portfolio_value = new_portfolio_value
#         self.portfolio_history.append(self.portfolio_value)

#         # 將停損賣出和一般賣出都視為已實現利潤
#         reward = 0.0
#         if previous_portfolio_value > 1:  # 避免除 0，計算公式為 log(目前價值 / 先前價值)
#             reward = np.log(new_portfolio_value / previous_portfolio_value)
            
#         # 對 "持有" 動作施加一個極小的懲罰
#         if info['action_taken'] == 'hold':
#             reward -= 0.0001  # 0.001% 的持有成本，鼓勵模型行動

#         # 對無效行為，施加一個小的、固定的懲罰 (這些懲罰也應該是百分比尺度，以保持一致性)
#         if info['action_taken'] in ['invalid_buy', 'invalid_sell']:
#             reward -= 0.0005  # 無效操作，給予 0.05% 的固定懲罰

#         if info['action_taken'] == 'stop_loss_sell':
#             reward -= 0.001   # 強制停損，給予 0.1% 的固定懲罰

#         # # (可選) 風險懲罰項：波動率懲罰
#         # # 鼓勵模型在賺取同等回報下，選擇波動更小的路徑
#         # if len(self.portfolio_history) > 20:
#         #     # 計算最近 20 筆資產淨值的對數回報率的標準差
#         #     log_returns = np.log(np.array(self.portfolio_history[-21:]) / np.array(self.portfolio_history[-22:-1]))
#         #     portfolio_volatility = np.std(log_returns)
            
#         #     # 懲罰值應非常小，避免蓋過主要的回報獎勵
#         #     risk_penalty = portfolio_volatility * 0.05 
#         #     reward -= risk_penalty

#         # 最後，將單步獎勵裁剪到一個合理的範圍內
#         # 這可以防止因價格劇烈波動產生的極端獎勵值，干擾模型訓練的穩定性
#         # 這裡限制單步回報率在 -5% 到 +5% 之間
#         reward = np.clip(reward, -0.05, 0.05)

#         # --- 步驟 3: 準備並回傳結果 ---
#         next_state = self._get_state()
#         info['historical_trades'] = self.historical_trades

#         return next_state, reward, done, False, info
class TradingEnv:
    """
    一個模擬金融交易的自訂 Gym 環境
    [重大變更]
    1. 倉位管理簡化：一次只允許持有一筆倉位。
    2. 獎勵函數革新：使用「超額報酬」(策略報酬 - 大盤報酬) 作為獎勵。
    """
    def __init__(self, df, lstm_model, device):
        self.df = df
        self.lstm_model = lstm_model
        self.device = device
        self.sequence_length = lstm_params['sequence_length']
        self.n_features = len(lstm_params['feature_columns'])
        
        # [變更] State 維度不變，但內部計算會基於新的倉位邏輯
        self.observation_space_shape = (lstm_params['output_size'] + 5,)
        self.action_space_n = 3

        # [變更] 依據您的新設定
        self.initial_cash = 100000
        self.trade_amount = 50000
        self.stop_loss_threshold = 0.03
        self.commission_rate = 0.0002

        # --- [核心變更 1] 倉位管理簡化 ---
        # [移除] 不再使用 active_trades_buy_prices 和 active_trades_shares 列表
        self.position_shares = 0.0
        self.position_avg_price = 0.0
        
        self.reset()

    def _get_state(self):
        # 1. LSTM 輸出 (邏輯不變)
        if self.current_step < self.sequence_length - 1:
            lstm_state = torch.zeros(lstm_params['output_size'])
        else:
            start = self.current_step - self.sequence_length + 1
            end = self.current_step + 1
            sequence_df = self.df.iloc[start:end]
            x = torch.tensor(sequence_df[lstm_params['feature_columns']].values, dtype=torch.float32).unsqueeze(0)
            x_normalized = normalize_sequences(x).to(self.device)
            self.lstm_model.eval()
            with torch.no_grad():
                logits = self.lstm_model(x_normalized)
                lstm_state = F.softmax(logits, dim=1).squeeze(0).cpu()

        # 2. 根據新的倉位變數計算 State 特徵
        current_price = self.df['close'].iloc[self.current_step]
        cash_ratio = self.cash / self.initial_cash
        
        # [變更] holdings_value 的計算
        holdings_value = self.position_shares * current_price
        holdings_ratio = holdings_value / self.initial_cash

        # [變更] PnL 的計算
        unrealized_pnl_percentage = 0.0
        avg_price_ratio = 0.0
        if self.position_shares > 0:
            if self.position_avg_price > 0:
                unrealized_pnl_percentage = (current_price - self.position_avg_price) / self.position_avg_price
            if current_price > 0:
                avg_price_ratio = (current_price - self.position_avg_price) / current_price

        ma5 = self.df['ma5'].iloc[self.current_step]
        price_ma_ratio = (current_price - ma5) / ma5 if ma5 > 0 else 0.0
            
        additional_state = torch.tensor([
            cash_ratio,
            holdings_ratio,
            unrealized_pnl_percentage,
            price_ma_ratio,
            avg_price_ratio 
        ], dtype=torch.float32)

        return torch.cat((lstm_state, additional_state))

    def reset(self):
        """ [變更] 重置整個環境狀態，包含新的倉位變數 """
        self.current_step = self.sequence_length - 1
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.portfolio_history = [self.initial_cash] * (self.sequence_length) # 初始化足夠長的歷史紀錄
        self.historical_trades = []
        
        # 直接重置倉位變數
        self.position_shares = 0.0
        self.position_avg_price = 0.0
        
        return self._get_state()

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df['close'].iloc[self.current_step]
        
        info = {'action_taken': 'hold'}
        
        # --- [核心變更 1] 倉位管理簡化 ---
        # --- 步驟 0: 強制停損 ---
        stop_loss_triggered = False
        if self.position_shares > 0:
            if (current_price - self.position_avg_price) / self.position_avg_price <= -self.stop_loss_threshold:
                stop_loss_triggered = True
                info['action_taken'] = 'stop_loss_sell'
                # 執行賣出邏輯
                revenue = self.position_shares * current_price
                self.cash += revenue * (1 - self.commission_rate)
                # 記錄與清空倉位
                # ... (賣出邏輯統一寫在下面)
        
        # --- 步驟 1: 執行 Agent 動作 (若未停損) ---
        if not stop_loss_triggered:
            # 動作 1: 買入
            if action == 1:
                # 只有在沒有持倉時才允許買入，實現「單一倉位」規則
                if self.position_shares == 0:
                    cost_with_fee = self.trade_amount * (1 + self.commission_rate)
                    if self.cash >= cost_with_fee:
                        info['action_taken'] = 'buy'
                        self.position_shares = self.trade_amount / current_price
                        self.position_avg_price = current_price
                        self.cash -= cost_with_fee
                    else:
                        info['action_taken'] = 'invalid_buy'
                else:
                    # 如果已有持倉，則視為無效操作
                    info['action_taken'] = 'invalid_buy'

            # 動作 2: 賣出
            elif action == 2:
                # 只有在有持倉時才允許賣出
                if self.position_shares > 0:
                    info['action_taken'] = 'sell'
                else:
                    info['action_taken'] = 'invalid_sell'
        
        # --- 統一處理所有賣出事件 (來自Agent或停損) ---
        if info['action_taken'] in ['sell', 'stop_loss_sell']:
            realized_profit = (current_price - self.position_avg_price) * self.position_shares
            revenue = self.position_shares * current_price
            # 僅在 stop_loss_sell 時上面已更新過 cash，這裡不再重複
            if info['action_taken'] == 'sell':
                 self.cash += revenue * (1 - self.commission_rate)

            trade_log = {
                'buy_price': self.position_avg_price,
                'sell_price': current_price,
                'total_shares': self.position_shares,
                'profit': realized_profit,
                'step': self.current_step
            }
            self.historical_trades.append(trade_log)
            # 賣出後清空倉位
            self.position_shares = 0.0
            self.position_avg_price = 0.0

        # --- [核心變更 2] 獎勵函數革新：超額報酬 ---
        previous_portfolio_value = self.portfolio_value
        current_holdings_value = self.position_shares * current_price
        new_portfolio_value = self.cash + current_holdings_value
        self.portfolio_value = new_portfolio_value
        self.portfolio_history.append(new_portfolio_value)
        
        reward = 0.0
        # 確保有足夠的歷史資料來計算
        if self.current_step > 0 and previous_portfolio_value > 0:
            # 計算策略的對數報酬率
            strategy_return = np.log(new_portfolio_value / previous_portfolio_value)
            # 計算基準(大盤)的對數報酬率
            benchmark_return = np.log(self.df['close'].iloc[self.current_step] / self.df['close'].iloc[self.current_step - 1])
            # Reward = 超額報酬
            reward = strategy_return - benchmark_return
        
        # [變更] 移除 hold 懲罰和 clip，但保留對無效/強制行為的懲罰
        if info['action_taken'] in ['invalid_buy', 'invalid_sell']:
            reward -= 0.0001  # 一個微小的固定懲罰
        
        if info['action_taken'] == 'stop_loss_sell':
            reward -= 0.0005  # 停損是更嚴重的錯誤，給予更大懲罰

        # --- 步驟 3: 準備並回傳結果 ---
        next_state = self._get_state()
        info['historical_trades'] = self.historical_trades
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

        entropy = m.entropy()

        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss - A3C_PARAMS['entropy_beta'] * entropy).mean()
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

def run_evaluation(df, gnet, lstm_model, device, title):
    """
    在給定的資料集上執行評估.
    此函式現在會回傳 vectorbt 需要的 price, entries, exits 資料.
    """
    print(f"\n在 {title} 上執行評估...")
    eval_df_for_env = df.reset_index(drop=True) # 環境需要數字索引
    eval_env = TradingEnv(eval_df_for_env, lstm_model, device)
    
    s = eval_env.reset()
    done = False
    
    # 初始化訊號列表
    entries = [False] * len(df)
    exits = [False] * len(df)
    
    while not done:
        a = gnet.choose_best_action(v_wrap(s.unsqueeze(0)))
        s_next, r, done, _, info = eval_env.step(a)
        
        current_step_idx = eval_env.current_step
        action_taken = info.get('action_taken', 'hold')

        if action_taken == 'buy':
            entries[current_step_idx] = True
        elif action_taken in ['sell', 'stop_loss_sell']: # [修改] 將停損賣出也視為出場訊號
            exits[current_step_idx] = True
        
        s = s_next

    # --- 產生 vectorbt 所需的格式 ---
    price_series = df['close']
    entry_signals = pd.Series(entries, index=df.index)
    exit_signals = pd.Series(exits, index=df.index)
    
    print(f"{title} 評估完成。")
    return price_series, entry_signals, exit_signals

def analyze_and_plot_comparison(train_signals, test_signals, filename_prefix):
    """
    產生訓練集與測試集的並列比較圖 (3x2)，並分別印出統計數據。
    【修正版】: 使用 plotly.subplots 來正確處理子圖。
    """
    # --- 建立投資組合 ---
    common_pf_kwargs = {
        'init_cash': 100_000,
        'fees': 0.0002,
        'freq': '1D',
        'size': 50000,          # 每次買入的現金金額，與 self.trade_amount 一致
        'size_type': 'amount'   # 告訴 vectorbt，size 的單位是'金額'
    }
    
    pf_train = vbt.Portfolio.from_signals(
        train_signals['price'], 
        train_signals['entries'], 
        train_signals['exits'],
        **common_pf_kwargs
    )
    
    pf_test = vbt.Portfolio.from_signals(
        test_signals['price'], 
        test_signals['entries'], 
        test_signals['exits'],
        **common_pf_kwargs
    )

    # --- 印出統計數據 ---
    print("\n" + "="*40)
    print("--- 訓練集 (Training Set) 表現 ---")
    print("="*40)
    print(pf_train.stats())
    
    print("\n" + "="*40)
    print("--- 測試集 (Test Set) 表現 ---")
    print("="*40)
    print(pf_test.stats())

    # --- 步驟 1: 使用 plotly 建立 (3, 2) 的子圖框架 ---
    fig = make_subplots(
        rows=3, 
        cols=2, 
        shared_xaxes=False,
        # 設定每個子圖的標題
        subplot_titles=(
            'Training Set Orders', 'Test Set Orders', 
            'Training Set Trade PnL', 'Test Set Trade PnL', 
            'Training Set Cumulative Returns', 'Test Set Cumulative Returns'
        ),
        vertical_spacing=0.1
    )

    # --- 步驟 2: 提取圖表數據 (Traces) 並新增到主圖框架中 ---
    
    # 處理圖例，避免重複
    added_legends = set()
    def add_traces_to_fig(source_fig, row, col):
        for trace in source_fig.data:
            # 如果圖例名稱沒有出現過，則正常顯示，否則隱藏
            if trace.name not in added_legends:
                added_legends.add(trace.name)
                trace.showlegend = True
            else:
                trace.showlegend = False
            fig.add_trace(trace, row=row, col=col)

    # -- Orders Plots (Row 1) --
    add_traces_to_fig(pf_train.plot_orders(), row=1, col=1)
    add_traces_to_fig(pf_test.plot_orders(), row=1, col=2)
    
    # -- Trade PnL Plots (Row 2) --
    add_traces_to_fig(pf_train.trades.plot_pnl(), row=2, col=1)
    add_traces_to_fig(pf_test.trades.plot_pnl(), row=2, col=2)

    # -- Cumulative Returns Plots (Row 3) --
    add_traces_to_fig(pf_train.cumulative_returns().vbt.plot(), row=3, col=1)
    add_traces_to_fig(pf_test.cumulative_returns().vbt.plot(), row=3, col=2)

    # --- 步驟 3: 更新整體圖表佈局並儲存/顯示 ---
    fig.update_layout(
        height=1200, 
        title_text='Training vs. Test Set Performance Comparison',
        showlegend=True # 顯示圖例
    )
    
    plot_filename = f"{filename_prefix}_comparison_plot.png"
    # 使用 write_image 儲存靜態圖片
    # fig.write_image(plot_filename)
    print(f"\n並列比較圖表已儲存至: {plot_filename}")
    fig.show()

    # --- 儲存分析所需的資料 ---
    pickle_filename = f"{filename_prefix}_signals_data.pkl"
    signals_data = {
        'train': train_signals,
        'test': test_signals
    }
    with open(pickle_filename, 'wb') as f:
        pickle.dump(signals_data, f)
    print(f"訓練與測試訊號資料已儲存至: {pickle_filename}")

# --- 主程式 ---
A3C_PARAMS = {
    'update_global_iter': 20,
    'gamma': 0.99,
    'max_ep': 2000,
    'lr': 1e-4,
    'entropy_beta': 0.01
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

    # 檢查是否存在價格為 0 的異常資料
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
    num_workers = min(mp.cpu_count(), 16) # 方便除錯，使用單一 worker
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
    
    # --- [修改] 步驟 6: 在測試集上評估並使用 vectorbt 進行分析 ---
    print("\n--- 開始執行最終評估 ---")

    # 在訓練集上運行評估，獲取訊號
    print("\n--- 對訓練集進行評估以供比較 ---")
    train_price, train_entries, train_exits = run_evaluation(
        df=train_df, # 使用 train_df
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Training Data"
    )
    train_signals = {'price': train_price, 'entries': train_entries, 'exits': train_exits}

    # 在測試集上運行評估，獲取訊號
    print("\n--- 對測試集進行評估 ---")
    test_df = full_data_df[split_point:]
    test_price, test_entries, test_exits = run_evaluation(
        df=test_df,
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Testing Data"
    )
    test_signals = {'price': test_price, 'entries': test_entries, 'exits': test_exits}

    analyze_and_plot_comparison(train_signals, test_signals, filename_prefix="A3C_final_report")

    print("\n所有評估與分析完成。")
