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

# --- Project Path Settings and Module Import ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import data processing functions and model classes from QLSTM
from QLSTM.QLSTM_trading_final import (
    create_sequences, normalize_sequences,
    H_layer, RY_layer, entangling_layer, q_function,
    VQC, CustomQLSTMCell, CustomLSTM
)
# Import helper utilities for A3C
from QA3C.utils import v_wrap, set_init, push_and_pull, record
from QA3C.plot_functions import full_plotting
from QA3C.shared_adam import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"

def print_model_summary(model: nn.Module, model_name: str = "Model"):
 """
 Print a layered parameter summary for a PyTorch model, including the total number of parameters.

 Args:
     model (nn.Module): The PyTorch model to analyze.
     model_name (str): The name of the model to display in the summary title.
 """
    print("=" * 70)
    print(f"{model_name} Parameters Summary")
    print("-" * 70)
    print(f"{'Layer Name':<35} {'Shape':<20} {'Number of Parameters':>12}")
    print("-" * 70)
    
    total_params = 0
    
    # Iterate through all named parameters in the model
    for name, param in model.named_parameters():
        # Only count parameters that require gradient calculation (trainable parameters)
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            shape_str = str(list(param.shape))
            print(f"{name:<35} {shape_str:<20} {num_params:>12,}")
            
    print("-" * 70)
    print(f"Total Trainable Parameters: {total_params:>15,}")
    print("=" * 70)

# QLSTM parameters
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
     Data preparation function specifically designed for trading environments.
     Includes pre-calculation of technical indicators like ma5 and ma10, and removal of NaN values for simplification.
     """
    df = pd.read_csv(file_path)
    df = df[::-1].reset_index(drop=True)
    print(df.head())
    
    # Pre-calculate all features
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Check for abnormal data where the price is 0
    zero_price_rows = df[df['close'] == 0]
    if not zero_price_rows.empty:
        print("Error: Abnormal row found in data where 'close' price is 0:")
        print(zero_price_rows)
        raise ValueError("Data validation failed: 'close' price should not be 0. Please check the original data file.")

    if df.empty:
        raise ValueError("Error: After data cleaning, there is no valid data remaining for training. Please check the original data file.")

    return df

class TradingEnv:
    """
     The Proven Best Trading Environment - State Corrected Version
    
     This version corrects the state representation by removing all invalid placeholders
     and uses 8 complete, normalized financial features to construct the observations,
     providing the Agent with a rich and stable basis for decision-making.
    """
    def __init__(self, df, lstm_model, device, time_penalty=0.02):
        self.df = df.copy()
        self.lstm_model = lstm_model
        self.device = device
        self.sequence_length = qlstm_params['sequence_length']
        self.feature_columns = qlstm_params['feature_columns']
        self.time_penalty = time_penalty
        self.volatility_window = 20  # New addition: window size for calculating volatility
        
        # Pre-calculate all required technical indicators
        for window in [5, 20, 60]:
            if f'ma{window}' not in self.df.columns:
                self.df[f'ma{window}'] = self.df['close'].rolling(window=window).mean()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        
        # [Corrected] Observation space dimension = LSTM output dimension + 8 additional features
        self.observation_space_shape = (qlstm_params['output_size'] + 8,)
        self.action_space_n = 3

        self.initial_cash = 50000 # TODO
        self.trade_amount_per_time = 45000 # TODO
        self.reset()

    def _get_state(self):
        """
        # [Corrected] Generate a state vector composed of 8 meaningful features.
        """
        # 1. LSTM Output (Market trend prediction)
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

        # 2. Calculate 8 additional financial features
        current_price = self.df['close'].iloc[self.current_step]
        ma5 = self.df['ma5'].iloc[self.current_step]
        ma20 = self.df['ma20'].iloc[self.current_step]
        ma60 = self.df['ma60'].iloc[self.current_step]

        # Feature 1: Cash ratio
        cash_ratio = self.cash / self.initial_cash
        
       # Feature 2: Position value ratio
        holdings_value = sum(self.active_trades_shares) * current_price
        holdings_ratio = holdings_value / self.initial_cash
        
        # Feature 3 & 4: Unrealized profit/loss and Cost-to-current price ratio
        unrealized_pnl_pct = 0.0
        avg_price_ratio = 0.0
        if sum(self.active_trades_shares) > 0:
            avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
            if avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if current_price > 0:
                avg_price_ratio = (current_price - avg_buy_price) / current_price
        
        # Feature 5 & 6: Deviation rate of price from medium/long-term moving averages
        price_ma20_ratio = (current_price - ma20) / ma20 if ma20 > 0 else 0.0
        price_ma60_ratio = (current_price - ma60) / ma60 if ma60 > 0 else 0.0

        # Feature 7: Deviation rate of short-term from medium-term moving average (to judge trend momentum)
        ma5_ma20_ratio = (ma5 - ma20) / ma20 if ma20 > 0 else 0.0
        
        # Feature 8: Recent price volatility
        if self.current_step >= self.volatility_window:
            recent_prices = self.df['close'].iloc[self.current_step - self.volatility_window : self.current_step]
            mean_price = np.clip(np.mean(recent_prices), 1e-9, np.inf)
            price_volatility = np.std(recent_prices) / mean_price
        else:
            price_volatility = 0.0

        # Combine all features into a single tensor
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
        # Start from step 60 to ensure all MA indicators have values
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
        # Default reward is a time penalty, encouraging the Agent to take action
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
                reward += 0.5  # Immediate reward for trend-following trades
            elif self.cash >= self.trade_amount_per_time and not is_uptrend:
                reward -= 2.0  # Penalty for counter-trend buying
            else:
                info['action_taken'] = 'invalid_buy'
                reward -= 0.5  # Penalty for invalid buying

        elif action == 2:  # Sell
            total_shares = sum(self.active_trades_shares)
            if total_shares > 0:
                avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
                realized_profit = (current_price - avg_buy_price) * total_shares
                cost_basis = avg_buy_price * total_shares
                pnl_pct = realized_profit / cost_basis if cost_basis > 0 else 0

                if realized_profit > 0:
                    # if pnl_pct >= 0.015:
                    #     # Reached profit target, grant a huge reward
                    #     # reward += 10.0 + pnl_pct * 50
                    # else:
                    #     # Profitable but not yet at target, grant a smaller reward to encourage waiting
                    #     # reward += 10.0 + pnl_pct * 10
                    reward += 10.0 + pnl_pct * 50
                else:
                    cost_basis = avg_buy_price * total_shares
                    pnl_pct = realized_profit / cost_basis if cost_basis > 0 else 0
                    reward -= 2.0 + pnl_pct * 10 # Penalty for selling at a loss
                    
                self.cash += total_shares * current_price
                info['action_taken'] = 'sell'
                self._reset_trade_state()
            else:
                info['action_taken'] = 'invalid_sell'
                reward -= 0.5 # Penalty for attempting to sell when there is no position to sell
        
        # Continuous penalty for realized losses
        total_shares = sum(self.active_trades_shares)
        if total_shares > 0 and info['action_taken'] != 'sell':
            avg_buy_price = np.average(self.active_trades_buy_prices, weights=self.active_trades_shares)
            unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if unrealized_pnl_pct < 0:
                # The larger the loss, the heavier the penalty (quadratic penalty)
                reward += unrealized_pnl_pct * abs(unrealized_pnl_pct) * 5.0
        
        current_holdings_value = sum(self.active_trades_shares) * current_price
        self.portfolio_value = self.cash + current_holdings_value
        self.portfolio_history.append(self.portfolio_value)
        final_reward = np.clip(reward, -15.0, 30.0)  # [-15, 15]
        next_state = self._get_state()
        
        if info['action_taken'] == 'sell':
            self.historical_trades.append({'profit': realized_profit})

        return next_state, final_reward, done, False, info

# --- A3C Model and Worker ---
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
        """
        # [New] Select the action with the highest probability during evaluation
        """
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
            episode_start_time = time.time()  
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            episode_steps = 0  
            
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
                    episode_time = time.time() - episode_start_time  
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    print(f"{self.name} | Episode {self.g_ep.value} | Time: {episode_time:.2f}s | Steps: {episode_steps} | Reward: {ep_r:.2f}")
                    break
                
                s = s_
                total_step += 1
        self.res_queue.put(None)

def plot_trade_history(df, trade_log, portfolio_history, initial_cash, filename, title):
    """
     [Modified] Plot the visualization chart for closing prices, buy/sell points, and cumulative profit/loss.
    ""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create two subplots, stacked vertically, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(
        2, 1, 
        figsize=(20, 15), 
        sharex=True, 
        gridspec_kw={'height_ratios': [3, 1]}
    )
    fig.suptitle(f'Agent Trading Performance on {title}', fontsize=22)

    # --- Upper Plot: Price and Trading Signals ---
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

   # --- Lower Plot: Cumulative PnL (Cumulative Profit/Loss) ---
    if portfolio_history:
        portfolio_values = np.array(portfolio_history)
        cumulative_pnl = portfolio_values - initial_cash
        
        ax2.plot(df.index, cumulative_pnl, label='Cumulative PnL', color='purple', linewidth=2)
        
        # Color the area for positive and negative profit/loss
        ax2.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl >= 0), color='green', alpha=0.3, interpolate=True)
        ax2.fill_between(df.index, cumulative_pnl, where=(cumulative_pnl < 0), color='red', alpha=0.3, interpolate=True)
        
        ax2.axhline(0, color='grey', linestyle='--', linewidth=1)
        ax2.set_ylabel('Cumulative PnL (USD)', fontsize=15)
        ax2.grid(True)
    
    ax2.set_xlabel(f'Time Steps in {title}', fontsize=15)
    ax2.tick_params(axis='both', labelsize=12)
    
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.1) 
    # plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"交易可視化圖表已儲存至：{filename}")

def run_evaluation(df, gnet, lstm_model, device, title, filename):
    """
     [Modified] Execute evaluation, plotting, and return a dictionary containing all results.
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

   
    plot_trade_history(
        df=eval_df, 
        trade_log=trade_log_for_plot, 
        portfolio_history=padded_history,
        initial_cash=eval_env.initial_cash,
        filename=filename, 
        title=title
    )

    
    performance_metrics = calculate_and_print_metrics(eval_env, title, trade_log_for_plot)
    print(f"{title} Evaluation and plotting complete.")

    # Pack all results into a dictionary and return
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
     [Modified] Calculate, print, and return a dictionary of detailed trading performance metrics.
    """
    print(f"\n--- {title} Performance Metrics ---")
    metrics = {} # New addition: dictionary to store metrics

    # 1. Total Return
    initial_value = eval_env.initial_cash
    final_value = eval_env.portfolio_value
    total_return_pct = ((final_value - initial_value) / initial_value) * 100
    metrics['total_return_pct'] = total_return_pct
    print(f"1. Total Return: {total_return_pct:.2f}%")

    # 2. Max Drawdown
    portfolio_history = np.array(eval_env.portfolio_history)
    max_drawdown_pct = 0.0
    if len(portfolio_history) >= 2:
        peaks = np.maximum.accumulate(portfolio_history)
        drawdowns = (peaks - portfolio_history) / peaks
        if np.any(drawdowns > 0):
            max_drawdown_pct = np.max(drawdowns) * 100
    metrics['max_drawdown_pct'] = max_drawdown_pct
    print(f"2. Max Drawdown: {max_drawdown_pct:.2f}%")

    # --- Calculate win rate and other metrics from the trading log ---
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
    print(f"3. Total Trades: {total_trades}")

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
        print(f"4. Win Rate: {win_rate_pct:.2f}%")
        print(f"5. Best Trade: {best_trade_pct:.2f}%")
        print(f"6. Worst Trade: {worst_trade_pct:.2f}%")
    else:
        metrics.update({'win_rate_pct': 0, 'best_trade_pct': 0, 'worst_trade_pct': 0})
        

    print("-" * 25)
    
    return metrics

# --- main ---
A3C_PARAMS = {
    'update_global_iter': 30,
    'gamma': 0.995,
    'max_ep': 2000,  # 500
    'lr': 1e-5,  # 1e-5
    'entropy_beta': 0.05,  # 0.01
    'train_split': 0.8,
    'weight_path': 'QA3C/models/qa3c_model_ep_630.pth',  # Path to checkpoint model to resume training from
}

if __name__ == "__main__":
    # --- Step 1: Load Pre-trained LSTM Model ---
    print("載入預訓練的 QLSTM 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize QLSTM model
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

    # --- Step 2: Prepare Data ---
    print("準備交易資料...")
    data_path = os.path.join(project_root, 'QLSTM', 'USD_TWD_Historical Data.csv')
    
    full_data_df = prepare_trading_data(file_path=data_path, num_rows=10000)

    
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
    print_model_summary(gnet, model_name="A3C Agent")
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

    # --- Step 4: Start Parallel Training ---
    num_workers = min(mp.cpu_count(), 10)
    print(f'[cpu_count] {mp.cpu_count()}')
    print(f"Starting {num_workers} Workers for parallel training...")
    
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
    
    training_start_time = time.time()  
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
    training_total_time = time.time() - training_start_time  
    print(f"\nTraining complete. Total training time: {training_total_time:.2f} seconds") 
    print(f"Total {len(res)} episodes executed") 
    print(f"Average time per episode: {training_total_time/len(res):.2f} seconds")

    # --- Step 5: Save and Plot ---
    print("Saving model and plotting results...")
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
    print("Results saved.")
    
    # --- [Modified] Step 6: Evaluate and Plot Trading Points on Training and Testing Sets ---
    all_results = {}
    print("\n--- Starting evaluation ---")
    
    
    train_results = run_evaluation(
        df=train_df, 
        gnet=gnet, 
        lstm_model=lstm_model, 
        device=device,
        title="Training Data",
        filename="A3C_trade_visual_train.png"
    )
    all_results['training'] = train_results
    
    
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
    
    print(f"\nAll train and test trading results have been successfully saved to: {pickle_filepath}")
    print("\nAll evaluation and plotting complete.")
