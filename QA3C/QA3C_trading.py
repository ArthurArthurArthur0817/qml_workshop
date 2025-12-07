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

# --- Project root path and module imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import data processing functions and QLSTM classes
from QLSTM.QLSTM_trading_final import (
    normalize_sequences,
    CustomQLSTMCell,
    CustomLSTM,
)

# Import A3C helper utilities
from QA3C.utils import v_wrap, set_init, push_and_pull, record
from QA3C.plot_functions import full_plotting
from QA3C.shared_adam import SharedAdam

os.environ["OMP_NUM_THREADS"] = "1"


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print a summary of a PyTorch model, listing trainable parameters per layer
    and the total number of parameters.
    """
    print("=" * 70)
    print(f"{model_name} Parameters Summary")
    print("-" * 70)
    print(f"{'Layer Name':<35} {'Shape':<20} {'#Params':>12}")
    print("-" * 70)

    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            shape_str = str(list(param.shape))
            print(f"{name:<35} {shape_str:<20} {num_params:>12,}")

    print("-" * 70)
    print(f"Total Trainable Parameters: {total_params:>15,}")
    print("=" * 70)


# ===== Quantum Layer Functions for VQC =====
def H_layer(nqubits: int):
    """Layer of single-qubit Hadamard gates."""
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def RY_layer(w):
    """Layer of parametrized qubit rotations around the Y axis."""
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)


def entangling_layer(nqubits: int):
    """Layer of CNOTs in an even-odd pattern."""
    for i in range(0, nqubits - 1, 2):  # even indices: i=0,2,...,N-2
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):  # odd indices: i=1,3,...,N-3
        qml.CNOT(wires=[i, i + 1])


def q_function(x, q_weights, n_class: int):
    """Variational quantum circuit used in the actor/critic networks."""
    n_dep = q_weights.shape[0]
    n_qub = q_weights.shape[1]

    H_layer(n_qub)

    # Feature embedding
    RY_layer(x)

    # Trainable variational layers
    for k in range(n_dep):
        entangling_layer(n_qub)
        RY_layer(q_weights[k])

    # Expectation values in the Z basis
    exp_vals = [qml.expval(qml.PauliZ(position)) for position in range(n_class)]
    return exp_vals


class TorchVQC(nn.Module):
    """
    Quantum Variational Circuit wrapped as a PyTorch module.
    Used inside the QA3C actor and critic networks.
    """

    def __init__(self, vqc_depth: int, n_qubits: int, n_class: int):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(vqc_depth, n_qubits))
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.VQC = qml.QNode(q_function, self.dev, interface="torch")
        self.n_class = n_class

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X: shape (batch_size, n_qubits)
        Returns: (batch_size, n_class)
        """
        y_preds = torch.stack(
            [torch.stack(self.VQC(x, self.weights, self.n_class)).float() for x in X]
        )
        return y_preds


# QLSTM hyperparameters (must be consistent with QLSTM_trading_final.py)
qlstm_params = {
    "feature_columns": ["open", "high", "low", "close", "ma5", "ma10"],
    "sequence_length": 4,
    "input_size": 6,
    "hidden_size": 2,
    "output_size": 2,
    "qnn_depth": 1,
}


def load_qlstm_model(model_path: str, device: torch.device):
    """
    Load the pretrained QLSTM model on the given device.
    The model architecture must match QLSTM_trading_final.py.
    """
    qlstm_cell = CustomQLSTMCell(
        input_size=qlstm_params["input_size"],
        hidden_size=qlstm_params["hidden_size"],
        output_size=qlstm_params["output_size"],
        vqc_depth=qlstm_params["qnn_depth"],
    ).float().to(device)

    lstm_model = CustomLSTM(
        input_size=qlstm_params["input_size"],
        hidden_size=qlstm_params["hidden_size"],
        lstm_cell_QT=qlstm_cell,
    ).float().to(device)

    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()
    return lstm_model


def prepare_trading_data(file_path: str, num_rows: int = 10000) -> pd.DataFrame:
    """
    Prepare price data for the trading environment.
    Computes MA5 and MA10 and removes NaN rows.
    Also validates that there are no zero 'close' prices.
    """
    df = pd.read_csv(file_path)
    df = df[::-1].reset_index(drop=True)
    print(df.head())

    df["ma5"] = df["close"].rolling(window=5).mean()
    df["ma10"] = df["close"].rolling(window=10).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    zero_price_rows = df[df["close"] == 0]
    if not zero_price_rows.empty:
        print("Error: Found rows with 'close' price equal to 0:")
        print(zero_price_rows)
        raise ValueError(
            "Data validation failed: 'close' price should not be 0. Please check the raw data file."
        )

    if df.empty:
        raise ValueError(
            "Error: No valid data left after cleaning. Please check the raw data file."
        )

    return df


class TradingEnv:
    """
    Trading environment used by the QA3C agent.

    State representation:
        - 2-dimensional QLSTM softmax output (up/down).
        - 8 additional engineered trading features.

    Action space:
        0: Hold
        1: Buy (long, up to trade_amount_per_time per trade)
        2: Sell (close all open positions)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lstm_model: nn.Module,
        device: torch.device,
        time_penalty: float = 0.02,
    ):
        self.df = df.copy()
        self.lstm_model = lstm_model
        self.device = device
        self.sequence_length = qlstm_params["sequence_length"]
        self.feature_columns = qlstm_params["feature_columns"]
        self.time_penalty = time_penalty
        self.volatility_window = 20  # window size for volatility calculation

        # Precompute additional moving averages
        for window in [5, 20, 60]:
            if f"ma{window}" not in self.df.columns:
                self.df[f"ma{window}"] = self.df["close"].rolling(window=window).mean()
        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        # Observation space: QLSTM output (2) + 8 engineered features
        self.observation_space_shape = (qlstm_params["output_size"] + 8,)
        self.action_space_n = 3

        self.initial_cash = 50000.0
        self.trade_amount_per_time = 45000.0
        self.reset()

    def _get_state(self) -> torch.Tensor:
        """
        Build the current state:
            - QLSTM softmax output (2-dim)
            - 8 additional normalized trading features
        """
        # 1. QLSTM output (market regime forecast)
        start = self.current_step - self.sequence_length + 1
        end = self.current_step + 1
        sequence_df = self.df.iloc[start:end]

        x = torch.tensor(
            sequence_df[self.feature_columns].values, dtype=torch.float32
        ).unsqueeze(0)
        x_normalized = normalize_sequences(x).to(self.device)

        with torch.no_grad():
            outputs, _ = self.lstm_model(x_normalized)
            logits = outputs[:, -1, :]  # last timestep output
            lstm_state = F.softmax(logits, dim=1).squeeze(0).cpu()

        # 2. Additional 8 engineered features
        current_price = self.df["close"].iloc[self.current_step]
        ma5 = self.df["ma5"].iloc[self.current_step]
        ma20 = self.df["ma20"].iloc[self.current_step]
        ma60 = self.df["ma60"].iloc[self.current_step]

        # Feature 1: cash ratio
        cash_ratio = self.cash / self.initial_cash

        # Feature 2: holdings value ratio
        holdings_value = sum(self.active_trades_shares) * current_price
        holdings_ratio = holdings_value / self.initial_cash

        # Feature 3 & 4: unrealized PnL (%) and price-cost ratio
        unrealized_pnl_pct = 0.0
        avg_price_ratio = 0.0
        if sum(self.active_trades_shares) > 0:
            avg_buy_price = np.average(
                self.active_trades_buy_prices, weights=self.active_trades_shares
            )
            if avg_buy_price > 0:
                unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if current_price > 0:
                avg_price_ratio = (current_price - avg_buy_price) / current_price

        # Feature 5 & 6: price deviation vs MA20 and MA60
        price_ma20_ratio = (current_price - ma20) / ma20 if ma20 > 0 else 0.0
        price_ma60_ratio = (current_price - ma60) / ma60 if ma60 > 0 else 0.0

        # Feature 7: MA5 vs MA20 deviation
        ma5_ma20_ratio = (ma5 - ma20) / ma20 if ma20 > 0 else 0.0

        # Feature 8: recent price volatility (20-day)
        if self.current_step >= self.volatility_window:
            recent_prices = self.df["close"].iloc[
                self.current_step - self.volatility_window : self.current_step
            ]
            mean_price = np.clip(np.mean(recent_prices), 1e-9, np.inf)
            price_volatility = np.std(recent_prices) / mean_price
        else:
            price_volatility = 0.0

        additional_state = torch.tensor(
            [
                cash_ratio,
                holdings_ratio,
                np.clip(unrealized_pnl_pct, -1.0, 1.0),
                np.clip(avg_price_ratio, -1.0, 1.0),
                np.clip(price_ma20_ratio, -0.2, 0.2),
                np.clip(price_ma60_ratio, -0.2, 0.2),
                np.clip(ma5_ma20_ratio, -0.1, 0.1),
                np.clip(price_volatility, 0.0, 1.0),
            ],
            dtype=torch.float32,
        )

        return torch.cat((lstm_state, additional_state))

    def _reset_trade_state(self):
        self.active_trades_buy_prices = []
        self.active_trades_shares = []

    def reset(self) -> torch.Tensor:
        """
        Reset the environment.
        Start from step 60 so that all moving averages are valid.
        """
        self.current_step = 60
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.portfolio_history = [self.initial_cash] * self.current_step
        self.historical_trades = []
        self._reset_trade_state()
        return self._get_state()

    def step(self, action: int):
        """
        Take one step in the environment.

        Returns:
            next_state, reward, done, truncated, info
        """
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        current_price = self.df["close"].iloc[self.current_step]
        info = {"action_taken": "hold"}

        # Base reward: time penalty to encourage efficient trading
        reward = -self.time_penalty

        ma5 = self.df["ma5"].iloc[self.current_step]
        ma20 = self.df["ma20"].iloc[self.current_step]
        is_uptrend = ma5 > ma20

        # Action 1: Buy
        if action == 1:
            if self.cash >= self.trade_amount_per_time and is_uptrend:
                shares_bought = self.trade_amount_per_time / current_price
                self.cash -= self.trade_amount_per_time
                self.active_trades_buy_prices.append(current_price)
                self.active_trades_shares.append(shares_bought)
                info["action_taken"] = "buy"
                reward += 0.5  # immediate reward for trend-following entry
            elif self.cash >= self.trade_amount_per_time and not is_uptrend:
                reward -= 2.0  # penalty for counter-trend buy
            else:
                info["action_taken"] = "invalid_buy"
                reward -= 0.5  # penalty for invalid buy

        # Action 2: Sell (close all positions)
        elif action == 2:
            total_shares = sum(self.active_trades_shares)
            if total_shares > 0:
                avg_buy_price = np.average(
                    self.active_trades_buy_prices, weights=self.active_trades_shares
                )
                realized_profit = (current_price - avg_buy_price) * total_shares
                cost_basis = avg_buy_price * total_shares
                pnl_pct = realized_profit / cost_basis if cost_basis > 0 else 0.0

                if realized_profit > 0:
                    reward += 10.0 + pnl_pct * 50.0
                else:
                    reward -= 2.0 + pnl_pct * 10.0

                self.cash += total_shares * current_price
                info["action_taken"] = "sell"
                self._reset_trade_state()
            else:
                info["action_taken"] = "invalid_sell"
                reward -= 0.5  # penalty for selling with no position

        # Penalty for unrealized losses when holding positions
        total_shares = sum(self.active_trades_shares)
        if total_shares > 0 and info["action_taken"] != "sell":
            avg_buy_price = np.average(
                self.active_trades_buy_prices, weights=self.active_trades_shares
            )
            unrealized_pnl_pct = (current_price - avg_buy_price) / avg_buy_price
            if unrealized_pnl_pct < 0:
                reward += unrealized_pnl_pct * abs(unrealized_pnl_pct) * 5.0

        current_holdings_value = sum(self.active_trades_shares) * current_price
        self.portfolio_value = self.cash + current_holdings_value
        self.portfolio_history.append(self.portfolio_value)

        final_reward = np.clip(reward, -15.0, 30.0)
        next_state = self._get_state()

        if info["action_taken"] == "sell":
            self.historical_trades.append({"profit": realized_profit})

        return next_state, final_reward, done, False, info


# --- QA3C Network and Worker definition ---
class Net(nn.Module):
    def __init__(self, s_dim: int, a_dim: int):
        super(Net, self).__init__()

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

    def forward(self, x: torch.Tensor):
        # Actor
        pi1 = torch.tanh(self.pi1(x))
        pi1 = torch.tanh(self.pi_vqc(pi1))
        logits = self.pi2(pi1)

        # Critic
        v1 = torch.tanh(self.v1(x))
        v1 = torch.tanh(self.v_vqc(v1))
        values = self.v2(v1)

        return logits, values

    def choose_action(self, s: torch.Tensor) -> int:
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        m = self.distribution(prob)
        return m.sample().numpy()[0]

    def choose_best_action(self, s: torch.Tensor) -> int:
        """
        Greedy action selection used for evaluation.
        """
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1)
        return torch.argmax(prob, dim=1).item()

    def loss_func(self, s: torch.Tensor, a: torch.Tensor, v_t: torch.Tensor):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values
        c_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        entropy = m.entropy()

        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss - A3C_PARAMS["entropy_beta"] * entropy).mean()
        return total_loss


class Worker(mp.Process):
    def __init__(
        self,
        gnet: Net,
        opt: torch.optim.Optimizer,
        global_ep,
        global_ep_r,
        res_queue,
        name: int,
        df: pd.DataFrame,
        model_path: str,
        device: torch.device,
    ):
        super(Worker, self).__init__()
        self.name = f"w{name:02d}"
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt

        self.df = df
        self.model_path = model_path
        self.device = device

        # Observation space: 2 (QLSTM) + 8 engineered features
        N_S = qlstm_params["output_size"] + 8
        N_A = 3
        self.lnet = Net(N_S, N_A)

    def run(self):
        # Each worker loads its own copy of the QLSTM model on CPU
        lstm_model = load_qlstm_model(self.model_path, self.device)
        self.env = TradingEnv(self.df, lstm_model, self.device)

        total_step = 1
        while self.g_ep.value < A3C_PARAMS["max_ep"]:
            episode_start_time = time.time()
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.0
            episode_steps = 0

            while True:
                episode_steps += 1
                a = self.lnet.choose_action(v_wrap(s.unsqueeze(0)))
                s_, r, done, _, _ = self.env.step(a)

                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % A3C_PARAMS["update_global_iter"] == 0 or done:
                    push_and_pull(
                        self.opt,
                        self.lnet,
                        self.gnet,
                        done,
                        s_,
                        buffer_s,
                        buffer_a,
                        buffer_r,
                        A3C_PARAMS["gamma"],
                    )
                    buffer_s, buffer_a, buffer_r = [], [], []

                if done:
                    episode_time = time.time() - episode_start_time
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                    print(
                        f"{self.name} | Episode {self.g_ep.value} | "
                        f"Time: {episode_time:.2f}s | Steps: {episode_steps} | "
                        f"Reward: {ep_r:.2f}"
                    )
                    break

                s = s_
                total_step += 1

        self.res_queue.put(None)


def plot_trade_history(
    df: pd.DataFrame,
    trade_log,
    portfolio_history,
    initial_cash: float,
    filename: str,
    title: str,
):
    """
    Plot closing prices with buy/sell markers and cumulative PnL.
    """
    plt.style.use("seaborn-v0_8-darkgrid")

    # Price + trade signals + PnL subplot
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 15), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"Agent Trading Performance on {title}", fontsize=22)

    # Upper subplot: price and trade signals
    ax1.plot(
        df.index,
        df["close"],
        label="Close Price",
        color="dodgerblue",
        alpha=0.8,
        linewidth=1.5,
    )

    buy_indices = [item["step"] for item in trade_log if item["action"] == "buy"]
    buy_prices = [item["price"] for item in trade_log if item["action"] == "buy"]

    sell_indices = [item["step"] for item in trade_log if item["action"] == "sell"]
    sell_prices = [item["price"] for item in trade_log if item["action"] == "sell"]

    if buy_indices:
        ax1.scatter(
            buy_indices,
            buy_prices,
            marker="^",
            color="lime",
            s=120,
            label="Buy Signal",
            edgecolors="black",
            zorder=5,
        )
    if sell_indices:
        ax1.scatter(
            sell_indices,
            sell_prices,
            marker="v",
            color="red",
            s=120,
            label="Sell Signal",
            edgecolors="black",
            zorder=5,
        )

    ax1.set_ylabel("Price", fontsize=15)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.tick_params(axis="y", labelsize=12)

    # Lower subplot: cumulative PnL
    if portfolio_history:
        portfolio_values = np.array(portfolio_history)
        cumulative_pnl = portfolio_values - initial_cash

        ax2.plot(
            df.index, cumulative_pnl, label="Cumulative PnL", color="purple", linewidth=2
        )

        ax2.fill_between(
            df.index,
            cumulative_pnl,
            where=(cumulative_pnl >= 0),
            color="green",
            alpha=0.3,
            interpolate=True,
        )
        ax2.fill_between(
            df.index,
            cumulative_pnl,
            where=(cumulative_pnl < 0),
            color="red",
            alpha=0.3,
            interpolate=True,
        )

        ax2.axhline(0, color="grey", linestyle="--", linewidth=1)
        ax2.set_ylabel("Cumulative PnL", fontsize=15)
        ax2.grid(True)

    ax2.set_xlabel(f"Time Steps in {title}", fontsize=15)
    ax2.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.1)
    plt.savefig(filename)
    plt.show()
    plt.close()
    print(f"Trading visualization saved to: {filename}")

    # Separate PnL-only figure
    if portfolio_history:
        fig_pnl, ax_pnl = plt.subplots(1, 1, figsize=(12, 8))
        fig_pnl.suptitle(f"Agent Trading Performance on {title}", fontsize=24)

        portfolio_values = np.array(portfolio_history)
        cumulative_pnl = portfolio_values - initial_cash

        ax_pnl.plot(df.index, cumulative_pnl, color="purple", linewidth=2)

        ax_pnl.fill_between(
            df.index,
            cumulative_pnl,
            where=(cumulative_pnl >= 0),
            color="green",
            alpha=0.3,
            interpolate=True,
        )
        ax_pnl.fill_between(
            df.index,
            cumulative_pnl,
            where=(cumulative_pnl < 0),
            color="red",
            alpha=0.3,
            interpolate=True,
        )

        ax_pnl.axhline(0, color="grey", linestyle="--", linewidth=1)

        ax_pnl.set_xlabel(f"Time Steps in {title}", fontsize=22)
        ax_pnl.set_ylabel("Cumulative PnL", fontsize=22)
        ax_pnl.grid(True, alpha=0.3)
        ax_pnl.tick_params(axis="both", labelsize=20)

        plt.tight_layout()
        plt.show()
        plt.close()


def calculate_and_print_metrics(
    eval_env: TradingEnv, title: str, trade_log_for_plot
):
    """
    Compute, print and return a dictionary of evaluation metrics.
    """
    print(f"\n--- {title} Performance Metrics ---")
    metrics = {}

    # 1. Total return
    initial_value = eval_env.initial_cash
    final_value = eval_env.portfolio_value
    total_return_pct = ((final_value - initial_value) / initial_value) * 100.0
    metrics["total_return_pct"] = total_return_pct
    print(f"1. Total Return: {total_return_pct:.2f}%")

    # 2. Maximum drawdown
    portfolio_history = np.array(eval_env.portfolio_history)
    max_drawdown_pct = 0.0
    if len(portfolio_history) >= 2:
        peaks = np.maximum.accumulate(portfolio_history)
        drawdowns = (peaks - portfolio_history) / peaks
        if np.any(drawdowns > 0):
            max_drawdown_pct = np.max(drawdowns) * 100.0
    metrics["max_drawdown_pct"] = max_drawdown_pct
    print(f"2. Max Drawdown: {max_drawdown_pct:.2f}%")

    # From trade log, compute trade-level stats
    completed_trades = []
    active_buy_prices = []
    trade_amount = eval_env.trade_amount_per_time

    for trade in trade_log_for_plot:
        if trade["action"] == "buy":
            active_buy_prices.append(trade["price"])
        elif trade["action"] == "sell" and len(active_buy_prices) > 0:
            sell_price = trade["price"]
            avg_buy_price = np.mean(active_buy_prices)
            cost_basis = len(active_buy_prices) * trade_amount
            if cost_basis > 0:
                total_shares_bought = sum(
                    [trade_amount / p for p in active_buy_prices if p > 0]
                )
                profit = (sell_price - avg_buy_price) * total_shares_bought
                trade_return = (profit / cost_basis) * 100.0
                completed_trades.append({"profit": profit, "return_pct": trade_return})
            active_buy_prices = []

    total_trades = len(completed_trades)
    metrics["total_trades"] = total_trades
    print(f"3. Total Trades: {total_trades}")

    if total_trades > 0:
        winning_trades = sum(1 for t in completed_trades if t["profit"] > 0)
        win_rate_pct = (winning_trades / total_trades) * 100.0
        trade_returns = [t["return_pct"] for t in completed_trades]
        best_trade_pct = max(trade_returns)
        worst_trade_pct = min(trade_returns)
        metrics.update(
            {
                "win_rate_pct": win_rate_pct,
                "best_trade_pct": best_trade_pct,
                "worst_trade_pct": worst_trade_pct,
            }
        )
        print(f"4. Win Rate: {win_rate_pct:.2f}%")
        print(f"5. Best Trade: {best_trade_pct:.2f}%")
        print(f"6. Worst Trade: {worst_trade_pct:.2f}%")
    else:
        metrics.update(
            {"win_rate_pct": 0.0, "best_trade_pct": 0.0, "worst_trade_pct": 0.0}
        )

    print("-" * 25)
    return metrics


def run_evaluation(
    df: pd.DataFrame,
    gnet: Net,
    lstm_model: nn.Module,
    device: torch.device,
    title: str,
    filename: str,
):
    """
    Run evaluation with a greedy policy and plot the trade history.
    Returns a dictionary with all evaluation results.
    """
    print(f"\nRunning evaluation on {title}...")
    eval_df = df.reset_index(drop=True)
    eval_env = TradingEnv(eval_df, lstm_model, device)

    s = eval_env.reset()
    done = False
    trade_log_for_plot = []

    while not done:
        a = gnet.choose_best_action(v_wrap(s.unsqueeze(0)))
        s_next, r, done, _, info = eval_env.step(a)

        action_taken = info.get("action_taken", "hold")
        if action_taken in ["buy", "sell"]:
            price = eval_df["close"].iloc[eval_env.current_step - 1]
            trade_log_for_plot.append(
                {"step": eval_env.current_step - 1, "action": action_taken, "price": price}
            )
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
        title=title,
    )

    performance_metrics = calculate_and_print_metrics(
        eval_env, title, trade_log_for_plot
    )
    print(f"{title} evaluation and plotting completed.")

    results = {
        "trade_log": trade_log_for_plot,
        "portfolio_history": portfolio_history,
        "initial_cash": eval_env.initial_cash,
        "final_portfolio_value": eval_env.portfolio_value,
        "performance_metrics": performance_metrics,
    }
    return results


# --- A3C Hyperparameters ---
A3C_PARAMS = {
    "update_global_iter": 30,
    "gamma": 0.995,
    "max_ep": 500, # 2000
    "lr": 1e-5,
    "entropy_beta": 0.05,
    "train_split": 0.8,
    "weight_path": None,  # checkpoint path to resume training from
}


if __name__ == "__main__":
    # Force CPU usage only
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load pretrained QLSTM model
    print("Loading pretrained QLSTM model...")
    qlstm_cell = CustomQLSTMCell(
        input_size=qlstm_params["input_size"],
        hidden_size=qlstm_params["hidden_size"],
        output_size=qlstm_params["output_size"],
        vqc_depth=qlstm_params["qnn_depth"],
    ).float().to(device)

    lstm_model = CustomLSTM(
        input_size=qlstm_params["input_size"],
        hidden_size=qlstm_params["hidden_size"],
        lstm_cell_QT=qlstm_cell,
    ).float().to(device)

    model_path = os.path.join(
        project_root, "QLSTM", "models", "qlstm_model_epochs_50.pth"
    )
    lstm_model.load_state_dict(torch.load(model_path, map_location=device))
    lstm_model.eval()
    print("QLSTM model loaded.")

    # Step 2: Prepare data
    print("Preparing trading data...")
    data_path = os.path.join(project_root, "QLSTM", "USD_TWD_Historical Data.csv")
    full_data_df = prepare_trading_data(file_path=data_path, num_rows=10000)

    zero_price_rows = full_data_df[full_data_df["close"] == 0]
    if not zero_price_rows.empty:
        print("Error: Found rows with 'close' price equal to 0 in prepared data:")
        print(zero_price_rows)
        raise ValueError(
            "Data validation failed after preparation: 'close' price should not be 0."
        )

    if full_data_df.empty:
        raise ValueError(
            "Error: No valid data left after preparation. Please check the raw data file."
        )

    split_point = int(len(full_data_df) * A3C_PARAMS["train_split"])
    train_df = full_data_df[:split_point]
    print(f"Data prepared. Number of training samples: {len(train_df)}")
    print("*** Training data preview ***")
    print(train_df.head(2), train_df.tail(2))

    # Step 3: Initialize QA3C
    dummy_env = TradingEnv(train_df, lstm_model, device)
    N_S = dummy_env.observation_space_shape[0]
    N_A = dummy_env.action_space_n
    del dummy_env

    gnet = Net(N_S, N_A)
    print_model_summary(gnet, model_name="QA3C Agent")
    gnet.share_memory()

    # Load checkpoint if provided
    if A3C_PARAMS["weight_path"] is not None:
        if not os.path.isabs(A3C_PARAMS["weight_path"]):
            checkpoint_path = os.path.join(project_root, A3C_PARAMS["weight_path"])
        else:
            checkpoint_path = A3C_PARAMS["weight_path"]

        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                gnet.load_state_dict(checkpoint["model_state_dict"])
                start_episode = checkpoint.get("episode", 0)
                print(f"Resumed from episode {start_episode}")
            else:
                gnet.load_state_dict(checkpoint)
                start_episode = 0
                print("Loaded checkpoint (state_dict only, starting from episode 0)")
        else:
            start_episode = 0
            print(f"Warning: checkpoint path '{checkpoint_path}' not found, starting fresh.")
    else:
        start_episode = 0

    opt = SharedAdam(gnet.parameters(), lr=A3C_PARAMS["lr"], betas=(0.92, 0.999))
    global_ep = mp.Value("i", start_episode)
    global_ep_r = mp.Value("d", 0.0)
    res_queue = mp.Queue()

    # Step 4: Start parallel training
    num_workers = min(mp.cpu_count(), 10)
    print(f"[cpu_count] {mp.cpu_count()}")
    print(f"Starting {num_workers} workers for parallel training...")

    models_dir = os.path.join(project_root, "QA3C", "models")
    os.makedirs(models_dir, exist_ok=True)

    def save_checkpoint_periodically(gnet, global_ep, interval: int = 14):
        last_saved_ep = 0
        while global_ep.value < A3C_PARAMS["max_ep"]:
            current_ep = global_ep.value
            if current_ep > 0 and current_ep % interval == 0 and current_ep > last_saved_ep:
                checkpoint = {
                    "model_state_dict": gnet.state_dict(),
                    "episode": current_ep,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                }
                checkpoint_path = os.path.join(
                    models_dir, f"qa3c_model_ep_{current_ep}.pth"
                )
                torch.save(checkpoint, checkpoint_path)
                print(
                    f"\n[Checkpoint] Saved model at episode {current_ep} to {checkpoint_path}"
                )
                last_saved_ep = current_ep
            time.sleep(2)

    checkpoint_thread = threading.Thread(
        target=save_checkpoint_periodically, args=(gnet, global_ep), daemon=True
    )
    checkpoint_thread.start()

    training_start_time = time.time()
    workers = [
        Worker(
            gnet, opt, global_ep, global_ep_r, res_queue, i, train_df, model_path, device
        )
        for i in range(num_workers)
    ]
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
    print(f"\nTraining finished. Total training time: {training_total_time:.2f} seconds")
    print(f"Total episodes: {len(res)}")
    if len(res) > 0:
        print(
            f"Average episode time: {training_total_time / len(res):.2f} seconds per episode"
        )

    # Step 5: Save model and plot training rewards
    print("Saving model and plotting training rewards...")
    full_plotting(
        _fileTitle="A3C_Trading_Agent", _trainingLength=len(res), _currentRewardList=res
    )

    final_episode = global_ep.value
    final_checkpoint = {
        "model_state_dict": gnet.state_dict(),
        "episode": final_episode,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }

    final_model_path = os.path.join(
        models_dir, f"qa3c_model_ep_{final_episode}_final.pth"
    )
    torch.save(final_checkpoint, final_model_path)
    print(f"Final model saved to: {final_model_path}")

    torch.save(gnet.state_dict(), "A3C_trading_model.pth")
    print("Model weights (state_dict) saved as: A3C_trading_model.pth")

    # Step 6: Evaluate on training and testing sets
    print("\n--- Starting evaluation ---")

    all_results = {}

    train_results = run_evaluation(
        df=train_df,
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Training Data",
        filename="A3C_trade_visual_train.png",
    )
    all_results["training"] = train_results

    test_df = full_data_df[split_point:]
    print("*** Testing data preview ***")
    print(test_df.head(2), test_df.tail(2))

    test_results = run_evaluation(
        df=test_df,
        gnet=gnet,
        lstm_model=lstm_model,
        device=device,
        title="Testing Data",
        filename="A3C_trade_visual_test.png",
    )
    all_results["testing"] = test_results

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    pickle_filepath = os.path.join(results_dir, "all_trading_results.pkl")

    with open(pickle_filepath, "wb") as f:
        pickle.dump(all_results, f)

    print(f"\nAll training and testing trading results saved to: {pickle_filepath}")
    print("\nAll evaluations and plotting completed.")

 

