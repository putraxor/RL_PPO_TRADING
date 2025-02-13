import os
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# =============================
# 1. Environment Trading
# =============================
class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100_000, fee_rate=0.0001, holding_cost=0.000001, 
                 drawdown_penalty_rate=0.01, min_balance=100, data_index=None, position_size_percentage=0.1, max_candles=15000):
        super(TradingEnv, self).__init__()
        # Reset index data
        self.data_index = data_index
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.max_balance = initial_balance
        self.fee_rate = fee_rate
        self.holding_cost = holding_cost
        self.drawdown_penalty_rate = drawdown_penalty_rate
        self.min_balance = min_balance
        self.position_size_percentage = position_size_percentage
        self.max_candles = max_candles
        self.current_candle = 0

        # Status posisi: 0 = tidak ada posisi, 1 = LONG, -1 = SHORT
        self.current_mode = 0
        self.open_price = 0.0
        self.current_step = 0
        self.current_time = self.data_index[self.current_step] if self.data_index is not None else None

        # Kolom fitur (14 fitur)
        self.feature_cols = ['m5', 'm5_prev', 'm15', 'm15_prev',
                             'h1', 'h1_prev', 'h4', 'h4_prev',
                             'd1', 'd1_prev', 'W1', 'W1_prev',
                             'M1', 'M1_prev']
        # Observation: 14 fitur + 2 tambahan (current_mode, current_log_return)
        obs_low = -np.inf * np.ones(16, dtype=np.float32)
        obs_high = np.inf * np.ones(16, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action Space: [1, -1, 0] dengan mapping:
        #   0 -> LONG, 1 -> SHORT, 2 -> HOLD
        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {0: 1, 1: -1, 2: 0}
        print("Trading Environment initialized.")

    def reset(self):
        self.current_balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.current_mode = 0
        self.open_price = 0.0
        self.current_step = 0
        self.current_candle = 0
        self.current_time = self.data_index[self.current_step] if self.data_index is not None else None

        row = self.data.iloc[self.current_step]
        obs_features = row[self.feature_cols].values.astype(np.float32)
        current_log_return = 0.0
        obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])
        return obs

    def step(self, action_idx):
        self.current_candle += 1
        action = self.action_map[action_idx]
        done = False
        reward = 0.0

        # Ambil harga saat ini
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['open']
        if self.data_index is not None:
            self.current_time = self.data_index[self.current_step]

        # Tentukan ukuran posisi
        position_size = self.initial_balance * self.position_size_percentage

        # Jika balance di bawah minimum, episode selesai
        if self.current_balance < self.min_balance:
            print(f"Balance di bawah minimum ({self.current_balance:,.0f} < {self.min_balance:,.0f}). Mengakhiri episode.")
            done = True
            # reward = -1.0
            info = {"balance": self.current_balance, "max_balance": self.max_balance, "terminated": True, "reason": "balance_too_low"}
            row = self.data.iloc[-1]
            obs_features = row[self.feature_cols].values.astype(np.float32)
            current_log_return = 0.0
            obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])
            return obs, reward, done, info
        elif self.current_candle > self.max_candles:
            print(f"Mencapai batas candle ({self.current_candle} > {self.max_candles}). Mengakhiri episode.")
            done = True
            # reward = -1.0
            info = {"balance": self.current_balance, "max_balance": self.max_balance, "terminated": True, "reason": "max_candles_reached"}
            row = self.data.iloc[-1]
            obs_features = row[self.feature_cols].values.astype(np.float32)
            current_log_return = 0.0
            obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])
            return obs, reward, done, info

        # Jika sudah mencapai akhir data, pastikan posisi ditutup
        if self.current_step >= len(self.data) - 1:
            if self.current_mode != 0:
                if self.current_mode == 1:  # LONG
                    profit_pct = (current_price - self.open_price) / self.open_price
                elif self.current_mode == -1:  # SHORT
                    profit_pct = (self.open_price - current_price) / self.open_price

                trade_profit = position_size * profit_pct
                fee_close = position_size * self.fee_rate
                self.current_balance += trade_profit - fee_close
                reward += (trade_profit - fee_close) / self.initial_balance
                self.current_mode = 0
            done = True
            info = {"balance": self.current_balance, "max_balance": self.max_balance, "terminated": False}
            row = self.data.iloc[-1]
            obs_features = row[self.feature_cols].values.astype(np.float32)
            current_log_return = 0.0
            obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])
            return obs, reward, done, info

        # Jika aksi berbeda dari posisi saat ini (misal, membuka posisi baru atau mengganti posisi)
        if action != 0 and action != self.current_mode:
            # Jika sudah ada posisi, tutup posisi tersebut terlebih dahulu
            if self.current_mode != 0:
                if self.current_mode == 1:
                    profit_pct = (current_price - self.open_price) / self.open_price
                elif self.current_mode == -1:
                    profit_pct = (self.open_price - current_price) / self.open_price

                trade_profit = position_size * profit_pct
                fee_close = position_size * self.fee_rate
                self.current_balance += trade_profit - fee_close
                reward += (trade_profit - fee_close) / self.initial_balance
                self.current_mode = 0

            # Buka posisi baru jika aksi bukan HOLD
            if action != 0:
                fee_open = position_size * self.fee_rate
                # Pastikan balance cukup untuk fee membuka posisi
                if self.current_balance < fee_open or (self.current_balance - fee_open) < self.min_balance:
                    print("Balance tidak cukup untuk membuka posisi baru. Mengakhiri episode.")
                    done = True
                    reward = -1.0
                    info = {"balance": self.current_balance, "max_balance": self.max_balance, "terminated": True}
                    row = self.data.iloc[-1]
                    obs_features = row[self.feature_cols].values.astype(np.float32)
                    current_log_return = 0.0
                    obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])
                    return obs, reward, done, info
                self.current_balance -= fee_open
                reward -= fee_open / self.initial_balance
                self.current_mode = action
                self.open_price = current_price
        else:
            # Jika aksi adalah HOLD (atau memilih aksi yang sama dengan posisi yang sedang berjalan)
            reward = -self.holding_cost

        # Update penalti drawdown
        self.max_balance = max(self.max_balance, self.current_balance)
        if self.max_balance > 0:
            drawdown = (self.max_balance - self.current_balance) / self.max_balance
        else:
            drawdown = 0.0
        reward -= drawdown * self.drawdown_penalty_rate

        # Pindah ke step berikutnya
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        # Ambil observasi berikutnya
        row = self.data.iloc[self.current_step] if not done else self.data.iloc[-1]
        obs_features = row[self.feature_cols].values.astype(np.float32)
        # Jika ada posisi terbuka, hitung current_log_return dengan benar
        if self.current_mode != 0:
            next_price = current_price  # Harga dari step sebelumnya sebagai pembanding
            # Namun, untuk observasi baru, kita gunakan harga dari baris baru
            next_row = self.data.iloc[self.current_step]
            next_price = next_row['open']
            if self.current_mode == 1:
                current_log_return = np.log((next_price + 1e-8) / (self.open_price + 1e-8))
            else:  # SHORT
                current_log_return = np.log((self.open_price + 1e-8) / (next_price + 1e-8))
        else:
            current_log_return = 0.0
        obs = np.concatenate([obs_features, np.array([self.current_mode, current_log_return], dtype=np.float32)])

        info = {"balance": self.current_balance, "max_balance": self.max_balance, "terminated": False}
        return obs, reward, done, info

# =============================
# 2. PPO Agent dengan LSTM
# =============================
class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, action_dim=3):
        super(ActorCritic, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        # LSTM Layer 1
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        # LSTM Layer 2
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        # Head untuk policy dan value
        self.actor = nn.Linear(hidden_dim2, action_dim)
        self.critic = nn.Linear(hidden_dim2, 1)
        print("ActorCritic model initialized.")

    def forward(self, x, hidden1=None, hidden2=None):
        # x: (batch, seq_len, input_dim)
        if hidden1 is None:
            h0_1 = torch.zeros(1, x.size(0), self.hidden_dim1).to(x.device)
            c0_1 = torch.zeros(1, x.size(0), self.hidden_dim1).to(x.device)
        else:
            h0_1, c0_1 = hidden1
        out1, hidden1 = self.lstm1(x, (h0_1, c0_1))

        if hidden2 is None:
            h0_2 = torch.zeros(1, x.size(0), self.hidden_dim2).to(x.device)
            c0_2 = torch.zeros(1, x.size(0), self.hidden_dim2).to(x.device)
        else:
            h0_2, c0_2 = hidden2
        out2, hidden2 = self.lstm2(out1, (h0_2, c0_2))

        # Ambil output dari time step terakhir
        out = out2[:, -1, :]  # (batch, hidden_dim2)
        policy_logits = self.actor(out)
        value = self.critic(out)
        return policy_logits, value, hidden1, hidden2

# Fungsi untuk menghitung advantage menggunakan Generalized Advantage Estimation (GAE)
def compute_gae(rewards, values, masks, gamma=0.99, lam=0.95):
    values = values + [0]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

# Fungsi update PPO
def ppo_update(agent, optimizer, trajectories, clip_epsilon=0.2, epochs=10):
    states = torch.stack(trajectories['states'])
    actions = torch.tensor(trajectories['actions'], dtype=torch.long)
    old_log_probs = torch.stack(trajectories['log_probs']).detach()
    returns = torch.tensor(trajectories['returns'], dtype=torch.float32)
    values = torch.stack(trajectories['values']).detach().squeeze()
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    print("Starting PPO update...")
    for epoch in range(epochs):
        print(f"  Epoch {epoch+1}/{epochs}")
        states_batch = states.unsqueeze(1)  # (T, 1, input_dim)
        policy_logits, value, _, _ = agent(states_batch)
        dist = Categorical(logits=policy_logits)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - value.squeeze()).pow(2).mean()
        entropy_loss = dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("PPO update complete.")

# =============================
# 3. Training Loop
# =============================
def main():
    versi = '0.0.1'
    checkpoint = f"brains/ppo_trading_checkpoint_{versi}.pth"
    print('Memuat data: data/processed.pkl')
    data = pd.read_pickle('data/processed.pkl')
    data = data[data.index.year>=2023]
    data_index = data.index
    print('Data berhasil dimuat.')

    print('Inisialisasi Trading Environment...')
    min_balance = 100
    position_size_percentage = 0.2  # 0.1% dari initial balance
    env = TradingEnv(data, initial_balance=100_000, min_balance=min_balance, data_index=data_index, 
                     position_size_percentage=position_size_percentage)
    input_dim = 16  # 14 fitur + current_mode + current_log_return
    action_dim = 3  # [LONG, SHORT, HOLD]
    print('Inisialisasi Actor-Critic Agent...')
    agent = ActorCritic(input_dim, hidden_dim1=64, hidden_dim2=32, action_dim=action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4)

        # Cek apakah checkpoint tersedia dan muat jika ada
    if os.path.isfile(checkpoint):
        print(f"Memuat checkpoint dari {checkpoint}...")
        agent.load_state_dict(torch.load(checkpoint))
        print("Checkpoint berhasil dimuat.")
    else:
        print("Tidak ada checkpoint yang ditemukan. Memulai training dari awal.")

    num_episodes = 1000
    gamma = 0.99
    lam = 0.95
    clip_epsilon = 0.2
    ppo_epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent.to(device)
    print(f"Using device: {device}")

    print('\nMemulai Training Loop...')
    for episode in range(num_episodes):
        print(f"\nEpisode {episode+1}/{num_episodes} dimulai...")
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        done = False

        trajectories = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'masks': [],
            'values': []
        }
        episode_reward = 0

        while not done:
            state_input = state.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
            policy_logits, value, _, _ = agent(state_input)
            dist = Categorical(logits=policy_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, info = env.step(action.item())
            episode_reward += reward

            trajectories['states'].append(state)
            trajectories['actions'].append(action.item())
            trajectories['log_probs'].append(log_prob)
            trajectories['rewards'].append(torch.tensor(reward, dtype=torch.float32).to(device))
            mask = 0 if done else 1
            trajectories['masks'].append(mask)
            trajectories['values'].append(value.squeeze())

            state = torch.tensor(next_state, dtype=torch.float32).to(device)

            print(f"[{env.current_time}] Step: {env.current_step}, Action Index: {action.item()}, Action: {env.action_map[action.item()]:>2} -> Reward: {reward:.4f}, Balance: {info['balance']:,.0f}, Max Balance: {info['max_balance']:,.0f}")

            if info.get("terminated", False):
                print("Episode diakhiri karena", info.get('reason', 'unknown_reason'))
                break

        rewards = [r.item() for r in trajectories['rewards']]
        values = [v.item() for v in trajectories['values']]
        masks = trajectories['masks']
        returns = compute_gae(rewards, values, masks, gamma, lam)
        trajectories['returns'] = returns

        if len(trajectories['states']) > 1:
            ppo_update(agent, optimizer, trajectories, clip_epsilon, ppo_epochs)
        else:
            print("Episode terlalu pendek untuk melakukan PPO update. Melewati update.")
            continue

        print(f"Episode {episode+1:4d} | Reward: {episode_reward:.4f} | Balance: {info['balance']:.2f}")

        if (episode + 1) % 100 == 0:
            print(f"Menyimpan checkpoint model episode {episode+1}...")
            # torch.save(agent.state_dict(), f"ppo_trading_checkpoint_{episode+1}.pth")
            torch.save(agent.state_dict(), checkpoint)
            print(f"Checkpoint episode {episode+1} berhasil disimpan.")

    print('\nTraining Loop selesai.')

if __name__ == '__main__':
    main()
