"""
Double DQN Agent
=================
Implements the RL Agent / Policy Module from the Design and Solution doc
(Section 4, 6, 7):

  - Network: state (R^12) -> FC(128, ReLU) -> FC(64, ReLU) -> 10 Q-values
  - Separate online + target networks, target synced every C steps
  - Double DQN update rule (decouples action *selection* from *evaluation*
    to reduce Q-value overestimation bias -- see Section 7 justification)
  - Works with the prioritized ReplayBuffer in replay_buffer.py

This is intentionally framework-light (pure PyTorch) so it is easy to
demo/read during the code review, while still matching the architecture
and loss function specified in the Design and Solution document exactly.
"""

from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    """FC(128, ReLU) -> FC(64, ReLU) -> Linear(n_actions), as specified in
    Design and Solution Section 6 (Machine Learning Model Architecture)."""

    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int = 12,
        n_actions: int = 10,
        lr: float = 1e-3,
        gamma: float = 0.95,
        buffer_capacity: int = 50_000,
        target_sync_every: int = 200,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync_every = target_sync_every
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.online_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net = QNetwork(state_dim, n_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self._train_steps = 0

    # ------------------------------------------------------------------
    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """epsilon-greedy action selection (Section 7: eps_greedy(Q_theta, s, eps))."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(s)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.store(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    def train_step(self, batch_size: int = 64, beta: float = 0.4) -> float | None:
        """One gradient update using the Double DQN loss:

            y = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))
            L = (y - Q_online(s, a))^2

        Returns the loss value, or None if the buffer is not warmed up yet.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, is_weights = \
            self.replay_buffer.sample(batch_size, beta=beta)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device)
        weights_t = torch.as_tensor(is_weights, device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            # Double DQN: select action with ONLINE net, evaluate with TARGET net
            next_actions = torch.argmax(self.online_net(next_states_t), dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            targets = rewards_t + self.gamma * next_q_target * (1.0 - dones_t)

        td_errors = (targets - q_values).detach().cpu().numpy()
        loss = (weights_t * (targets - q_values).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.replay_buffer.update_priorities(indices, td_errors)

        self._train_steps += 1
        if self._train_steps % self.target_sync_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "train_steps": self._train_steps,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self._train_steps = checkpoint.get("train_steps", 0)
