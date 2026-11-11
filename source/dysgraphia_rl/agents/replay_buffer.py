"""
Replay Buffer with optional Prioritized Experience Replay (PER).
Matches Design and Solution Section 6: "Double DQN ... prioritized
experience replay" and Section 7 pseudocode: `R.sample_prioritized()`.
"""

from __future__ import annotations
import numpy as np
from collections import deque


class ReplayBuffer:
    """Simple prioritized experience replay buffer.

    Priorities are proportional to |TD-error| + epsilon, raised to alpha.
    Importance-sampling weights use beta, annealed externally by the caller.
    """

    def __init__(self, capacity: int = 50_000, alpha: float = 0.6, eps: float = 1e-3):
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def store(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_prio)

    def sample(self, batch_size: int, beta: float = 0.4):
        n = len(self.buffer)
        if n == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch_size = min(batch_size, n)

        prios = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs = prios / prios.sum()

        indices = np.random.choice(n, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(abs(err)) + self.eps
