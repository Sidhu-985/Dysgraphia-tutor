"""
Baseline agents used for comparison against the Double DQN policy, as
required by Objective 5 / Section 7 ("Testing") of the project docs:
  - Random policy
  - Rule-based policy (fixed heuristic, e.g. hint after N wrong attempts)
  - Contextual Bandit (ignores sequential state transitions -- cold-start
    / sanity-check baseline per Section 7 "Agent choice and justification")
"""

from __future__ import annotations
import random
import numpy as np

from envs.dysgraphia_tutor_env import (
    DECREASE_DIFFICULTY, INCREASE_DIFFICULTY, SHOW_HINT,
    SUGGEST_BREAK, CONTINUE_NO_CHANGE, N_ACTIONS,
)
# Note: RuleBasedAgent below only ever selects a subset of the 10 actions
# (decrease/increase difficulty, hint, break, no-change) -- the remaining
# actions (enlarge template, replay animation, switch mode, repeat letter)
# are legitimate agent choices in the environment but are not part of this
# particular fixed heuristic, so they are not imported here.

# state vector indices, matches envs/dysgraphia_tutor_env.py _build_state()
IDX_PREV_SCORE = 0
IDX_RETRIES = 6
IDX_FATIGUE = 7
IDX_FRUSTRATION = 10


class AlwaysBreakAgent:
    """Degenerate reference policy: always requests SUGGEST_BREAK. Added
    after the Phase A audit showed the original reward let this policy
    score ~49/episode (more than most baselines) with zero quality gain.
    Kept as a permanent reward-hacking canary: any reward function under
    which this policy looks good is broken."""
    name = "always_break"

    def act(self, state: np.ndarray) -> int:
        return SUGGEST_BREAK


class DoNothingAgent:
    """'Do-nothing' tutor: never intervenes (always CONTINUE_NO_CHANGE),
    i.e. the child just keeps practising the same exercise. This is the
    counterfactual that any adaptive policy's contribution must be measured
    against."""
    name = "do_nothing"

    def act(self, state: np.ndarray) -> int:
        return CONTINUE_NO_CHANGE


class RandomAgent:
    """Uniform-random action selection. Lower bound on performance."""
    name = "random"

    def __init__(self, seed: int | None = None):
        # seed=None keeps the original behaviour (Python's global `random`
        # stream); a seed gives the agent its own reproducible stream.
        self._rng = random if seed is None else random.Random(seed)

    def act(self, state: np.ndarray) -> int:
        return self._rng.randrange(N_ACTIONS)


class RuleBasedAgent:
    """Simple, brittle, hand-written heuristic -- the kind of system this
    project explicitly argues against in the Problem Statement:
        - suggest a break if fatigue is high
        - show a hint after 3 consecutive wrong attempts (retries proxy)
        - increase difficulty after sustained high performance
        - decrease difficulty after sustained low performance
        - otherwise continue with no change
    """
    name = "rule_based"

    def __init__(self, retry_threshold: float = 0.6, fatigue_threshold: float = 0.75,
                 high_score: float = 0.75, low_score: float = 0.3):
        self.retry_threshold = retry_threshold
        self.fatigue_threshold = fatigue_threshold
        self.high_score = high_score
        self.low_score = low_score

    def act(self, state: np.ndarray) -> int:
        fatigue = state[IDX_FATIGUE]
        retries = state[IDX_RETRIES]
        prev_score = state[IDX_PREV_SCORE]

        if fatigue >= self.fatigue_threshold:
            return SUGGEST_BREAK
        if retries >= self.retry_threshold:
            return SHOW_HINT
        if prev_score >= self.high_score:
            return INCREASE_DIFFICULTY
        if prev_score <= self.low_score:
            return DECREASE_DIFFICULTY
        return CONTINUE_NO_CHANGE


class ContextualBanditAgent:
    """Linear contextual bandit (epsilon-greedy over a per-action linear
    reward model), fit online via stochastic gradient updates. Unlike the
    Double DQN agent, it treats every trial independently and has no notion
    of the next state, so it cannot represent delayed consequences such as
    fatigue accumulation across a session (Section 7 justification).
    """
    name = "contextual_bandit"

    def __init__(self, state_dim: int = 12, n_actions: int = N_ACTIONS,
                 lr: float = 0.05, epsilon: float = 0.1, seed: int | None = None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.lr = lr
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        # one linear weight vector per action: predicted_reward = w_a . state + b_a
        self.weights = np.zeros((n_actions, state_dim), dtype=np.float64)
        self.bias = np.zeros(n_actions, dtype=np.float64)

    def _predict(self, state: np.ndarray) -> np.ndarray:
        return self.weights @ state + self.bias

    def act(self, state: np.ndarray) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self._predict(state)
        return int(np.argmax(q))

    def update(self, state: np.ndarray, action: int, reward: float):
        pred = self.weights[action] @ state + self.bias[action]
        error = reward - pred
        self.weights[action] += self.lr * error * state
        self.bias[action] += self.lr * error
