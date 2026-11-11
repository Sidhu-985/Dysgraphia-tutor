"""
Offline pre-training script (Design and Solution Section 7, "Phase 1").

Trains the Double DQN agent inside DysgraphiaTutorEnv across randomly
sampled virtual learner profiles, then saves a checkpoint and a training
curve plot. This corresponds to the "Prototype implementation" work stream
in the project timeline (due 15 Oct) -- this script is the first runnable
slice of it for the Code Review milestone.

Reproducibility: --seed fixes Python's `random`, NumPy's global RNG, and
PyTorch's RNG (network init and prioritized-replay sampling) in addition to
the environment's own internal RNG. Two runs with identical arguments
produce bit-identical reward curves -- see
tests/test_extra.py::test_training_is_reproducible_given_seed.

Usage:
    python3 scripts/train_dqn.py --episodes 400 --seed 42
"""

from __future__ import annotations
import argparse
import os
import random
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv, STATE_DIM, N_ACTIONS
from agents.double_dqn_agent import DoubleDQNAgent


def set_global_seed(seed: int) -> None:
    """Seed every RNG the training loop touches. Called once, up front, so
    that `--seed 42` genuinely reproduces the same run end to end (network
    initialization, epsilon-greedy exploration, and prioritized-replay
    sampling all draw from global `random`/`numpy`/`torch` state)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear_decay(step, start, end, decay_steps):
    frac = min(1.0, step / max(1, decay_steps))
    return start + frac * (end - start)


def train(
    episodes: int = 400,
    max_steps: int = 40,
    batch_size: int = 64,
    warmup_steps: int = 500,
    seed: int = 0,
    lr: float = 1e-3,
    gamma: float = 0.95,
    buffer_capacity: int = 50_000,
    target_sync_every: int = 200,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_frac: float = 0.7,
    beta_start: float = 0.4,
    beta_end: float = 1.0,
    checkpoint_path: str = "checkpoints/double_dqn.pt",
    log_every: int = 20,
    quiet: bool = False,
    env_kwargs: dict | None = None,
):
    """Run one full offline pre-training job and save the resulting policy.

    All hyperparameters are exposed as arguments (rather than hard-coded)
    so the exact configuration used to produce a reported result is always
    visible in the command line / calling code, per the reproducibility
    requirements of the Code Review.
    """
    set_global_seed(seed)

    env = DysgraphiaTutorEnv(max_steps=max_steps, seed=seed, **(env_kwargs or {}))
    agent = DoubleDQNAgent(
        state_dim=STATE_DIM,
        n_actions=N_ACTIONS,
        lr=lr,
        gamma=gamma,
        buffer_capacity=buffer_capacity,
        target_sync_every=target_sync_every,
    )

    eps_decay_episodes = int(episodes * eps_decay_frac)

    episode_rewards = []
    episode_final_quality = []
    total_steps = 0

    for ep in range(episodes):
        state, info = env.reset(seed=seed + ep)
        epsilon = linear_decay(ep, eps_start, eps_end, eps_decay_episodes)
        beta = linear_decay(ep, beta_start, beta_end, episodes)

        ep_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            # `done` for the replay buffer is `terminated` ONLY, not
            # `terminated or truncated`: a true terminal state (the
            # environment's fatigue safety cutoff) should zero the Bellman
            # bootstrap term, but an ordinary time-limit truncation
            # (max_steps reached) should NOT -- the episode could in
            # principle have continued, so bootstrapping through it is the
            # standard, correct treatment. See envs/dysgraphia_tutor_env.py::step()
            # and tests/test_extra.py::test_train_dqn_stores_done_flag_on_true_termination.
            # Store the action that actually EXECUTED: the environment's
            # consecutive-break safety rule may replace a requested
            # SUGGEST_BREAK with CONTINUE_NO_CHANGE, and the observed
            # (reward, next_state) are the consequence of the executed
            # action, not the requested one.
            executed = step_info.get("executed_action", action)
            agent.store(state, executed, reward, next_state, float(terminated))

            if total_steps > warmup_steps:
                agent.train_step(batch_size=batch_size, beta=beta)

            state = next_state
            ep_reward += reward
            total_steps += 1

        episode_rewards.append(ep_reward)
        episode_final_quality.append(env.score_history[-1])

        if not quiet and (ep + 1) % log_every == 0:
            recent_r = np.mean(episode_rewards[-log_every:])
            recent_q = np.mean(episode_final_quality[-log_every:])
            print(f"Episode {ep+1:4d}/{episodes} | eps={epsilon:.3f} | "
                  f"avg_reward(last {log_every})={recent_r:7.3f} | "
                  f"avg_final_quality={recent_q:.3f}")

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    agent.save(checkpoint_path)
    if not quiet:
        print(f"\nSaved trained policy to {checkpoint_path}")

    return agent, episode_rewards, episode_final_quality


def plot_training_curve(episode_rewards, episode_final_quality, out_path="results/training_curve.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    def smooth(x, w=10):
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    axes[0].plot(episode_rewards, alpha=0.3, color="tab:blue", label="raw")
    axes[0].plot(smooth(episode_rewards), color="tab:blue", label="smoothed (w=10)")
    axes[0].set_title("Episode reward during offline training")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative reward")
    axes[0].legend()

    axes[1].plot(episode_final_quality, alpha=0.3, color="tab:green", label="raw")
    axes[1].plot(smooth(episode_final_quality), color="tab:green", label="smoothed (w=10)")
    axes[1].set_title("Final handwriting-quality score per episode")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Quality score")
    axes[1].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved training curve plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_capacity", type=int, default=50_000)
    parser.add_argument("--target_sync_every", type=int, default=200)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/double_dqn.pt")
    parser.add_argument("--plot_path", type=str, default="results/training_curve.png")
    args = parser.parse_args()

    agent, rewards, qualities = train(
        episodes=args.episodes, max_steps=args.max_steps, seed=args.seed,
        lr=args.lr, gamma=args.gamma, batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity, target_sync_every=args.target_sync_every,
        warmup_steps=args.warmup_steps, eps_start=args.eps_start, eps_end=args.eps_end,
        checkpoint_path=args.checkpoint,
    )
    plot_training_curve(rewards, qualities, out_path=args.plot_path)
