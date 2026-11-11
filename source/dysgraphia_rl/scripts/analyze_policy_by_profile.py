"""
Interpretability analysis: which intervention strategies does the trained
policy actually use for which kind of learner?

Directly addresses Proposal Objective 6: "Analyse whether the learned
policy can provide interpretable personalised support, such as identifying
which intervention strategies are most useful for children with different
handwriting difficulties." Earlier versions of this repository trained and
evaluated the policy with profiles pooled together, which demonstrates
overall performance but says nothing about *personalization* -- this
script closes that gap by fixing one learner profile at a time and reading
off the trained policy's action distribution and outcome for each.

This is a diagnostic/reporting tool, not a new training method: it loads
the existing checkpoint and simply runs it, per profile, with epsilon=0
(fully greedy, no exploration noise) so the printed action distribution
reflects the policy's actual learned preference, not residual randomness.

Usage:
    python3 scripts/analyze_policy_by_profile.py --episodes 60 --seed 0 \
        --checkpoint checkpoints/double_dqn.pt
"""

from __future__ import annotations
import argparse
import csv
import os
import random
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv, PROFILES, STATE_DIM, N_ACTIONS, ACTION_NAMES
from agents.double_dqn_agent import DoubleDQNAgent


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_profile(agent, profile, n_episodes, max_steps, seed_offset):
    """Run the frozen, fully-greedy policy against ONE fixed learner
    profile for n_episodes episodes, and return the aggregated action
    distribution and outcome metrics."""
    env = DysgraphiaTutorEnv(profile=profile, max_steps=max_steps)
    action_counts = np.zeros(N_ACTIONS, dtype=int)
    final_qualities, fatigues, frustrations, engagements = [], [], [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_offset + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.act(state, epsilon=0.0)  # fully greedy: read the policy's true preference
            action_counts[action] += 1
            state, _, terminated, truncated, _ = env.step(action)
        final_qualities.append(env.score_history[-1])
        fatigues.append(env.fatigue)
        frustrations.append(env.frustration)
        engagements.append(env.engagement)

    total_actions = action_counts.sum()
    freqs = action_counts / total_actions if total_actions > 0 else action_counts.astype(float)
    return {
        "profile": profile.name,
        "mean_final_quality": float(np.mean(final_qualities)),
        "mean_fatigue": float(np.mean(fatigues)),
        "mean_frustration": float(np.mean(frustrations)),
        "mean_engagement": float(np.mean(engagements)),
        "action_frequencies": {ACTION_NAMES[i]: float(freqs[i]) for i in range(N_ACTIONS)},
    }


def top_k_actions(action_frequencies: dict, k: int = 3):
    return sorted(action_frequencies.items(), key=lambda kv: kv[1], reverse=True)[:k]


def main(episodes: int, max_steps: int, checkpoint: str, seed: int,
         csv_path: str = "results/policy_by_profile.csv",
         plot_path: str = "results/policy_by_profile.png"):
    set_global_seed(seed)

    agent = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
    if not os.path.exists(checkpoint):
        print(f"WARNING: no checkpoint at {checkpoint} -- analyzing an UNTRAINED network "
              f"(the point of this script is to interpret a TRAINED policy, so results "
              f"will not be meaningful without one).")
    else:
        agent.load(checkpoint)
        print(f"Loaded Double DQN checkpoint from {checkpoint}\n")

    results = []
    for i, profile in enumerate(PROFILES):
        r = run_profile(agent, profile, episodes, max_steps, seed_offset=6000 + i * 1000)
        results.append(r)

    # --- print a compact per-profile summary --------------------------------
    header = f"{'Profile':<20}{'MeanQuality':>13}{'MeanFatigue':>13}{'MeanFrustration':>17}   Top actions (share of steps)"
    print(header)
    print("-" * len(header))
    for r in results:
        top3 = top_k_actions(r["action_frequencies"], k=3)
        top3_str = ", ".join(f"{name} ({share*100:.0f}%)" for name, share in top3)
        print(f"{r['profile']:<20}{r['mean_final_quality']:>13.3f}{r['mean_fatigue']:>13.3f}"
              f"{r['mean_frustration']:>17.3f}   {top3_str}")

    print(
        "\nInterpretation note: if the trained policy is genuinely "
        "personalizing (Objective 6), the top actions and their shares "
        "should differ meaningfully across profiles -- e.g. a fatigue-prone "
        "profile should see more SUGGEST_BREAK than a fast improver, and a "
        "spacing-struggler should see more scaffolding actions (hints / "
        "enlarged template) than a profile without that specific weakness. "
        "If every profile's top-3 list looks identical, the policy has "
        "learned one generic strategy rather than a personalized one, and "
        "that would be worth reporting honestly rather than hidden."
    )

    save_outputs(results, csv_path=csv_path, plot_path=plot_path, meta=dict(
        episodes=episodes, max_steps=max_steps, seed=seed, checkpoint=checkpoint))
    return results


def save_outputs(results, csv_path, plot_path, meta=None):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fieldnames = ["profile", "mean_final_quality", "mean_fatigue", "mean_frustration",
                  "mean_engagement"] + [f"action_freq_{name}" for name in ACTION_NAMES]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"profile": r["profile"], "mean_final_quality": r["mean_final_quality"],
                   "mean_fatigue": r["mean_fatigue"], "mean_frustration": r["mean_frustration"],
                   "mean_engagement": r["mean_engagement"]}
            for name in ACTION_NAMES:
                row[f"action_freq_{name}"] = r["action_frequencies"][name]
            writer.writerow(row)
    print(f"\nSaved per-profile metrics to {csv_path}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profiles = [r["profile"] for r in results]
    action_matrix = np.array([[r["action_frequencies"][name] for name in ACTION_NAMES] for r in results])

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(action_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(ACTION_NAMES)))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(profiles)))
    ax.set_yticklabels(profiles)
    ax.set_title("Action frequency by learner profile (frozen, greedy policy)")
    fig.colorbar(im, ax=ax, label="fraction of steps")
    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    print(f"Saved action-frequency heatmap to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/double_dqn.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv_path", type=str, default="results/policy_by_profile.csv")
    parser.add_argument("--plot_path", type=str, default="results/policy_by_profile.png")
    args = parser.parse_args()
    main(args.episodes, args.max_steps, args.checkpoint, args.seed, args.csv_path, args.plot_path)
