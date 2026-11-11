"""
Train one Double DQN per seed (docs/EVALUATION_PROTOCOL.md): reward v2,
800 episodes each, seeds 0-4. Saves:
  checkpoints/seeds/double_dqn_seed{k}.pt
  results/training/seed{k}_curve.csv     (episode, reward, final_quality)
  results/training/training_summary.json (settings + per-seed last-50 means)
Then copies the seed-0 checkpoint (the deployment choice fixed BEFORE
training) to checkpoints/double_dqn.pt.

Usage (from dysgraphia_rl/):  python3 scripts/train_all_seeds.py [--episodes 800]
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_dqn import train  # noqa: E402

DEPLOYED_SEED = 0  # fixed before training


def main(seeds, episodes, out_dir="results/training", ckpt_dir="checkpoints/seeds"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    summary = {"settings": {"episodes": episodes, "seeds": seeds, "reward_version": "v3", "observation_version": "v2 (Round 3; DysgraphiaTutorEnv defaults)",
                            "max_steps": 40, "max_consecutive_breaks": 1, "break_fatigue_threshold": 0.5,
                            "other_hyperparameters": "scripts/train_dqn.py::train defaults",
                            "deployed_seed": DEPLOYED_SEED},
               "per_seed": {}}
    for k in seeds:
        t0 = time.time()
        path = os.path.join(ckpt_dir, f"double_dqn_seed{k}.pt")
        _, rewards, qualities = train(episodes=episodes, seed=k, checkpoint_path=path, quiet=True)
        with open(os.path.join(out_dir, f"seed{k}_curve.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "reward", "final_quality"])
            for i, (r, q) in enumerate(zip(rewards, qualities)):
                w.writerow([i, r, q])
        summary["per_seed"][str(k)] = {
            "checkpoint": path, "train_seconds": round(time.time() - t0, 1),
            "last50_mean_reward": float(np.mean(rewards[-50:])),
            "first50_mean_reward": float(np.mean(rewards[:50])),
            "last50_mean_final_quality": float(np.mean(qualities[-50:])),
        }
        print(f"seed {k}: {summary['per_seed'][str(k)]}", flush=True)
    shutil.copyfile(os.path.join(ckpt_dir, f"double_dqn_seed{DEPLOYED_SEED}.pt"), "checkpoints/double_dqn.pt")
    summary["deployed_checkpoint"] = "checkpoints/double_dqn.pt (copy of seed %d)" % DEPLOYED_SEED
    summary["generated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("saved", os.path.join(out_dir, "training_summary.json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.seeds, a.episodes)
