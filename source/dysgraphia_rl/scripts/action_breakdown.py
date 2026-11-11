"""
Action-level breakdown for the DQN checkpoints on the protocol's evaluation
seeds (docs/EVALUATION_PROTOCOL.md): counts and fractions of PROPOSED
(policy output) and EXECUTED actions for all 10 actions, per seed and in
total. Also reports the fatigue and frustration at which executed breaks
happened. In the simulator the only override between proposal and execution
is the consecutive-break cap, so "accepted" equals "executed" there. Live
proposed/accepted/executed counts come from the backend's attempts table:
raw_action_id, safe_action_id, and the intervention action, which is the
same as safe_action_id.

Usage (from dysgraphia_rl/):
  python scripts/action_breakdown.py --checkpoints "checkpoints/seeds/double_dqn_seed{k}.pt" --out f.json [--env-kwargs '{}']
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv, ACTION_NAMES, N_ACTIONS, SUGGEST_BREAK  # noqa: E402
from agents.double_dqn_agent import DoubleDQNAgent  # noqa: E402
from scripts.evaluate_baselines import eval_seeds  # noqa: E402


def breakdown(pattern, runs, n_episodes, env_kwargs):
    out = {"runs": {}, "env_kwargs": env_kwargs, "episodes_per_run": n_episodes}
    tot_p = np.zeros(N_ACTIONS, int)
    tot_e = np.zeros(N_ACTIONS, int)
    fat, fru = [], []
    for k in runs:
        agent = DoubleDQNAgent(state_dim=12, n_actions=N_ACTIONS)
        agent.load(pattern.format(k=k))
        env = DysgraphiaTutorEnv(max_steps=40, **env_kwargs)
        p = np.zeros(N_ACTIONS, int)
        e = np.zeros(N_ACTIONS, int)
        kfat = []
        for seed in eval_seeds(k, n_episodes):
            s, _ = env.reset(seed=seed)
            done = False
            while not done:
                a = agent.act(s, epsilon=0.0)
                f0, fr0 = env.fatigue, env.frustration
                s, _, te, tr, info = env.step(a)
                p[a] += 1
                e[info["executed_action"]] += 1
                if info["executed_action"] == SUGGEST_BREAK:
                    kfat.append(f0)
                    fru.append(fr0)
                done = te or tr
        fat += kfat
        tot_p += p
        tot_e += e
        out["runs"][str(k)] = {
            "total_actions": int(p.sum()),
            "proposed_counts": {ACTION_NAMES[i]: int(p[i]) for i in range(N_ACTIONS)},
            "executed_counts": {ACTION_NAMES[i]: int(e[i]) for i in range(N_ACTIONS)},
            "proposed_break_fraction": float(p[SUGGEST_BREAK] / p.sum()),
            "executed_break_fraction": float(e[SUGGEST_BREAK] / e.sum()),
            "median_fatigue_at_executed_break": float(np.median(kfat)) if kfat else None,
        }
    out["total"] = {
        "total_actions": int(tot_p.sum()),
        "proposed_counts": {ACTION_NAMES[i]: int(tot_p[i]) for i in range(N_ACTIONS)},
        "executed_counts": {ACTION_NAMES[i]: int(tot_e[i]) for i in range(N_ACTIONS)},
        "executed_fractions": {ACTION_NAMES[i]: float(tot_e[i] / tot_e.sum()) for i in range(N_ACTIONS)},
        "proposed_break_fraction": float(tot_p[SUGGEST_BREAK] / tot_p.sum()),
        "executed_break_fraction": float(tot_e[SUGGEST_BREAK] / tot_e.sum()),
        "breaks_replaced_by_cap": int(tot_p[SUGGEST_BREAK] - tot_e[SUGGEST_BREAK]),
        "executed_breaks_at_fatigue_below_0.5": float(np.mean(np.array(fat) < 0.5)) if fat else None,
        "median_fatigue_at_executed_break": float(np.median(fat)) if fat else None,
        "median_frustration_at_executed_break": float(np.median(fru)) if fru else None,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="checkpoints/seeds/double_dqn_seed{k}.pt")
    ap.add_argument("--runs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--env-kwargs", default='{"reward_version": "v3", "max_consecutive_breaks": 1, "break_fatigue_threshold": 0.5, "observation_version": "v2"}')
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = breakdown(a.checkpoints, a.runs, a.episodes, json.loads(a.env_kwargs))
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res["total"], indent=1))
