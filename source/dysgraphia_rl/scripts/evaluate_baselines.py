"""
Multi-seed evaluation of the Double DQN policy against baselines, following
the PREDECLARED protocol in docs/EVALUATION_PROTOCOL.md (reward version 2).

Policies (all evaluated on identical episode seeds within a run -- paired):
    Random, Rule-based, Contextual Bandit (online), Contextual Bandit
    (frozen), Do-nothing (always CONTINUE_NO_CHANGE), Always-break (always
    requests SUGGEST_BREAK), Double DQN (one checkpoint per training seed).

Per run k and policy: mean episode reward, quality gain
(score_history[-1] - score_history[0]), difficulty-matched skill gain,
final difficulty, executed/requested break fractions, break-cap rate,
executed action distribution, termination fraction, generalized
reward-hacking audit. Across runs: mean and 95% CI (t-distribution, df =
n_runs - 1). Every per-run value is saved -- nothing is dropped.

All results are SIMULATED / OFFLINE (DysgraphiaTutorEnv), not real-child data.

Usage (from dysgraphia_rl/):
    python3 scripts/evaluate_baselines.py \
        --checkpoints checkpoints/seeds/double_dqn_seed{k}.pt --runs 0 1 2 3 4
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.dysgraphia_tutor_env import (
    DysgraphiaTutorEnv, STATE_DIM, N_ACTIONS, DECREASE_DIFFICULTY, SUGGEST_BREAK, ACTION_NAMES,
)
from agents.baselines import (
    RandomAgent, RuleBasedAgent, ContextualBanditAgent, AlwaysBreakAgent, DoNothingAgent,
)
from utils.reward_utils import audit_action_distribution

EVAL_SEED_BASE = 100_000
BANDIT_PRETRAIN_SEED_BASE = 50_000
# two-sided 95% t critical values by degrees of freedom
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228}

# Round 2 (docs/EVALUATION_PROTOCOL.md amendment): reward v3. Round 1 used
# "v2"; its results are kept in results/round1/ (pass --reward-version v2
# to re-evaluate the round-1 checkpoints in checkpoints/seeds_round1/).
# Round 3 (amendment): observation_version "v2" (backend-consistent
# observations and quality formula). Rounds 1-2 used "v1"; reproduce them with
# --reward-version v2|v3 --observation-version v1.
ENV_KW = dict(reward_version="v3", max_consecutive_breaks=1, break_fatigue_threshold=0.5,
              observation_version="v2")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def eval_seeds(run: int, n_episodes: int) -> list[int]:
    return [EVAL_SEED_BASE + 1000 * run + ep for ep in range(n_episodes)]


def mean_ci(values) -> dict:
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(np.mean(v)) if n else float("nan")
    if n < 2:
        return {"mean": m, "ci_low": float("nan"), "ci_high": float("nan"), "n": n}
    t = _T975.get(n - 1, 1.96)
    half = t * float(np.std(v, ddof=1)) / math.sqrt(n)
    return {"mean": m, "ci_low": m - half, "ci_high": m + half, "n": n}


def run_episode(env, policy_fn, seed, bandit_agent=None):
    state, _ = env.reset(seed=seed)
    start_q = env.score_history[0]
    start_skill = env.latent_skill()
    ep_reward = 0.0
    executed, requested, capped = [], [], 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = policy_fn(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        ex = info.get("executed_action", action)
        if bandit_agent is not None:
            bandit_agent.update(state, ex, reward)
        executed.append(ex)
        requested.append(action)
        capped += int(info.get("break_capped", False))
        ep_reward += reward
        state = next_state
    return {
        "reward": ep_reward,
        "quality_gain": env.score_history[-1] - start_q,
        "final_quality": env.score_history[-1],
        "skill_gain": env.latent_skill() - start_skill,
        "final_difficulty": env.difficulty,
        "terminated": bool(terminated),
        "executed": executed, "requested": requested, "capped": capped,
    }


def pretrain_bandit(bandit_agent, n_episodes, max_steps, seed_base):
    env = DysgraphiaTutorEnv(max_steps=max_steps, **ENV_KW)
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_base + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = bandit_agent.act(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            bandit_agent.update(state, info.get("executed_action", action), reward)
            state = next_state


def evaluate_policy(name, policy_fn, seeds, max_steps, bandit_agent=None, env_kwargs=None) -> dict:
    """Evaluate a policy on fixed seeds, optionally using a separate env population.

    ``env_kwargs`` is additive and leaves the historical evaluation protocol
    unchanged when omitted. Deployment candidate experiments use it to create
    a held-out synthetic profile population that is distinct from training.
    """
    merged_env_kwargs = {**ENV_KW, **(env_kwargs or {})}
    env = DysgraphiaTutorEnv(max_steps=max_steps, **merged_env_kwargs)
    eps = [run_episode(env, policy_fn, s, bandit_agent=bandit_agent) for s in seeds]
    executed = [a for e in eps for a in e["executed"]]
    requested = [a for e in eps for a in e["requested"]]
    counts = np.bincount(executed, minlength=N_ACTIONS) / max(1, len(executed))
    q_gain = float(np.mean([e["quality_gain"] for e in eps]))
    audit = audit_action_distribution(
        executed, N_ACTIONS, easy_actions=(DECREASE_DIFFICULTY,),
        mean_quality=float(np.mean([e["final_quality"] for e in eps])),
        quality_gain=q_gain)
    return {
        "policy": name,
        "n_episodes": len(eps),
        "mean_reward": float(np.mean([e["reward"] for e in eps])),
        "std_reward": float(np.std([e["reward"] for e in eps])),
        "quality_gain": q_gain,
        "mean_final_quality": float(np.mean([e["final_quality"] for e in eps])),
        "skill_gain": float(np.mean([e["skill_gain"] for e in eps])),
        "mean_final_difficulty": float(np.mean([e["final_difficulty"] for e in eps])),
        "executed_break_fraction": float(counts[SUGGEST_BREAK]),
        "requested_break_fraction": float(np.mean([a == SUGGEST_BREAK for a in requested])),
        "break_cap_rate": float(sum(e["capped"] for e in eps) / max(1, len(requested))),
        "termination_fraction": float(np.mean([e["terminated"] for e in eps])),
        "action_distribution": {ACTION_NAMES[i]: float(counts[i]) for i in range(N_ACTIONS)},
        "audit_flags": audit["flags"],
        "reward_hacking_suspected": audit["reward_hacking_suspected"],
    }


def evaluate_run(run: int, checkpoint: str | None, n_episodes: int, max_steps: int,
                 bandit_pretrain_episodes: int) -> list[dict]:
    seeds = eval_seeds(run, n_episodes)
    set_global_seed(run)
    rows = []
    rows.append(evaluate_policy("Random", RandomAgent(seed=run).act, seeds, max_steps))
    rows.append(evaluate_policy("Rule-based", RuleBasedAgent().act, seeds, max_steps))
    online = ContextualBanditAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, seed=run)
    rows.append(evaluate_policy("Contextual Bandit (online)", online.act, seeds, max_steps,
                                bandit_agent=online))
    frozen = ContextualBanditAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, seed=run + 1)
    pretrain_bandit(frozen, bandit_pretrain_episodes, max_steps, BANDIT_PRETRAIN_SEED_BASE + 1000 * run)
    frozen.epsilon = 0.0
    rows.append(evaluate_policy("Contextual Bandit (frozen)", frozen.act, seeds, max_steps))
    rows.append(evaluate_policy("Do-nothing", DoNothingAgent().act, seeds, max_steps))
    rows.append(evaluate_policy("Always-break", AlwaysBreakAgent().act, seeds, max_steps))
    if checkpoint:
        from agents.double_dqn_agent import DoubleDQNAgent
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"DQN checkpoint for run {run} not found: {checkpoint}")
        dqn = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS)
        dqn.load(checkpoint)
        row = evaluate_policy("Double DQN", lambda s: dqn.act(s, epsilon=0.0), seeds, max_steps)
        row["checkpoint"] = checkpoint
        rows.append(row)
    for r in rows:
        r["run"] = run
    return rows


SUMMARY_METRICS = ["mean_reward", "quality_gain", "skill_gain", "mean_final_difficulty",
                   "executed_break_fraction", "requested_break_fraction", "termination_fraction"]


def aggregate(per_run: list[dict]) -> dict:
    policies = list(dict.fromkeys(r["policy"] for r in per_run))
    out = {}
    for p in policies:
        rows = sorted([r for r in per_run if r["policy"] == p], key=lambda r: r["run"])
        out[p] = {m: mean_ci([r[m] for r in rows]) for m in SUMMARY_METRICS}
        out[p]["per_run"] = {m: [r[m] for r in rows] for m in SUMMARY_METRICS}
        out[p]["runs"] = [r["run"] for r in rows]
        out[p]["audit_flagged_runs"] = [r["run"] for r in rows if r["reward_hacking_suspected"]]
        acts = {a: float(np.mean([r["action_distribution"][a] for r in rows])) for a in ACTION_NAMES}
        out[p]["action_distribution_mean"] = acts
    return out


def paired_difference(per_run, a, b, metric):
    ra = {r["run"]: r[metric] for r in per_run if r["policy"] == a}
    rb = {r["run"]: r[metric] for r in per_run if r["policy"] == b}
    runs = sorted(set(ra) & set(rb))
    return mean_ci([ra[k] - rb[k] for k in runs])


def acceptance(agg: dict, per_run: list[dict]) -> dict:
    if "Double DQN" not in agg:
        return {"status": "NOT RUN", "reason": "no DQN checkpoints supplied"}
    dqn, ab = agg["Double DQN"], agg["Always-break"]
    res = {}
    ab_mean = ab["mean_reward"]["mean"]
    if ab_mean > 0:
        ratio = dqn["mean_reward"]["mean"] / ab_mean
        res["reward_vs_always_break"] = {
            "rule": "DQN mean reward >= 2x always-break (always-break mean > 0)",
            "ratio": ratio, "pass": bool(ratio >= 2.0)}
    else:
        res["reward_vs_always_break"] = {
            "rule": "always-break mean <= 0, ratio meaningless: DQN CI low > always-break CI high",
            "dqn_ci_low": dqn["mean_reward"]["ci_low"], "always_break_ci_high": ab["mean_reward"]["ci_high"],
            "pass": bool(dqn["mean_reward"]["ci_low"] > ab["mean_reward"]["ci_high"])}
    res["positive_quality_gain"] = {
        "rule": "95% CI lower bound of DQN mean quality gain > 0",
        "ci_low": dqn["quality_gain"]["ci_low"], "pass": bool(dqn["quality_gain"]["ci_low"] > 0)}
    res["break_fraction"] = {
        "rule": "mean executed break fraction < 0.25",
        "value": dqn["executed_break_fraction"]["mean"],
        "pass": bool(dqn["executed_break_fraction"]["mean"] < 0.25)}
    res["no_reward_hacking"] = {
        "rule": "generalized audit flags the DQN in no run",
        "flagged_runs": dqn["audit_flagged_runs"], "pass": len(dqn["audit_flagged_runs"]) == 0}
    res["contribution_over_do_nothing"] = {
        m: paired_difference(per_run, "Double DQN", "Do-nothing", m)
        for m in ("mean_reward", "quality_gain", "skill_gain")}
    res["all_pass"] = all(v["pass"] for k, v in res.items() if isinstance(v, dict) and "pass" in v)
    return res


def fmt(ci, digits=2):
    if math.isnan(ci["ci_low"]):
        return f"{ci['mean']:.{digits}f}"
    return f"{ci['mean']:.{digits}f} [{ci['ci_low']:.{digits}f}, {ci['ci_high']:.{digits}f}]"


def print_table(agg):
    header = (f"{'Policy':<28}{'Reward [95% CI]':>26}{'Quality gain':>26}"
              f"{'Skill gain':>24}{'Exec break %':>14}{'Final diff':>12}{'Term %':>8}")
    print("\n" + header)
    print("-" * len(header))
    for p, a in agg.items():
        print(f"{p:<28}{fmt(a['mean_reward']):>26}{fmt(a['quality_gain'], 3):>26}"
              f"{fmt(a['skill_gain'], 3):>24}{a['executed_break_fraction']['mean']*100:>13.1f}%"
              f"{a['mean_final_difficulty']['mean']:>12.2f}{a['termination_fraction']['mean']*100:>7.1f}%")


def main(runs, checkpoint_pattern, n_episodes, max_steps, bandit_pretrain_episodes,
         out_json, out_csv, plot_path):
    per_run = []
    for k in runs:
        ckpt = checkpoint_pattern.format(k=k) if checkpoint_pattern else None
        per_run.extend(evaluate_run(k, ckpt, n_episodes, max_steps, bandit_pretrain_episodes))
    agg = aggregate(per_run)
    acc = acceptance(agg, per_run)
    print_table(agg)
    print("\nAcceptance (docs/EVALUATION_PROTOCOL.md):")
    print(json.dumps(acc, indent=2, default=float))

    payload = {
        "metadata": {
            "protocol": "docs/EVALUATION_PROTOCOL.md",
            "runs": list(runs), "episodes_per_run": n_episodes, "max_steps": max_steps,
            "env_kwargs": ENV_KW, "checkpoint_pattern": checkpoint_pattern,
            "bandit_pretrain_episodes": bandit_pretrain_episodes,
            "eval_seed_rule": f"{EVAL_SEED_BASE} + 1000*run + episode",
            "ci_rule": "mean +/- t_0.975,df * sd/sqrt(n), df = n_runs - 1",
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "SIMULATED / OFFLINE results from DysgraphiaTutorEnv -- not real-child data.",
        },
        "aggregate": agg, "acceptance": acc, "per_run": per_run,
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"Saved {out_json}")

    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy", "run", "n_episodes", "mean_reward", "quality_gain", "skill_gain",
                    "mean_final_difficulty", "executed_break_fraction", "requested_break_fraction",
                    "break_cap_rate", "termination_fraction", "reward_hacking_suspected", "audit_flags"])
        for r in per_run:
            w.writerow([r["policy"], r["run"], r["n_episodes"], r["mean_reward"], r["quality_gain"],
                        r["skill_gain"], r["mean_final_difficulty"], r["executed_break_fraction"],
                        r["requested_break_fraction"], r["break_cap_rate"], r["termination_fraction"],
                        r["reward_hacking_suspected"], ";".join(r["audit_flags"])])
    print(f"Saved {out_csv}")
    if plot_path:
        plot_comparison(agg, plot_path)
    return payload


def plot_comparison(agg, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(agg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, metric, title in zip(
            axes, ["mean_reward", "quality_gain", "executed_break_fraction"],
            ["Mean episode reward (reward v2)", "Quality gain (final - start)", "Executed break fraction"]):
        means = [agg[n][metric]["mean"] for n in names]
        lo = [agg[n][metric]["mean"] - agg[n][metric]["ci_low"] for n in names]
        hi = [agg[n][metric]["ci_high"] - agg[n][metric]["mean"] for n in names]
        ax.bar(range(len(names)), means, yerr=[lo, hi], capsize=4, color="#5b8f8a")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title + "\n(mean over runs, 95% CI)", fontsize=10)
    fig.suptitle("SIMULATED results (DysgraphiaTutorEnv) -- not real-child data", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--checkpoints", type=str, default="checkpoints/seeds/double_dqn_seed{k}.pt",
                    help="checkpoint path pattern with {k} for the run/training seed; '' to skip the DQN")
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--reward-version", choices=["v2", "v3"], default="v3")
    ap.add_argument("--observation-version", choices=["v1", "v2"], default="v2")
    ap.add_argument("--max_steps", type=int, default=40)
    ap.add_argument("--bandit_pretrain_episodes", type=int, default=800)
    ap.add_argument("--out_json", type=str, default="results/evaluation_summary.json")
    ap.add_argument("--out_csv", type=str, default="results/metrics.csv")
    ap.add_argument("--plot_path", type=str, default="results/baseline_comparison.png")
    a = ap.parse_args()
    ENV_KW["reward_version"] = a.reward_version
    ENV_KW["observation_version"] = a.observation_version
    main(a.runs, a.checkpoints, a.episodes, a.max_steps, a.bandit_pretrain_episodes,
         a.out_json, a.out_csv, a.plot_path)
