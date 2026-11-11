"""Train an experimental DQN on a diverse synthetic learner population.

Run from source/backend:
  python ../dysgraphia_rl/scripts/deployment/train_synthetic_population_candidate.py --episodes 5000 --profiles 1000 --seed 42

This writes only under checkpoints/deployment_candidates/SYNTHETIC_POPULATION.
It never overwrites or promotes the live deployment checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

BACKEND = Path.cwd()
sys.path.insert(0, str(BACKEND))
from app.dysgraphia_rl_bridge import rl_root  # noqa: E402
from app.rl_integration import behavioral_gate as BG  # noqa: E402

sys.path.insert(0, str(rl_root / "scripts"))
sys.path.insert(0, str(rl_root / "scripts" / "deployment"))
from train_dqn import train  # noqa: E402
import evaluate_baselines as EV  # noqa: E402
from gate_one import gate_path  # noqa: E402

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--profiles", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    if args.episodes < args.profiles:
        parser.error("--episodes should be >= --profiles to expose the agent to the population")

    outdir = rl_root / "checkpoints" / "deployment_candidates" / "SYNTHETIC_POPULATION"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt = Path(args.checkpoint) if args.checkpoint else outdir / f"seed{args.seed}.pt"
    if ckpt.exists():
        raise SystemExit(f"Refusing to overwrite existing candidate: {ckpt}")

    started = time.time()
    train(
        episodes=args.episodes,
        seed=args.seed,
        checkpoint_path=str(ckpt),
        quiet=True,
        env_kwargs={
            "improvement_signal": "latent",
            "profile_sampling": "synthetic_population",
            "synthetic_profile_count": args.profiles,
        },
    )

    report, q_fn = gate_path(ckpt)
    # Held-out population and jitter checks mirror the pre-existing candidate protocol.
    import numpy as np
    heldout = {}
    for s in (3001, 3002, 3003, 3004, 3005):
        acts = np.argmax(q_fn(BG.low_fatigue_population(seed=s)), axis=1)
        heldout[str(s)] = float(np.mean(acts == BG.SUGGEST_BREAK))
    rng = np.random.default_rng(77)
    probes = {
        "G1_fresh_good_writer": (BG.FRESH_GOOD_WRITER, (BG.SUGGEST_BREAK, BG.DECREASE_DIFFICULTY)),
        "G1_fresh_good_writer_deployed": (BG.FRESH_GOOD_WRITER_DEPLOYED, (BG.SUGGEST_BREAK, BG.DECREASE_DIFFICULTY)),
        "G2_mid_session_writer": (BG.MID_SESSION_WRITER, (BG.SUGGEST_BREAK,)),
        "G2_mid_session_writer_deployed": (BG.MID_SESSION_WRITER_DEPLOYED, (BG.SUGGEST_BREAK,)),
        "G3_struggling_writer": (BG.STRUGGLING_WRITER, (BG.INCREASE_DIFFICULTY, BG.SWITCH_TO_FREE_WRITING_MODE)),
    }
    jitter_violations = {}
    for name, (state, forbidden) in probes.items():
        states = np.clip(state[None, :] + rng.uniform(-0.03, 0.03, (50, state.size)), -1.0, 2.0).astype(np.float32)
        acts = np.argmax(q_fn(states), axis=1)
        jitter_violations[name] = int(np.isin(acts, forbidden).sum())
    supplementary_ok = (
        all(v < BG.MAX_LOW_FATIGUE_BREAK_RATE for v in heldout.values())
        and all(v == 0 for v in jitter_violations.values())
    )

    from agents.double_dqn_agent import DoubleDQNAgent
    from envs.dysgraphia_tutor_env import N_ACTIONS, STATE_DIM
    agent = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, device="cpu")
    agent.load(str(ckpt))
    offline_seeds = list(range(110000, 110150))
    heldout_profile_count = max(500, args.profiles // 2)
    heldout_profile_seed = args.seed + 900_000
    offline = EV.evaluate_policy(
        "synthetic_population_candidate",
        lambda state: agent.act(state, epsilon=0.0),
        offline_seeds,
        max_steps=40,
        env_kwargs={
            "profile_sampling": "synthetic_population",
            "synthetic_profile_count": heldout_profile_count,
            "seed": heldout_profile_seed,
            "improvement_signal": "latent",
        },
    )
    offline_ok = (
        offline.get("skill_gain", 0.0) > 0
        and offline.get("executed_break_fraction", 1.0) < 0.25
    )
    qualifies = bool(report.passed and supplementary_ok and offline_ok)
    record = {
        "experiment": "synthetic_population",
        "warning": "Synthetic parameter diversity is not real-child evidence or clinical validation.",
        "seed": args.seed,
        "episodes": args.episodes,
        "synthetic_profile_count": args.profiles,
        "heldout_synthetic_profile_count": heldout_profile_count,
        "heldout_profile_seed": heldout_profile_seed,
        "training_seconds": round(time.time() - started, 2),
        "torch_version": torch.__version__,
        "checkpoint": str(ckpt),
        "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        "behavioral_gate": {"passed": report.passed, "failures": report.failures, "details": report.details},
        "heldout_low_fatigue_break_rates": heldout,
        "jitter_violations": jitter_violations,
        "supplementary_passed": supplementary_ok,
        "offline": {
            k: offline.get(k) for k in (
                "mean_reward", "quality_gain", "skill_gain",
                "executed_break_fraction", "requested_break_fraction",
                "action_distribution", "audit_flags", "reward_hacking_suspected"
            )
        },
        "offline_passed": bool(offline_ok),
        "qualifies_for_manual_review": qualifies,
        "promoted_to_live": False,
    }
    record_path = ckpt.with_suffix(".json")
    record_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(ckpt),
        "gate_passed": report.passed,
        "gate_failures": report.failures,
        "supplementary_passed": supplementary_ok,
        "offline_passed": bool(offline_ok),
        "qualifies_for_manual_review": qualifies,
        "promoted_to_live": False,
        "report": str(record_path),
    }, indent=2))
    return 0 if qualifies else 2

if __name__ == "__main__":
    raise SystemExit(main())
