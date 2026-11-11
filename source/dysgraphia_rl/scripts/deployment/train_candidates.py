"""Train and qualify DQN deployment candidates (docs/DQN_DEPLOYMENT_CANDIDATE_PROTOCOL.md).

Usage (from backend/, so the backend's gate is importable):
    python ../dysgraphia_rl/scripts/deployment/train_candidates.py RUNG SEED [SEED ...]

Writes dysgraphia_rl/checkpoints/deployment_candidates/<RUNG>/seed<k>.pt and
<...>/seed<k>.json. Never touches historical checkpoints.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BACKEND = Path.cwd()
sys.path.insert(0, str(BACKEND))
from app.dysgraphia_rl_bridge import ACTION_NAMES, rl_root  # noqa: E402  (puts dysgraphia_rl on sys.path)
from app.rl_integration import behavioral_gate as BG  # noqa: E402

sys.path.insert(0, str(rl_root / "scripts"))
sys.path.insert(0, str(rl_root / "scripts" / "deployment"))
from train_dqn import train  # noqa: E402  (the existing, approved training function)
import evaluate_baselines as EV  # noqa: E402
from gate_one import gate_path  # noqa: E402

RUNGS = {
    "C1": dict(episodes=800, env_kwargs=dict(improvement_signal="latent")),
    "C2": dict(episodes=1600, env_kwargs=dict(improvement_signal="latent")),
    "C3": dict(episodes=800, env_kwargs=dict(improvement_signal="latent",
                                             reward_weights=dict(w6_unnecessary_break=2.5))),
}
HELDOUT_POP_SEEDS = (2001, 2002, 2003, 2004, 2005)
JITTER_SEED, N_JITTER, JITTER = 77, 50, 0.03
OFFLINE_SEEDS = list(range(100_000, 100_150))


def supplementary(q_fn) -> dict:
    out = {"heldout_break_rates": {}}
    for s in HELDOUT_POP_SEEDS:
        a = np.argmax(q_fn(BG.low_fatigue_population(seed=s)), axis=1)
        out["heldout_break_rates"][s] = float(np.mean(a == BG.SUGGEST_BREAK))
    rng = np.random.default_rng(JITTER_SEED)
    bad = {}
    probes = {"G1_fresh_good_writer": (BG.FRESH_GOOD_WRITER, (BG.SUGGEST_BREAK, BG.DECREASE_DIFFICULTY)),
              "G1_fresh_good_writer_deployed": (BG.FRESH_GOOD_WRITER_DEPLOYED, (BG.SUGGEST_BREAK, BG.DECREASE_DIFFICULTY)),
              "G2_mid_session_writer": (BG.MID_SESSION_WRITER, (BG.SUGGEST_BREAK,)),
              "G2_mid_session_writer_deployed": (BG.MID_SESSION_WRITER_DEPLOYED, (BG.SUGGEST_BREAK,)),
              "G3_struggling_writer": (BG.STRUGGLING_WRITER, (BG.INCREASE_DIFFICULTY, BG.SWITCH_TO_FREE_WRITING_MODE))}
    for name, (st, forbidden) in probes.items():
        jit = np.clip(st[None, :] + rng.uniform(-JITTER, JITTER, (N_JITTER, st.size)), 0.0, None).astype(np.float32)
        a = np.argmax(q_fn(jit), axis=1)
        bad[name] = int(np.isin(a, forbidden).sum())
    out["jitter_violations"] = bad
    out["passed"] = (all(r < BG.MAX_LOW_FATIGUE_BREAK_RATE for r in out["heldout_break_rates"].values())
                     and all(v == 0 for v in bad.values()))
    return out


def offline(ckpt: Path) -> dict:
    from agents.double_dqn_agent import DoubleDQNAgent
    from envs.dysgraphia_tutor_env import N_ACTIONS, STATE_DIM
    ag = DoubleDQNAgent(state_dim=STATE_DIM, n_actions=N_ACTIONS, device="cpu")
    ag.load(str(ckpt))
    r = EV.evaluate_policy("candidate", lambda s: ag.act(s, epsilon=0.0), OFFLINE_SEEDS, max_steps=40)
    keep = ("mean_reward", "quality_gain", "skill_gain", "mean_final_difficulty", "n_episodes")
    out = {k: r[k] for k in keep if k in r}
    out.update({k: r[k] for k in ("executed_break_fraction", "requested_break_fraction", "break_cap_rate",
                                  "termination_fraction", "action_distribution", "audit_flags",
                                  "reward_hacking_suspected")})
    return out


def main(rung: str, seeds: list[int]) -> None:
    cfg = RUNGS[rung]
    outdir = rl_root / "checkpoints" / "deployment_candidates" / rung
    outdir.mkdir(parents=True, exist_ok=True)
    for k in seeds:
        path = outdir / f"seed{k}.pt"
        if path.exists():
            raise SystemExit(f"{path} exists; refusing to overwrite")
        t0 = time.time()
        train(episodes=cfg["episodes"], seed=k, checkpoint_path=str(path), quiet=True, env_kwargs=cfg["env_kwargs"])
        rep, q_fn = gate_path(path)
        sup = supplementary(q_fn)
        off = offline(path)
        exec_breaks = off["executed_break_fraction"]
        offline_ok = off["skill_gain"] > 0 and exec_breaks < 0.25
        rec = {"rung": rung, "seed": k, "train_config": {**cfg, "max_steps": 40, "other": "train_dqn.train defaults"},
               "checkpoint": str(path.relative_to(rl_root)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "train_seconds": round(time.time() - t0, 1), "torch": torch.__version__,
               "gate": {"passed": rep.passed, "failures": rep.failures, "details": rep.details},
               "supplementary": sup, "offline": off, "offline_ok": bool(offline_ok),
               "qualifies": bool(rep.passed and sup["passed"] and offline_ok)}
        (outdir / f"seed{k}.json").write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps({"rung": rung, "seed": k, "gate": rep.passed, "fail": rep.failures,
                          "probes": rep.details.get("probe_actions"),
                          "br": rep.details.get("low_fatigue_break_rate"), "sup": sup["passed"],
                          "heldout": sup["heldout_break_rates"], "jit": sup["jitter_violations"],
                          "skill_gain": off["skill_gain"], "breaks": exec_breaks, "qualifies": rec["qualifies"]}),
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]])
