"""
Section 6 check: simulator state distribution vs backend state distribution.

Simulator: 100 episodes of the rule-based policy in DysgraphiaTutorEnv (v2).
Backend:   30 sessions x 12 attempts through the live API (TestClient). The
attempts are SYNTHETIC: clean traces, perturbed traces at random severity
and scribbles. Pressure is null, as on a touch screen.
For each of the 12 state features, reports mean/std/min/max on both sides,
the share of backend values outside the simulator's [1st, 99th] percentile
range, and whether the backend value is constant. A feature is FLAGGED when more than 20% of backend values fall outside the
simulator's [1st, 99th] percentile range. That is a mismatch the trained
policy has never seen. A constant backend value (a placeholder) is reported
but not flagged on its own. (The Phase D version of this script also
flagged constants. That was right then, because the simulator never
produced the placeholders.)

Usage (from backend/): python scripts/compare_state_distributions.py out.json
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("STORAGE_DB_PATH", os.path.join(tempfile.mkdtemp(), "cmp.db"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.dysgraphia_rl_bridge import STATE_FEATURE_NAMES, rl_root  # noqa: E402
from app.synthetic.handwriting_perturbations import perturb_single, clean_specimen  # noqa: E402
sys.path.insert(0, str(rl_root))
from envs.dysgraphia_tutor_env import DysgraphiaTutorEnv  # noqa: E402
from agents.baselines import RuleBasedAgent  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "live_scenarios", os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_scenarios.py"))
sc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sc)


def sim_states(n_eps=100):
    env, agent, out = DysgraphiaTutorEnv(), RuleBasedAgent(), []
    for ep in range(n_eps):
        s, _ = env.reset(seed=90000 + ep)
        out.append(s)
        done = False
        while not done:
            s, _, te, tr, _ = env.step(agent.act(s))
            out.append(s)
            done = te or tr
    return np.array(out)


def backend_states(n_sessions=30, n_attempts=12):
    rng = np.random.default_rng(7)
    c = TestClient(app)
    out = []
    for k in range(n_sessions):
        s = c.post("/session/start", json={"child_id": f"dist_{k:03d}", "data_source": "synthetic_test",
                                           "exercise_id": "loop_medium"}).json()
        ex = "loop_medium"
        for i in range(n_attempts):
            tpl = c.get(f"/exercise/{ex}").json()["reference_template"]
            kind = rng.choice(["clean", "jitter", "tremor", "slant", "scribble"])
            if kind == "clean":
                pts = sc.clean_trace(tpl)
            elif kind == "scribble":
                pts = sc.poor_trace(int(rng.integers(0, 10_000)))
            else:
                xy = perturb_single(clean_specimen(np.array(tpl), n_points=360), kind, float(rng.uniform(0.2, 1.0)),
                                    int(rng.integers(0, 10_000)))
                pts = [{"x": float(x), "y": float(y), "t": j / 120.0, "pressure": None, "stroke_id": 0}
                       for j, (x, y) in enumerate(xy)]
            r = c.post("/attempt/submit", json={"session_id": s["session_id"], "exercise_id": ex,
                                                "attempt_id": f"{k}_{i}", "points": pts}).json()
            out.append(r["state_vector"])
            if r["safety"]["safe_action_id"] == 7:
                c.post(f"/session/{s['session_id']}/break_complete")
            ex = r["next_exercise"]["exercise_id"]
        c.post(f"/session/{s['session_id']}/end")
    return np.array(out)


def main(path):
    S, B = sim_states(), backend_states()
    rows = {}
    for j, name in enumerate(STATE_FEATURE_NAMES):
        lo, hi = np.percentile(S[:, j], [1, 99])
        outside = float(np.mean((B[:, j] < lo - 1e-6) | (B[:, j] > hi + 1e-6)))
        rows[name] = {
            "sim": {"mean": float(S[:, j].mean()), "std": float(S[:, j].std()),
                    "p01": float(lo), "p99": float(hi)},
            "backend": {"mean": float(B[:, j].mean()), "std": float(B[:, j].std()),
                        "min": float(B[:, j].min()), "max": float(B[:, j].max())},
            "backend_outside_sim_p01_p99": outside,
            "backend_constant": bool(B[:, j].std() < 1e-6),
            # A constant backend value is only a mismatch if the simulator
            # never produces it. Since Round 3 the simulator emits the same
            # placeholders, so "constant" alone is reported but not flagged.
            "flag": bool(outside > 0.2),
        }
        print(f"{name:<20} sim {S[:, j].mean():.3f}+/-{S[:, j].std():.3f} [{lo:.2f},{hi:.2f}]   "
              f"backend {B[:, j].mean():.3f}+/-{B[:, j].std():.3f}   outside={outside:.0%}"
              f"{'  CONSTANT' if rows[name]['backend_constant'] else ''}{'  <-- FLAG' if rows[name]['flag'] else ''}")
    out = {"n_sim_states": len(S), "n_backend_states": len(B), "features": rows,
           "flagged_features": [n for n, r in rows.items() if r["flag"]],
           "data_label": "simulator states vs backend states from SYNTHETIC traces (no real-child data)"}
    json.dump(out, open(path, "w"), indent=1)
    print("flagged:", out["flagged_features"])


if __name__ == "__main__":
    main(sys.argv[1])
