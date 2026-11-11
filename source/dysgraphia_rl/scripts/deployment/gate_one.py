"""Gate one or more checkpoint files with the backend's unchanged behavioural gate.
Usage (from backend/): python ../dysgraphia_rl/scripts/deployment/gate_one.py CKPT [CKPT ...]"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import numpy as np, torch  # noqa: E402
from app.rl_integration.behavioral_gate import run_behavioral_gate  # noqa: E402
from app.dysgraphia_rl_bridge import DoubleDQNAgent, ENV_STATE_DIM, N_ACTIONS  # noqa: E402


def gate_path(p):
    ag = DoubleDQNAgent(state_dim=ENV_STATE_DIM, n_actions=N_ACTIONS, device="cpu")
    ag.load(str(p))
    def q(s):
        with torch.no_grad():
            return ag.online_net(torch.as_tensor(np.asarray(s, np.float32))).numpy()
    return run_behavioral_gate(q), q


if __name__ == "__main__":
    for p in sys.argv[1:]:
        r, _ = gate_path(p)
        print(json.dumps({"ckpt": p, "passed": r.passed, "break_rate": r.details.get("low_fatigue_break_rate"),
                          "probes": r.details.get("probe_actions"), "failures": r.failures}))
