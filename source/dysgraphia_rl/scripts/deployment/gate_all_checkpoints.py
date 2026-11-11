"""Run the backend's UNCHANGED behavioural gate on every checkpoint under
dysgraphia_rl/checkpoints/ and record hash, architecture and gate result.
Usage (from backend/): python ../dysgraphia_rl/scripts/deployment/gate_all_checkpoints.py OUT.json"""
import hashlib, json, sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from app.rl_integration.behavioral_gate import run_behavioral_gate  # noqa: E402
from app.dysgraphia_rl_bridge import DoubleDQNAgent, ENV_STATE_DIM, N_ACTIONS, rl_root  # noqa: E402
import numpy as np, torch  # noqa: E402

def gate(path):
    ag = DoubleDQNAgent(state_dim=ENV_STATE_DIM, n_actions=N_ACTIONS, device="cpu")
    ag.load(str(path))
    shapes = {k: list(v.shape) for k, v in ag.online_net.state_dict().items()}
    def q(s):
        with torch.no_grad():
            return ag.online_net(torch.as_tensor(np.asarray(s, np.float32))).numpy()
    r = run_behavioral_gate(q)
    return {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "online_shapes": shapes,
            "passed": r.passed, "failures": r.failures, "details": r.details}

out = {}
for p in sorted((rl_root / "checkpoints").rglob("*.pt")):
    rel = str(p.relative_to(rl_root))
    try:
        out[rel] = gate(p)
    except Exception as e:  # incompatible checkpoint
        out[rel] = {"error": f"{type(e).__name__}: {e}"}
    o = out[rel]
    print(rel, o.get("passed"), o.get("details", {}).get("low_fatigue_break_rate"), o.get("failures", o.get("error")))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
