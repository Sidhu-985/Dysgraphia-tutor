"""
Bridge into the existing `dysgraphia_rl/` package (the offline RL
pre-training core, built separately -- see its own README).

`dysgraphia_rl` is laid out to be run with its own directory on
`sys.path` (its internal modules do `from utils.reward_utils import ...`,
`from agents.double_dqn_agent import ...`, not
`dysgraphia_rl.utils.reward_utils`). Rather than restructure that package
(and risk invalidating its own already-passing test suite / checkpoint),
this module inserts its root onto `sys.path` once and re-exports exactly
the symbols the backend needs. This is the ONLY file in the backend that
knows about that layout detail -- everything else imports from here.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .config import settings

_rl_root = (Path(__file__).resolve().parent.parent / settings.rl.dysgraphia_rl_root).resolve()
if str(_rl_root) not in sys.path:
    sys.path.insert(0, str(_rl_root))

from utils.state_utils import (  # noqa: E402
    STATE_FEATURE_NAMES,
    STATE_DIM,
    STATE_FEATURE_RANGES,
    normalize_speed,
    normalize_pressure_variance,
    normalize_retries,
    compute_improvement_trend,
    compute_consistency,
    build_state_vector,
    describe_state,
    validate_state_vector,
)
from utils.reward_utils import (  # noqa: E402
    RewardWeights,
    compute_reward,
    audit_action_distribution,
    unnecessary_break_penalty,
    validate_weights,
)
# The torch-based agent is imported lazily (module __getattr__ below), so
# preprocessing / feature extraction / state building / safety can be
# imported and tested without PyTorch installed. Only PolicyService needs it.
from envs.dysgraphia_tutor_env import (  # noqa: E402
    ACTION_NAMES,
    N_ACTIONS,
    STATE_DIM as ENV_STATE_DIM,
)

__all__ = [
    "STATE_FEATURE_NAMES", "STATE_DIM", "STATE_FEATURE_RANGES",
    "normalize_speed", "normalize_pressure_variance", "normalize_retries",
    "compute_improvement_trend", "compute_consistency", "build_state_vector",
    "describe_state", "validate_state_vector",
    "RewardWeights", "compute_reward", "audit_action_distribution",
    "unnecessary_break_penalty", "validate_weights",
    "DoubleDQNAgent", "QNetwork",
    "ACTION_NAMES", "N_ACTIONS", "ENV_STATE_DIM",
    "rl_root",
]

rl_root = _rl_root


def __getattr__(name):
    if name in ("DoubleDQNAgent", "QNetwork"):
        from agents import double_dqn_agent  # noqa: WPS433 (deliberately lazy)
        return getattr(double_dqn_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
