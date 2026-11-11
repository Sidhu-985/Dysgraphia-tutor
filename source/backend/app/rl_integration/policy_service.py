"""
PolicyService: the ONLY place the backend talks to the trained Double DQN
agent from `dysgraphia_rl/`. Everything upstream (StateBuilder) and
downstream (safety layer, intervention controller) is decoupled from the
concrete RL implementation.

Policy modes (reported by /health and in every attempt response):
  * "dqn": checkpoint loaded AND passed the behavioural gate
    (behavioral_gate.py). The DQN's greedy action is the decision.
  * "rule_based_fallback": checkpoint missing, unloadable, or failed the
    gate. The transparent rule-based policy from dysgraphia_rl/agents
    makes every decision.
The gate cannot be bypassed for live decisions (2026-11-08): it always runs
when a checkpoint loads, and "dqn" requires a passed gate. The old
RL_BEHAVIORAL_GATE_ENABLED=false switch is still read, but it is only
reported; it no longer lets a gated-out checkpoint make decisions.
Per-decision safeguards, in either mode:
  * Invalid state (wrong shape, NaN/inf, out of canonical range): no model
    is queried. The decision is CONTINUE_NO_CHANGE with
    source="safe_default_invalid_state". The safety layer still runs after it.
  * Exception during DQN inference: the rule-based policy decides on that
    state, with source="rule_based_after_inference_error".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np

from ..config import settings
from ..dysgraphia_rl_bridge import (
    DoubleDQNAgent, N_ACTIONS, ACTION_NAMES, ENV_STATE_DIM, rl_root, validate_state_vector,
)
from .behavioral_gate import run_behavioral_gate, GateReport
from .intelligent_tutor import IntelligentRuleBasedTutor

CONTINUE_NO_CHANGE = ACTION_NAMES.index("CONTINUE_NO_CHANGE")


@dataclass
class PolicyDecision:
    action_id: int
    action_name: str
    q_values: list[float]
    greedy: bool
    source: str = "dqn"
    notes: list[str] = field(default_factory=list)


def _rule_based_agent():
    from agents.baselines import RuleBasedAgent  # dysgraphia_rl on sys.path via the bridge
    return RuleBasedAgent()


class PolicyService:
    def __init__(self, checkpoint_path: str | None = None, device: str | None = None,
                 behavioral_gate: bool | None = None):
        self.agent = DoubleDQNAgent(
            state_dim=ENV_STATE_DIM, n_actions=N_ACTIONS, device=device or settings.rl.device
        )
        self.fallback = _rule_based_agent()
        # The intelligent tutor is used in place of the brittle legacy
        # RuleBasedAgent whenever a SessionContext is available at
        # decision time (i.e. on the live API path). The legacy
        # ``RuleBasedAgent`` is retained so the RL training tests that
        # call ``agent.act(state)`` directly keep working unchanged.
        self.intelligent_fallback = IntelligentRuleBasedTutor()
        self.checkpoint_loaded = False
        self.load_error: str | None = None
        self.gate_report: GateReport | None = None

        repo_root = Path(__file__).resolve().parents[3]
        raw = Path(checkpoint_path or settings.rl.checkpoint_path)
        candidate = raw if raw.is_absolute() else (repo_root / raw).resolve()
        if not candidate.exists() and not raw.is_absolute():
            candidate = (rl_root / raw).resolve()

        if candidate.exists():
            try:
                self.agent.load(str(candidate))
                self.checkpoint_loaded = True
            except Exception as exc:  # corrupt / incompatible checkpoint
                self.load_error = f"{type(exc).__name__}: {exc}"
        else:
            self.load_error = "checkpoint file not found"
        self.checkpoint_path = str(candidate)

        # The gate ALWAYS runs on a loaded checkpoint. The configuration
        # switch is kept only so old configs do not fail; it is reported in
        # /health and never lets a checkpoint skip the gate.
        self.gate_config_requested = (settings.rl.behavioral_gate_enabled
                                      if behavioral_gate is None else behavioral_gate)
        self.gate_enabled = True
        if self.checkpoint_loaded:
            try:
                self.gate_report = run_behavioral_gate(self._q_values_batch)
            except Exception as exc:
                self.gate_report = GateReport(False, [f"gate crashed: {type(exc).__name__}: {exc}"])

        self.checkpoint_sha256 = (hashlib.sha256(candidate.read_bytes()).hexdigest()
                                  if candidate.exists() else None)
        meta_path = candidate.with_suffix(".json")
        self.checkpoint_metadata = None
        if meta_path.exists():
            try:
                self.checkpoint_metadata = json.loads(meta_path.read_text())
            except Exception as exc:
                self.checkpoint_metadata = {"error": f"unreadable metadata: {exc}"}

    # ------------------------------------------------------------------
    @property
    def mode(self) -> str:
        """"dqn" only if the checkpoint loaded AND passed the behavioural gate."""
        if not self.checkpoint_loaded or self.gate_report is None or not self.gate_report.passed:
            return "rule_based_fallback"
        return "dqn"

    def architecture(self) -> dict:
        net = self.agent.online_net
        layers = [{"in": m.in_features, "out": m.out_features}
                  for m in net.modules() if m.__class__.__name__ == "Linear"]
        return {"algorithm": "Double DQN", "network": "MLP 12-128-64-10 (ReLU)", "state_dim": ENV_STATE_DIM,
                "n_actions": N_ACTIONS, "linear_layers": layers, "device": str(self.agent.device),
                "deployment_epsilon": settings.rl.deployment_epsilon}

    def status(self) -> dict:
        reasons = []
        if not self.checkpoint_loaded:
            reasons.append(f"checkpoint not loaded ({self.load_error})")
        if self.gate_report is not None and not self.gate_report.passed:
            reasons.extend(self.gate_report.failures)
        mode = self.mode
        return {
            "policy_mode": mode,
            "dqn_making_decisions": mode == "dqn",
            "decision_maker": ("Double DQN (gated checkpoint)" if mode == "dqn"
                               else "rule-based fallback (transparent rules)"),
            "checkpoint_loaded": self.checkpoint_loaded,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_metadata": self.checkpoint_metadata,
            "architecture": self.architecture(),
            "behavioral_gate_enabled": self.gate_enabled,
            "behavioral_gate_config_requested": self.gate_config_requested,
            "behavioral_gate_passed": None if self.gate_report is None else self.gate_report.passed,
            "behavioral_gate_failures": [] if self.gate_report is None else list(self.gate_report.failures),
            "behavioral_gate_details": None if self.gate_report is None else self.gate_report.details,
            "fallback_reasons": reasons,
            "safety_layer_active": True,
        }

    def _q_values_batch(self, states: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            s = torch.as_tensor(np.asarray(states, dtype=np.float32), device=self.agent.device)
            return self.agent.online_net(s).cpu().numpy()

    # ------------------------------------------------------------------
    def select_action(self, state_vector: np.ndarray, epsilon: float | None = None,
                      session=None) -> PolicyDecision:
        """Pick the next action.

        Parameters
        ----------
        state_vector : np.ndarray (12,)
            The 12-D RL state for this attempt.
        epsilon : float | None
            Override of the deployment epsilon (default 0.0 = greedy).
        session : SessionContext | None
            When supplied (live API path) and the DQN is gated out, the
            intelligent rule-based tutor -- which sees the full session
            context -- makes the decision. When ``None`` (training /
            simulator path) the legacy ``RuleBasedAgent`` makes the
            decision so the RL training tests keep passing unchanged.

        The DQN path is unaffected: the gated checkpoint is queried with
        the state vector only, exactly as before. The behavioural gate
        is NOT bypassed -- ``mode`` still returns ``"rule_based_fallback"``
        when the checkpoint did not pass, and ``decision.source`` is
        ``"rule_based_fallback"``.
        """
        eps = settings.rl.deployment_epsilon if epsilon is None else epsilon
        state = np.asarray(state_vector, dtype=np.float32)

        problems = []
        if state.shape != (ENV_STATE_DIM,):
            problems.append(f"state shape {state.shape} != ({ENV_STATE_DIM},)")
        elif not np.all(np.isfinite(state)):
            problems.append("state contains NaN/inf")
        else:
            problems = validate_state_vector(state, strict=False)
        if problems:
            return PolicyDecision(CONTINUE_NO_CHANGE, ACTION_NAMES[CONTINUE_NO_CHANGE], [], True,
                                  source="safe_default_invalid_state", notes=problems)

        if self.mode != "dqn":
            # Use the intelligent tutor when a session is available
            # (live API path); otherwise fall back to the legacy
            # RuleBasedAgent for the simulator / training tests.
            if session is not None:
                a = int(self.intelligent_fallback.act(state, session=session))
            else:
                a = int(self.fallback.act(state))
            shadow: list[float] = []
            if self.checkpoint_loaded:
                # Logged for audit only ("what would the gated-out DQN
                # have preferred?"); never used to choose the action.
                try:
                    shadow = [float(q) for q in self._q_values_batch(state[None, :])[0]]
                except Exception:
                    shadow = []
            return PolicyDecision(a, ACTION_NAMES[a], shadow, True, source="rule_based_fallback",
                                  notes=["q_values are the gated-out DQN's (shadow, not used)"] if shadow else [])

        return self._dqn_decision(state, eps)

    def _dqn_decision(self, state: np.ndarray, eps: float,
                      session=None) -> PolicyDecision:
        """The DQN inference path: greedy argmax of the online network's
        Q-values (epsilon 0 in deployment). Any exception falls back to the
        rule-based policy for this one decision and says so. When a
        session is supplied, the intelligent tutor (with full session
        context) is preferred over the legacy agent for that fallback."""
        try:
            q_values = self._q_values_batch(state[None, :])[0]
            if not np.all(np.isfinite(q_values)):
                raise FloatingPointError("non-finite Q-values")
            action_id = self.agent.act(state, epsilon=eps)
        except Exception as exc:
            if session is not None:
                a = int(self.intelligent_fallback.act(state, session=session))
            else:
                a = int(self.fallback.act(state))
            return PolicyDecision(a, ACTION_NAMES[a], [], True,
                                  source="rule_based_after_inference_error",
                                  notes=[f"{type(exc).__name__}: {exc}"])
        return PolicyDecision(
            action_id=action_id,
            action_name=ACTION_NAMES[action_id],
            q_values=[float(q) for q in q_values],
            greedy=(eps == 0.0),
            source="dqn",
        )


@lru_cache(maxsize=1)
def get_policy_service() -> PolicyService:
    return PolicyService()
