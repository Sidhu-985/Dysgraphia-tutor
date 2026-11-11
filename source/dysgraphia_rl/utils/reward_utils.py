"""
Reward Computation Module
==========================
Standalone, testable implementation of the reward used by the simulator,
the offline evaluation scripts and the live backend's transition logger.

Version 2 (current default, introduced after the Phase A audit -- see
docs/RL_AUDIT_AND_FIXES.md):

    R_t = w1*quality_improvement + w2*completion + w3*engagement
          - w4*frustration + w5*consistency - w6*unnecessary_break

where
  * quality_improvement = q_t - q_start, the CUMULATIVE improvement of the
    most recent observed quality proxy relative to the learner's starting
    level in this session. (Version 1 used the per-step delta
    q_t - q_{t-1}; summed over an episode that telescopes to
    q_final - q_start, i.e. at most a few reward units, while the per-step
    engagement/frustration terms accumulate over every step. That made a
    policy that never lets the child write -- always SUGGEST_BREAK --
    score ~49 per episode with zero quality gain. See Phase A evidence.)
  * unnecessary_break in [0, 1] is > 0 only when a break is taken while the
    fatigue proxy is below the break threshold (0 when fatigue is at or
    above it), scaled linearly by how far below the threshold it was.
  * w1 remains the LARGEST weight (enforced by `validate_weights` and
    tests), so genuine handwriting improvement dominates the objective.

The function keeps its original positional signature (the first argument is
still called `delta_quality` for backward compatibility with existing
callers/tests); the caller decides which quality-improvement signal is
passed. The simulator and backend both pass the cumulative improvement in
reward version 2.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from typing import Iterable, Optional


@dataclass
class RewardWeights:
    w1_quality: float = 3.0       # highest weight: genuine skill improvement
    w2_completion: float = 1.0
    w3_engagement: float = 1.0
    w4_frustration: float = 1.5
    w5_consistency: float = 0.5
    # v2: penalty for a break taken while the learner is not fatigued.
    # Default 0.0 keeps the dataclass usable for the original (v1) formula;
    # reward version 2 callers use `RewardWeights.v2()`.
    w6_unnecessary_break: float = 0.0

    @classmethod
    def v2(cls, **overrides) -> "RewardWeights":
        base = dict(w6_unnecessary_break=1.5)
        base.update(overrides)
        return cls(**base)


def validate_weights(weights: RewardWeights) -> None:
    """Raise ValueError if any weight is negative or if w1 (quality) is not
    strictly the largest weight. This is a design constraint, not a tuning
    preference: the quality term must dominate so the objective cannot be
    maximized by engagement/frustration management alone."""
    values = {f.name: getattr(weights, f.name) for f in fields(weights)}
    for name, v in values.items():
        if v < 0:
            raise ValueError(f"reward weight {name}={v} must be non-negative")
    others = [v for k, v in values.items() if k != "w1_quality"]
    if not all(values["w1_quality"] > v for v in others):
        raise ValueError(f"w1_quality must be strictly the largest reward weight, got {values}")


def unnecessary_break_penalty(break_taken: bool, fatigue_before: float,
                              fatigue_threshold: float = 0.5) -> float:
    """0 if no break was taken or the learner was at/above the fatigue
    threshold; otherwise (threshold - fatigue)/threshold in (0, 1]."""
    if not break_taken or fatigue_threshold <= 0:
        return 0.0
    gap = fatigue_threshold - float(fatigue_before)
    return max(0.0, min(1.0, gap / fatigue_threshold))


def compute_reward(delta_quality: float, completion: float, engagement: float,
                    frustration: float, consistency: float,
                    weights: Optional[RewardWeights] = None,
                    unnecessary_break: float = 0.0) -> float:
    # Fresh default instance (avoid a shared mutable default argument).
    if weights is None:
        weights = RewardWeights()
    return (
        weights.w1_quality * delta_quality
        + weights.w2_completion * completion
        + weights.w3_engagement * engagement
        - weights.w4_frustration * frustration
        + weights.w5_consistency * consistency
        - weights.w6_unnecessary_break * unnecessary_break
    )


# Actions in which the child does not write during the step. In the
# simulator only SUGGEST_BREAK (7) skips the writing trial.
DEFAULT_NON_WRITING_ACTIONS: tuple[int, ...] = (7,)


def audit_action_distribution(action_log: list[int], n_actions: int,
                               easy_actions: tuple[int, ...] = (0,),
                               mean_quality: float | None = None,
                               quality_threshold: float = 0.35,
                               quality_gain: float | None = None,
                               non_writing_actions: Iterable[int] = DEFAULT_NON_WRITING_ACTIONS,
                               dominance_threshold: float = 0.5,
                               max_non_writing_fraction: float = 0.25) -> dict:
    """Reward-hacking guard.

    Original rule (kept unchanged for backward compatibility): flag a
    policy that leans on 'make it easier' actions (> 50%) WITHOUT genuine
    quality (mean_quality below `quality_threshold`).

    Generalized rules (added after the Phase A audit, which showed the
    original audit only watched DECREASE_DIFFICULTY and therefore missed a
    policy choosing SUGGEST_BREAK ~45% of the time with zero quality gain):
      * `non_writing_excess`: non-writing actions (breaks) exceed
        `max_non_writing_fraction` of all actions;
      * `single_action_dominance`: any one action exceeds
        `dominance_threshold` of all actions;
      * `no_quality_gain`: `quality_gain` (final - starting quality) was
        supplied and is <= 0.
    A policy is flagged if (easy-action dominance OR non-writing excess OR
    single-action dominance) coincides with a quality failure (mean quality
    below threshold, or non-positive quality gain), or if the non-writing
    fraction alone exceeds 50% (a tutor that mostly does not let the child
    write cannot be teaching handwriting, whatever the quality signal says).

    Frequency alone is still not treated as proof of hacking: a policy that
    uses an action heavily AND delivers a positive quality gain is not
    flagged by the dominance rules.
    """
    non_writing_actions = tuple(non_writing_actions)
    if not action_log:
        return {"frequencies": {}, "easy_action_fraction": 0.0,
                "non_writing_fraction": 0.0, "dominant_action": None,
                "dominant_fraction": 0.0, "mean_quality": mean_quality,
                "quality_gain": quality_gain, "flags": [],
                "reward_hacking_suspected": False}

    counts = {a: 0 for a in range(n_actions)}
    for a in action_log:
        counts[a] += 1
    total = len(action_log)
    freqs = {a: c / total for a, c in counts.items()}

    easy_fraction = sum(freqs.get(a, 0.0) for a in easy_actions)
    non_writing_fraction = sum(freqs.get(a, 0.0) for a in non_writing_actions)
    dominant_action = max(freqs, key=freqs.get)
    dominant_fraction = freqs[dominant_action]

    quality_low = mean_quality is not None and mean_quality < quality_threshold
    no_gain = quality_gain is not None and quality_gain <= 0.0
    quality_failure = quality_low or no_gain
    quality_unknown = mean_quality is None and quality_gain is None

    flags: list[str] = []
    if easy_fraction > 0.5:
        flags.append("easy_action_dominance")
    if non_writing_fraction > max_non_writing_fraction:
        flags.append("non_writing_excess")
    if dominant_fraction > dominance_threshold:
        flags.append("single_action_dominance")
    if no_gain:
        flags.append("no_quality_gain")
    if quality_low:
        flags.append("low_mean_quality")

    # Original easy-action rule, unchanged semantics.
    if mean_quality is None and quality_gain is None:
        suspected = easy_fraction > 0.5
    else:
        suspected = easy_fraction > 0.5 and quality_failure

    behavioural = {"easy_action_dominance", "non_writing_excess", "single_action_dominance"}
    if any(f in behavioural for f in flags) and (quality_failure or quality_unknown):
        suspected = suspected or ("non_writing_excess" in flags or "single_action_dominance" in flags
                                  or easy_fraction > 0.5)
    if non_writing_fraction > 0.5:
        suspected = True

    return {"frequencies": freqs, "easy_action_fraction": easy_fraction,
            "non_writing_fraction": non_writing_fraction,
            "dominant_action": dominant_action, "dominant_fraction": dominant_fraction,
            "mean_quality": mean_quality, "quality_gain": quality_gain,
            "flags": flags, "reward_hacking_suspected": bool(suspected)}
