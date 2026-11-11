/// Mirrors the backend's `SessionSummaryResponse`. Every number shown on the
/// progress / completion screens comes from here. Nothing is estimated or
/// filled in on the client.
class SessionSummary {
  final String sessionId;
  final int attemptCount;
  final List<double> scoreHistory;
  final int difficultyLevel;
  final int breakCount;
  final bool ended;

  const SessionSummary({
    required this.sessionId,
    required this.attemptCount,
    required this.scoreHistory,
    required this.difficultyLevel,
    required this.breakCount,
    required this.ended,
  });

  factory SessionSummary.fromJson(Map<String, dynamic> json) => SessionSummary(
        sessionId: json['session_id'] as String,
        attemptCount: json['attempt_count'] as int,
        scoreHistory: (json['score_history'] as List<dynamic>).map((v) => (v as num).toDouble()).toList(),
        difficultyLevel: json['difficulty_level'] as int,
        breakCount: (json['break_count'] as int?) ?? 0,
        ended: json['ended'] as bool,
      );
}

/// Backend `/health` response: is the server up, and which policy is live.
///
/// Shown to the supervising adult exactly as reported: the app never infers
/// or upgrades the mode. "dqn" is shown as the learned policy only when the
/// backend also reports that the DQN is making the decisions.
class BackendHealth {
  final bool ok;
  final String policyMode; // "dqn" or "rule_based_fallback"
  final List<String> fallbackReasons;

  /// True only when the backend says the gated DQN makes the decisions.
  final bool dqnMakingDecisions;

  /// Behavioural gate result (null when no checkpoint was gated).
  final bool? gatePassed;
  final List<String> gateFailures;
  final String? checkpointSha256;
  final String? checkpointPath;

  const BackendHealth({
    required this.ok,
    required this.policyMode,
    required this.fallbackReasons,
    this.dqnMakingDecisions = false,
    this.gatePassed,
    this.gateFailures = const [],
    this.checkpointSha256,
    this.checkpointPath,
  });

  /// The learned policy is shown as active only when BOTH fields agree.
  bool get learnedPolicyActive => policyMode == 'dqn' && dqnMakingDecisions;

  factory BackendHealth.fromJson(Map<String, dynamic> json) {
    final policy = (json['policy'] as Map?)?.cast<String, dynamic>() ?? const <String, dynamic>{};
    List<String> strings(Object? v) => ((v as List?) ?? const []).map((e) => e.toString()).toList();
    return BackendHealth(
      ok: json['status'] == 'ok',
      policyMode: (policy['policy_mode'] as String?) ?? 'unknown',
      fallbackReasons: strings(policy['fallback_reasons']),
      dqnMakingDecisions: policy['dqn_making_decisions'] == true,
      gatePassed: policy['behavioral_gate_passed'] is bool ? policy['behavioral_gate_passed'] as bool : null,
      gateFailures: strings(policy['behavioral_gate_failures']),
      checkpointSha256: policy['checkpoint_sha256'] is String ? policy['checkpoint_sha256'] as String : null,
      checkpointPath: policy['checkpoint_path'] is String ? policy['checkpoint_path'] as String : null,
    );
  }
}
