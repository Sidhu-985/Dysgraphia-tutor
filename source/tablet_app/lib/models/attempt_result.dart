import 'exercise.dart';

/// Mirrors the backend's `SafetyOut` schema -- reported to the UI for
/// transparency (e.g. a therapist/debug view could show when the safety
/// layer overrode the policy), never used to drive UI logic beyond
/// display.
class SafetyInfo {
  final int rawActionId;
  final String rawActionName;
  final int safeActionId;
  final String safeActionName;
  final bool overridden;
  final List<String> reasons;

  SafetyInfo.fromJson(Map<String, dynamic> json)
      : rawActionId = json['raw_action_id'] as int,
        rawActionName = json['raw_action_name'] as String,
        safeActionId = json['safe_action_id'] as int,
        safeActionName = json['safe_action_name'] as String,
        overridden = json['overridden'] as bool,
        reasons = (json['reasons'] as List<dynamic>).map((e) => e as String).toList();
}

/// Mirrors the backend's `InterventionOut` schema -- this IS the thing the
/// UI renders to the child (a message + a type telling it which widget to
/// show: hint / break / difficulty-change / letter-rotation / etc).
class InterventionInfo {
  final int actionId;
  final String actionName;
  final String type;
  final String message;
  final Map<String, dynamic> params;

  InterventionInfo.fromJson(Map<String, dynamic> json)
      : actionId = json['action_id'] as int,
        actionName = json['action_name'] as String,
        type = json['type'] as String,
        message = json['message'] as String,
        params = Map<String, dynamic>.from(json['params'] as Map);
}

/// Audit-driven (2026-11): per-letter mastery snapshot returned by the
/// backend. Tells the Flutter app which letters at the current difficulty
/// level have been mastered and how many remain, so the UI can show
/// progress without exposing internal policy labels. The learner-facing
/// UI only ever shows the friendly counts; the backend keeps the actual
/// mastery tracking.
class LetterMasteryInfo {
  final String currentExerciseId;
  final String? currentLetter;
  final int currentDifficultyLevel;
  final bool currentLetterMastered;
  final int masteredCountAtLevel;
  final int totalAtLevel;
  final int remainingAtLevel;
  final String nextExerciseId;
  final String? nextLetter;
  final bool allAtLevelMastered;

  LetterMasteryInfo({
    required this.currentExerciseId,
    required this.currentLetter,
    required this.currentDifficultyLevel,
    required this.currentLetterMastered,
    required this.masteredCountAtLevel,
    required this.totalAtLevel,
    required this.remainingAtLevel,
    required this.nextExerciseId,
    required this.nextLetter,
    required this.allAtLevelMastered,
  });

  factory LetterMasteryInfo.fromJson(Map<String, dynamic> json) =>
      LetterMasteryInfo(
        currentExerciseId: json['current_exercise_id'] as String? ?? '',
        currentLetter: json['current_letter'] as String?,
        currentDifficultyLevel: (json['current_difficulty_level'] as num?)?.toInt() ?? 0,
        currentLetterMastered: json['current_letter_mastered'] as bool? ?? false,
        masteredCountAtLevel: (json['mastered_count_at_level'] as num?)?.toInt() ?? 0,
        totalAtLevel: (json['total_at_level'] as num?)?.toInt() ?? 0,
        remainingAtLevel: (json['remaining_at_level'] as num?)?.toInt() ?? 0,
        nextExerciseId: json['next_exercise_id'] as String? ?? '',
        nextLetter: json['next_letter'] as String?,
        allAtLevelMastered: json['all_at_level_mastered'] as bool? ?? false,
      );
}

/// Audit-driven (2026-11): per-component handwriting score breakdown
/// returned by the backend. The "score" field is the learner-facing
/// 0-100 score; the breakdown is for an optional therapist / debug view
/// and is NOT shown to the learner by default.
class HandwritingScoreInfo {
  final double score; // 0-100
  final double normalized; // 0-1
  final Map<String, dynamic> breakdown;

  HandwritingScoreInfo({
    required this.score,
    required this.normalized,
    required this.breakdown,
  });

  factory HandwritingScoreInfo.fromJson(Map<String, dynamic> json) =>
      HandwritingScoreInfo(
        score: (json['score'] as num?)?.toDouble() ?? 0.0,
        normalized: (json['normalized'] as num?)?.toDouble() ?? 0.0,
        breakdown: json['breakdown'] is Map
            ? Map<String, dynamic>.from(json['breakdown'] as Map)
            : const <String, dynamic>{},
      );
}

/// Mirrors the backend's `AttemptSubmitResponse` schema. The UI reads
/// `qualityScore` / `intervention` / `nextExercise` to update itself; the
/// raw feature dicts and state vector are included for an optional
/// debug/therapist view but are not required for normal operation.
class AttemptResult {
  final String sessionId;
  final String attemptId;
  final String exerciseId;
  final double qualityScore;
  final Map<String, dynamic> staticFeatures;
  final Map<String, dynamic> dynamicFeatures;
  final List<double> stateVector;
  final SafetyInfo safety;
  final InterventionInfo intervention;
  final double reward;
  final Map<String, dynamic> rewardComponents;
  final Exercise nextExercise;
  final Map<String, dynamic> sessionSummary;

  /// Session presentation after this decision: difficulty level, template
  /// scale and mode ("tracing" or "free_writing"). Null for old backends.
  final Map<String, dynamic>? presentation;

  /// Which policy made the decision ("dqn", "rule_based_fallback", ...).
  /// Audit-only: kept on the model so a supervisor view can show it, but
  /// the learner-facing FeedbackOverlay no longer renders this label.
  final Map<String, dynamic>? policy;

  /// True when the backend replayed an earlier response for a resent attempt.
  final bool duplicateSubmission;

  /// Service mode when this decision was made ("dqn" | "rule_based_fallback").
  /// Audit-only: see `policy` above.
  final String? policyMode;

  /// What actually made THIS decision before the safety layer: "dqn",
  /// "rule_based_fallback", "safe_default_invalid_state" or
  /// "rule_based_after_inference_error". Null for older backends.
  /// Audit-only: see `policy` above.
  final String? decisionSource;

  /// True when the independent safety layer replaced the policy's action.
  /// Audit-only.
  final bool? safetyOverridden;

  /// Audit-driven (2026-11): per-letter mastery snapshot. Lets a
  /// supervisor view show how many letters at the current level the
  /// learner has mastered and how many remain. NOT used to drive UI
  /// logic; the learner-facing UI only shows child-friendly counts via
  /// the FeedbackOverlay.
  final LetterMasteryInfo? letterMastery;

  /// Audit-driven (2026-11): per-component handwriting score breakdown
  /// (shape similarity, stroke count, aspect ratio, curvature,
  /// direction, baseline alignment, optional pressure stability).
  /// NOT shown to the learner by default; available for a therapist /
  /// debug view.
  final HandwritingScoreInfo? handwritingScore;

  AttemptResult.fromJson(Map<String, dynamic> json)
      : sessionId = json['session_id'] as String,
        attemptId = json['attempt_id'] as String,
        exerciseId = json['exercise_id'] as String,
        qualityScore = (json['quality_score'] as num).toDouble(),
        staticFeatures = Map<String, dynamic>.from(json['static_features'] as Map),
        dynamicFeatures = Map<String, dynamic>.from(json['dynamic_features'] as Map),
        stateVector = (json['state_vector'] as List<dynamic>).map((v) => (v as num).toDouble()).toList(),
        safety = SafetyInfo.fromJson(json['safety'] as Map<String, dynamic>),
        intervention = InterventionInfo.fromJson(json['intervention'] as Map<String, dynamic>),
        reward = (json['reward'] as num).toDouble(),
        rewardComponents = Map<String, dynamic>.from(json['reward_components'] as Map),
        nextExercise = Exercise.fromJson(json['next_exercise'] as Map<String, dynamic>),
        sessionSummary = Map<String, dynamic>.from(json['session_summary'] as Map),
        presentation = json['presentation'] == null ? null : Map<String, dynamic>.from(json['presentation'] as Map),
        policy = json['policy'] == null ? null : Map<String, dynamic>.from(json['policy'] as Map),
        duplicateSubmission = (json['duplicate_submission'] as bool?) ?? false,
        policyMode = json['policy_mode'] is String ? json['policy_mode'] as String : null,
        decisionSource = json['decision_source'] is String
            ? json['decision_source'] as String
            : ((json['policy'] is Map && (json['policy'] as Map)['source'] is String)
                ? (json['policy'] as Map)['source'] as String
                : null),
        safetyOverridden = json['safety_overridden'] is bool ? json['safety_overridden'] as bool : null,
        letterMastery = json['letter_mastery'] is Map
            ? LetterMasteryInfo.fromJson(Map<String, dynamic>.from(json['letter_mastery'] as Map))
            : null,
        handwritingScore = json['handwriting_score'] is Map
            ? HandwritingScoreInfo.fromJson(Map<String, dynamic>.from(json['handwriting_score'] as Map))
            : null;

  /// Template scale requested by the backend (1.0 when not reported).
  double get templateScale {
    final v = presentation?['template_scale'];
    return v is num ? v.toDouble() : 1.0;
  }

  /// Supervisor-facing description of who decided, exactly as reported.
  /// Audit-only: kept on the model so a future supervisor view can show
  /// it; the learner-facing FeedbackOverlay does NOT render this string.
  String get decisionLabel {
    final src = switch (decisionSource) {
      'dqn' => 'learned policy (Double DQN)',
      'rule_based_fallback' => 'rule-based fallback',
      'safe_default_invalid_state' => 'safe default (state could not be used)',
      'rule_based_after_inference_error' => 'rule-based fallback (model error)',
      null => 'not reported',
      _ => decisionSource!,
    };
    return safetyOverridden == true ? '$src, changed by the safety rules' : src;
  }

  /// "tracing" (guide shown) or "free_writing" (guide hidden while writing).
  String get mode {
    final v = presentation?['mode'];
    return v is String ? v : 'tracing';
  }
}
