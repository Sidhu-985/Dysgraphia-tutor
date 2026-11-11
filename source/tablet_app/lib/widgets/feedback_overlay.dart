import 'package:flutter/material.dart';

import '../models/attempt_result.dart';
import '../models/exercise_instruction.dart';
import '../theme/app_theme.dart';
import 'app_components.dart';

/// Shows the backend's response to one attempt: the intervention message
/// the child sees, what the screen will do next, and the practice score.
///
/// The score is labelled as a **practice estimate of shape match**. It is
/// an engineering proxy (docs/QUALITY_SCORE.md), not a test result or a
/// clinical measure, and the UI must not present it as either. This widget
/// only maps `intervention.type` to an icon and to a plain description of the
/// action the writing screen carries out for that type. It does not decide
/// which intervention to show; the safety-checked policy on the server
/// already did.
///
/// Audit-driven (2026-11): internal implementation labels (Rule-based
/// fallback, rule_based_fallback, policy_mode, decision_source) are no
/// longer shown on this learner-facing card. The backend still reports
/// them in the API response for developer/audit use, and they remain in
/// the AttemptResult model for an optional supervisor view; this widget
/// simply does not render them. The "For the adult" line that previously
/// exposed `decisionLabel` has been removed; the supervisor-facing
/// `PolicyStatusCard` (collapsed by default) is the place where adults
/// see the audit information.
class FeedbackOverlay extends StatelessWidget {
  final AttemptResult result;
  final VoidCallback onDismiss;

  const FeedbackOverlay({super.key, required this.result, required this.onDismiss});

  static IconData iconFor(String type) {
    switch (type) {
      case 'hint':
        return Icons.lightbulb_outline;
      case 'break':
        return Icons.self_improvement;
      case 'difficulty_adjustment':
        return Icons.tune;
      case 'template_size':
        return Icons.zoom_in;
      case 'stroke_animation':
        return Icons.play_circle_outline;
      case 'mode_change':
        return Icons.swap_horiz;
      case 'repeat_exercise':
        return Icons.repeat;
      case 'letter_rotation':
        return Icons.swap_horiz;
      default:
        return Icons.thumb_up_alt_outlined;
    }
  }

  static String nextStepLabel(String type) {
    switch (type) {
      case 'hint':
        return 'Show me the hint';
      case 'break':
        return 'Start the break';
      case 'stroke_animation':
        return 'Watch it';
      case 'letter_rotation':
        return 'Try the next letter';
      default:
        return 'Next';
    }
  }

  /// What the writing screen does after "Next", in child-friendly words.
  /// Derived only from the intervention type and the next exercise and mode
  /// the backend sent; null when there is nothing to add to the message.
  static String? whatHappensNext(AttemptResult r) {
    final next = exerciseLabel(r.nextExercise);
    final how = r.mode == 'free_writing' ? 'on your own' : 'tracing';
    final upNext = next.isEmpty ? null : 'Up next: $next, $how.';
    switch (r.intervention.type) {
      case 'hint':
        return 'You will see where to start and which way to go.';
      case 'stroke_animation':
        return 'A moving dot will show how it is drawn.';
      case 'template_size':
        return 'The guide will be drawn bigger.';
      case 'break':
        return 'A short rest screen will open. You can skip it.';
      case 'letter_rotation':
        return 'You mastered this letter — well done! Now another letter at the same level.';
      default:
        return upNext;
    }
  }

  @override
  Widget build(BuildContext context) {
    final intervention = result.intervention;
    final text = Theme.of(context).textTheme;
    final score = result.qualityScore.clamp(0.0, 1.0).toDouble();
    final pct = (score * 100).round();
    final next = whatHappensNext(result);
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: AppMotion.duration(context, AppMotion.medium),
      curve: AppMotion.curve,
      builder: (context, t, child) => Opacity(
        opacity: t,
        child: Transform.translate(offset: Offset(0, 24 * (1 - t)), child: child),
      ),
      child: Semantics(
        liveRegion: true,
        container: true,
        child: Container(
          key: const Key('feedback-card'),
          margin: const EdgeInsets.all(AppSpacing.md),
          constraints: const BoxConstraints(maxWidth: 720),
          decoration: BoxDecoration(
            color: AppColors.paper,
            borderRadius: BorderRadius.circular(AppRadii.xl),
            border: Border.all(color: AppColors.secondary, width: 2),
            boxShadow: AppShadows.raised,
          ),
          child: LayoutBuilder(builder: (context, box) {
            final narrow = box.maxWidth < 560;
            final pad = narrow ? AppSpacing.md : AppSpacing.lg;
            return Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Content scrolls if the screen is short; the action below is
                // pinned so the learner can always continue.
                Flexible(
                  child: SingleChildScrollView(
                    padding: EdgeInsets.fromLTRB(pad, pad, pad, 0),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 56,
                              height: 56,
                              decoration: BoxDecoration(
                                  color: AppColors.secondarySoft, borderRadius: BorderRadius.circular(AppRadii.md)),
                              child: Icon(iconFor(intervention.type), size: 32, color: AppColors.secondary),
                            ),
                            const SizedBox(width: AppSpacing.md),
                            Expanded(
                                child:
                                    Text(intervention.message, style: narrow ? text.titleLarge : text.headlineSmall)),
                          ],
                        ),
                        const SizedBox(height: AppSpacing.md),
                        // 1. The backend's evaluation of this attempt.
                        Container(
                          key: const Key('feedback-evaluation'),
                          padding: const EdgeInsets.all(AppSpacing.md),
                          decoration:
                              BoxDecoration(color: AppColors.bg, borderRadius: BorderRadius.circular(AppRadii.md)),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const AppEyebrow('Your writing', icon: Icons.edit_rounded),
                              const SizedBox(height: AppSpacing.xs),
                              Text('Practice score (shape match): $pct%', style: text.titleMedium),
                              const SizedBox(height: AppSpacing.sm),
                              // The text above already states the value for screen
                              // readers; the bar is a visual repeat of it.
                              ExcludeSemantics(
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(8),
                                  child: TweenAnimationBuilder<double>(
                                    tween: Tween(begin: 0, end: score),
                                    duration: AppMotion.duration(context, AppMotion.slow),
                                    curve: AppMotion.curve,
                                    builder: (_, v, __) => LinearProgressIndicator(
                                      value: v,
                                      minHeight: 12,
                                      backgroundColor: AppColors.track,
                                      color: AppColors.secondary,
                                    ),
                                  ),
                                ),
                              ),
                              const SizedBox(height: AppSpacing.sm),
                              Text('A practice estimate from this app, not a test result.',
                                  style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
                            ],
                          ),
                        ),
                        const SizedBox(height: AppSpacing.sm),
                        // 2. The backend's recommendation (audit-driven:
                        // learner-facing only -- internal policy labels are
                        // no longer exposed here; they remain available to
                        // developers via the API response and the
                        // supervisor-facing PolicyStatusCard).
                        Container(
                          key: const Key('feedback-recommendation'),
                          padding: const EdgeInsets.all(AppSpacing.md),
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(AppRadii.md),
                            border: Border.all(color: AppColors.hairline),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const AppEyebrow('Next step', icon: Icons.east_rounded, color: AppColors.secondary),
                              if (next != null) ...[
                                const SizedBox(height: AppSpacing.xs),
                                Text(next, key: const Key('what-happens-next'), style: text.bodyLarge),
                              ],
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                Padding(
                  padding: EdgeInsets.fromLTRB(pad, AppSpacing.md, pad, pad),
                  child: Align(
                    alignment: narrow ? Alignment.center : Alignment.centerRight,
                    child: ElevatedButton(
                      key: const Key('feedback-next'),
                      onPressed: onDismiss,
                      style: ElevatedButton.styleFrom(
                          minimumSize: Size(narrow ? double.infinity : 0, AppSpacing.touchTarget)),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Flexible(child: Text(nextStepLabel(intervention.type), textAlign: TextAlign.center)),
                          const SizedBox(width: AppSpacing.sm),
                          const Icon(Icons.arrow_forward_rounded),
                        ],
                      ),
                    ),
                  ),
                ),
              ],
            );
          }),
        ),
      ),
    );
  }
}

/// Compact badge naming who chose the next step, exactly as the backend
/// reported it. "Learned policy" appears only for decision_source "dqn";
/// every other source is shown as what it is.
class DecisionBadge extends StatelessWidget {
  final AttemptResult result;

  const DecisionBadge({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    final (String label, IconData icon) = switch (result.decisionSource) {
      'dqn' => ('Learned policy', Icons.psychology_alt_outlined),
      'rule_based_fallback' => ('Rule-based fallback', Icons.rule_rounded),
      'rule_based_after_inference_error' => ('Rule-based (model error)', Icons.rule_rounded),
      'safe_default_invalid_state' => ('Safe default', Icons.shield_outlined),
      _ => (result.decisionSource ?? 'Not reported', Icons.help_outline_rounded),
    };
    return Container(
      key: const Key('decision-badge'),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(color: AppColors.bg, borderRadius: BorderRadius.circular(999)),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Icon(icon, size: 15, color: AppColors.inkMuted),
        const SizedBox(width: 4),
        Flexible(
          child:
              Text(label, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.inkMuted)),
        ),
        if (result.safetyOverridden == true) ...[
          const SizedBox(width: 6),
          const Icon(Icons.shield_outlined, size: 15, color: AppColors.inkMuted),
          const SizedBox(width: 2),
          const Flexible(
            child: Text('safety rules',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600, color: AppColors.inkMuted)),
          ),
        ],
      ]),
    );
  }
}
