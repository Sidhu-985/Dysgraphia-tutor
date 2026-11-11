import 'package:flutter/material.dart';

import '../models/session_summary.dart';
import '../theme/app_theme.dart';
import 'app_components.dart';

/// Supervisor-facing status of the tutor's decision policy, exactly as the
/// backend's `/health` reports it. The "learned policy" badge appears only
/// when the backend reports policy_mode "dqn" AND that the DQN is making the
/// decisions. Otherwise the card says the rule-based fallback is deciding and
/// lists the backend's reasons.
///
/// The card is collapsed by default (one line for the adult); the details
/// open on demand so they do not crowd the learner's screen.
class PolicyStatusCard extends StatelessWidget {
  final BackendHealth? health;
  final String? error;

  const PolicyStatusCard({super.key, required this.health, this.error});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final h = health;
    final String title;
    final String subtitle;
    final String summary;
    final IconData icon;
    final Color tint;
    final Color tintSoft;
    if (h == null) {
      title = error == null ? 'Checking the tutor server...' : 'Tutor server status unknown';
      subtitle = error ?? '';
      summary = subtitle;
      icon = error == null ? Icons.hourglass_empty_rounded : Icons.cloud_off_rounded;
      tint = error == null ? AppColors.inkMuted : AppColors.errorInk;
      tintSoft = error == null ? AppColors.bg : AppColors.errorSoft;
    } else if (h.learnedPolicyActive) {
      title = 'Decisions: learned policy (Double DQN)';
      subtitle = 'The checkpoint passed the behavioural safety gate. Safety rules are always on.';
      summary = 'Passed the safety gate. Safety rules always on.';
      icon = Icons.verified_rounded;
      tint = AppColors.secondary;
      tintSoft = AppColors.secondarySoft;
    } else {
      title = 'Decisions: rule-based fallback';
      subtitle = h.gatePassed == false
          ? 'The learned model did not pass the behavioural safety gate, so transparent rules decide. '
              'Safety rules are always on.'
          : 'The learned model is not in use. Transparent rules decide. Safety rules are always on.';
      summary = h.gatePassed == false
          ? 'Learned model did not pass the safety gate. Safety rules always on.'
          : 'Learned model not in use. Safety rules always on.';
      icon = Icons.rule_rounded;
      tint = AppColors.primary;
      tintSoft = AppColors.primarySoft;
    }
    final reasons = h == null ? const <String>[] : (h.fallbackReasons.isNotEmpty ? h.fallbackReasons : h.gateFailures);
    final detail = text.bodyMedium?.copyWith(fontSize: 16);
    return Semantics(
      container: true,
      label: 'For the adult. $title. $subtitle',
      child: Material(
        key: const Key('policy-status'),
        color: AppColors.paper,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(AppRadii.lg),
          side: const BorderSide(color: AppColors.hairline, width: 1.5),
        ),
        clipBehavior: Clip.antiAlias,
        child: ExcludeSemantics(
          child: Theme(
            data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
            child: ExpansionTile(
              tilePadding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: 2),
              childrenPadding: const EdgeInsets.fromLTRB(AppSpacing.md, 0, AppSpacing.md, AppSpacing.md),
              leading: Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(color: tintSoft, borderRadius: BorderRadius.circular(12)),
                child: Icon(icon, color: tint),
              ),
              title: Wrap(
                spacing: AppSpacing.sm,
                runSpacing: 4,
                crossAxisAlignment: WrapCrossAlignment.center,
                children: [
                  const AppTag(label: 'For the adult', background: AppColors.bg, foreground: AppColors.inkMuted),
                  Text(title, style: text.titleMedium?.copyWith(color: AppColors.ink)),
                ],
              ),
              subtitle: Padding(
                padding: const EdgeInsets.only(top: 2),
                child: Text(summary, style: text.bodyMedium?.copyWith(color: AppColors.inkMuted, fontSize: 16)),
              ),
              expandedCrossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Divider(),
                const SizedBox(height: AppSpacing.sm),
                Text('For the adult. $subtitle', style: detail),
                if (h != null) ...[
                  const SizedBox(height: AppSpacing.sm),
                  Text('Server mode: ${h.policyMode}', style: detail),
                  Text('Behavioural gate: ${h.gatePassed == null ? 'not run' : (h.gatePassed! ? 'passed' : 'failed')}',
                      style: detail),
                  if (h.checkpointSha256 != null)
                    Text(
                        'Checkpoint SHA-256: ${h.checkpointSha256!.substring(0, h.checkpointSha256!.length.clamp(0, 12))}...',
                        style: detail),
                  if (reasons.isNotEmpty) ...[
                    const SizedBox(height: AppSpacing.xs),
                    Text('Reasons reported by the server:', style: detail?.copyWith(fontWeight: FontWeight.w600)),
                    for (final r in reasons) Text('- $r', style: detail?.copyWith(color: AppColors.inkMuted)),
                  ],
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
