import 'package:flutter/material.dart';

import '../models/session_summary.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_components.dart';
import 'exercise_selection_screen.dart' show kExercisesRouteName;

/// End-of-session screen. It praises effort and reports what happened
/// (attempts, breaks), exactly as the backend recorded it. It does not claim
/// the child's handwriting improved: a change between two practice scores in
/// one session is not evidence of that.
class CompletionScreen extends StatefulWidget {
  final ApiService api;
  final String sessionId;

  const CompletionScreen({super.key, required this.api, required this.sessionId});

  @override
  State<CompletionScreen> createState() => _CompletionScreenState();
}

class _CompletionScreenState extends State<CompletionScreen> {
  // Created once (and again only on "Try again"), not on every rebuild.
  late Future<SessionSummary> _summary = widget.api.sessionSummary(widget.sessionId);

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppColors.primarySoft, AppColors.bg],
          ),
        ),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(AppSpacing.lg),
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: AppLayout.maxFormWidth),
                child: AppPanel(
                  raised: true,
                  padding: EdgeInsets.all(MediaQuery.sizeOf(context).width < 480 ? AppSpacing.lg : AppSpacing.xl),
                  child: FutureBuilder<SessionSummary>(
                    future: _summary,
                    builder: (context, snap) {
                      // While a (re)try is running, show loading: FutureBuilder keeps
                      // the previous error in the snapshot until the new result arrives.
                      final loading = snap.connectionState != ConnectionState.done;
                      final s = loading ? null : snap.data;
                      return Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          ExcludeSemantics(
                            child: Center(
                              child: Container(
                                width: 96,
                                height: 96,
                                decoration: const BoxDecoration(color: AppColors.successSoft, shape: BoxShape.circle),
                                child: const Icon(Icons.emoji_events_rounded, size: 56, color: AppColors.success),
                              ),
                            ),
                          ),
                          const SizedBox(height: AppSpacing.md),
                          Semantics(
                            header: true,
                            child: Text('Great practising!', style: text.displaySmall, textAlign: TextAlign.center),
                          ),
                          const SizedBox(height: AppSpacing.xs),
                          Text('You finished this practice session.',
                              style: text.bodyLarge?.copyWith(color: AppColors.inkMuted), textAlign: TextAlign.center),
                          const SizedBox(height: AppSpacing.lg),
                          if (s != null) ...[
                            Text(
                              'You did ${s.attemptCount} ${s.attemptCount == 1 ? 'attempt' : 'attempts'}'
                              ' and took ${s.breakCount} ${s.breakCount == 1 ? 'break' : 'breaks'}.',
                              style: text.titleLarge,
                              textAlign: TextAlign.center,
                            ),
                            const SizedBox(height: AppSpacing.md),
                            AppStatRow(alignment: WrapAlignment.center, stats: [
                              ('Attempts', '${s.attemptCount}', Icons.edit_rounded),
                              ('Breaks', '${s.breakCount}', Icons.self_improvement_rounded),
                            ]),
                          ] else if (!loading && snap.hasError)
                            AppStatusBanner.error(
                              title: 'Summary not available',
                              message: '${describeError(snap.error!)} The session itself has ended.',
                              actionLabel: 'Try again',
                              onAction: () => setState(() {
                                _summary = widget.api.sessionSummary(widget.sessionId);
                              }),
                            )
                          else
                            const AppLoadingState(message: 'Loading your summary...'),
                          const SizedBox(height: AppSpacing.xl),
                          ElevatedButton.icon(
                            key: const Key('choose-another'),
                            // Back to the dashboard when it is in the stack,
                            // otherwise to the first screen (never past it).
                            onPressed: () => Navigator.of(context)
                                .popUntil((r) => r.settings.name == kExercisesRouteName || r.isFirst),
                            icon: const Icon(Icons.grid_view_rounded),
                            label: const Text('Choose another exercise'),
                          ),
                          const SizedBox(height: AppSpacing.sm),
                          OutlinedButton(
                            key: const Key('back-to-start'),
                            onPressed: () => Navigator.of(context).popUntil((r) => r.isFirst),
                            child: const Text('Back to start'),
                          ),
                        ],
                      );
                    },
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
