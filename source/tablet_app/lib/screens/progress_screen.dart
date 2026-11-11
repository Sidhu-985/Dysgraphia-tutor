import 'package:flutter/material.dart';

import '../models/session_summary.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_components.dart';
import '../widgets/score_chart.dart';

/// This session's practice scores, exactly as recorded by the backend.
/// With fewer than two attempts there is nothing to compare, so the screen
/// says so instead of drawing a trend.
class ProgressScreen extends StatefulWidget {
  final ApiService api;
  final String sessionId;

  const ProgressScreen({super.key, required this.api, required this.sessionId});

  @override
  State<ProgressScreen> createState() => _ProgressScreenState();
}

class _ProgressScreenState extends State<ProgressScreen> {
  // Created once (and again only on "Try again"), not on every rebuild.
  late Future<SessionSummary> _summary = widget.api.sessionSummary(widget.sessionId);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('My practice')),
      body: SafeArea(
        child: FutureBuilder<SessionSummary>(
          future: _summary,
          builder: (context, snap) {
            if (snap.connectionState != ConnectionState.done) {
              return const AppLoadingState(message: 'Loading your practice...');
            }
            if (snap.hasError) {
              return Center(
                child: Padding(
                  padding: const EdgeInsets.all(AppSpacing.lg),
                  child: ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: AppLayout.maxFormWidth),
                    child: AppStatusBanner.error(
                      title: 'Practice summary not available',
                      message: describeError(snap.error!),
                      actionLabel: 'Try again',
                      onAction: () => setState(() {
                        _summary = widget.api.sessionSummary(widget.sessionId);
                      }),
                    ),
                  ),
                ),
              );
            }
            return ProgressBody(summary: snap.data!);
          },
        ),
      ),
    );
  }
}

class ProgressBody extends StatelessWidget {
  final SessionSummary summary;

  const ProgressBody({super.key, required this.summary});

  @override
  Widget build(BuildContext context) {
    final scores = summary.scoreHistory;
    return LayoutBuilder(builder: (context, c) {
      final gutter = AppSpacing.gutter(c.maxWidth);
      return Center(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: AppLayout.maxContentWidth),
          child: Padding(
            padding: EdgeInsets.all(gutter),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                AppStatRow(stats: [
                  ('Attempts', '${summary.attemptCount}', Icons.edit_rounded),
                  ('Breaks taken', '${summary.breakCount}', Icons.self_improvement_rounded),
                  ('Level now', '${summary.difficultyLevel}', Icons.signal_cellular_alt_rounded),
                ]),
                const SizedBox(height: AppSpacing.lg),
                Expanded(
                  child: AppPanel(
                    padding: const EdgeInsets.all(AppSpacing.lg),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const AppSectionHeader(
                          title: 'Practice score for each attempt',
                          subtitle: 'Shape-match estimates from this app, not test results.',
                          icon: Icons.show_chart_rounded,
                        ),
                        const SizedBox(height: AppSpacing.md),
                        Expanded(
                          child: scores.length < 2
                              ? AppEmptyState(
                                  icon: scores.isEmpty ? Icons.edit_note_rounded : Icons.looks_one_outlined,
                                  title: scores.isEmpty
                                      ? 'No attempts yet. Your scores will appear here.'
                                      : 'One attempt so far. Do another one to see your progress.',
                                )
                              : ScoreChart(scores: scores),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
    });
  }
}
