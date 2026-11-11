import 'package:flutter/material.dart';

import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_components.dart';
import 'exercise_selection_screen.dart';

/// Home screen. Explains in one glance what the app does, then the
/// supervising adult enters the learner's pseudonymous code and continues to
/// exercise selection. No network call is made until the adult presses
/// Start Session.
class SessionScreen extends StatefulWidget {
  final ApiService api;

  const SessionScreen({super.key, required this.api});

  @override
  State<SessionScreen> createState() => _SessionScreenState();
}

/// Same pattern the backend enforces (STUDY_LEARNER_ID_PATTERN): a short
/// code, never a real name or e-mail address.
final RegExp kLearnerCodePattern = RegExp(r'^[A-Za-z0-9][A-Za-z0-9_-]{2,39}$');

class _SessionScreenState extends State<SessionScreen> {
  final _childIdController = TextEditingController(text: 'demo_child_1');
  String? _fieldError;

  @override
  void dispose() {
    _childIdController.dispose();
    super.dispose();
  }

  void _start() {
    final code = _childIdController.text.trim();
    if (!kLearnerCodePattern.hasMatch(code)) {
      setState(() => _fieldError = 'Use a short code like P-0042 (letters, numbers, - or _), not a name.');
      return;
    }
    setState(() => _fieldError = null);
    Navigator.of(context).push(MaterialPageRoute(
      settings: const RouteSettings(name: kExercisesRouteName),
      builder: (_) => ExerciseSelectionScreen(api: widget.api, childId: code),
    ));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: LayoutBuilder(builder: (context, c) {
        final wide = c.maxWidth >= AppLayout.wideBreakpoint;
        final gutter = AppSpacing.gutter(c.maxWidth);
        final form = _StartPanel(
          controller: _childIdController,
          fieldError: _fieldError,
          onStart: _start,
        );
        Widget constrained(Widget child, {EdgeInsets? padding}) => Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: AppLayout.maxContentWidth),
                child: Padding(padding: padding ?? EdgeInsets.symmetric(horizontal: gutter), child: child),
              ),
            );
        return SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              AppHeroBand(
                child: SafeArea(
                  bottom: false,
                  child: constrained(
                    padding: EdgeInsets.fromLTRB(
                        gutter, wide ? AppSpacing.xxl : AppSpacing.lg, gutter, wide ? AppSpacing.xxl : AppSpacing.xl),
                    wide
                        ? Row(
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              const Expanded(flex: 6, child: _Story(wide: true)),
                              const SizedBox(width: AppSpacing.xxl),
                              Expanded(flex: 5, child: form),
                            ],
                          )
                        : const _Story(wide: false),
                  ),
                ),
              ),
              if (!wide) constrained(form, padding: EdgeInsets.fromLTRB(gutter, AppSpacing.lg, gutter, 0)),
              constrained(
                const _HowItWorks(),
                padding: EdgeInsets.fromLTRB(gutter, AppSpacing.xl, gutter, AppSpacing.xxl),
              ),
            ],
          ),
        );
      }),
    );
  }
}

/// Product mark, name, one-line promise and who it is for. White text on the
/// navy hero band (contrast checked in tests).
class _Story extends StatelessWidget {
  final bool wide;

  const _Story({required this.wide});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            ExcludeSemantics(
              child: Container(
                width: 52,
                height: 52,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  boxShadow: AppShadows.raised,
                ),
                child: const Icon(Icons.draw_rounded, size: 30, color: AppColors.primary),
              ),
            ),
            const SizedBox(width: AppSpacing.md),
            Flexible(
              child: Semantics(
                header: true,
                child: Text('Dysgraphia Tutor',
                    style: text.titleLarge?.copyWith(color: Colors.white, fontWeight: FontWeight.w700)),
              ),
            ),
          ],
        ),
        SizedBox(height: wide ? AppSpacing.xl : AppSpacing.lg),
        Text(
          'Handwriting practice that adapts after every try.',
          style: (wide ? text.displaySmall?.copyWith(fontSize: 40) : text.headlineMedium)
              ?.copyWith(color: Colors.white, height: 1.12),
        ),
        const SizedBox(height: AppSpacing.md),
        Text(
          'Adaptive handwriting practice for children who find writing hard. Watch a letter being written, '
          'trace it, then write it on your own. After each try the tutor suggests the next step.',
          style: text.bodyLarge?.copyWith(color: AppColors.onNavyMuted),
        ),
        const SizedBox(height: AppSpacing.lg),
        const Wrap(
          spacing: AppSpacing.sm,
          runSpacing: AppSpacing.sm,
          children: [
            _Pillar(icon: Icons.play_circle_outline_rounded, label: 'Watch'),
            _Pillar(icon: Icons.gesture_rounded, label: 'Trace'),
            _Pillar(icon: Icons.edit_rounded, label: 'Write'),
            _Pillar(icon: Icons.tune_rounded, label: 'Adapt'),
          ],
        ),
      ],
    );
  }
}

class _Pillar extends StatelessWidget {
  final IconData icon;
  final String label;

  const _Pillar({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.10),
          borderRadius: BorderRadius.circular(999),
          border: Border.all(color: Colors.white.withValues(alpha: 0.22)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, size: 18, color: Colors.white),
          const SizedBox(width: 6),
          Text(label, style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: Colors.white)),
        ]),
      );
}

class _StartPanel extends StatelessWidget {
  final TextEditingController controller;
  final String? fieldError;
  final VoidCallback onStart;

  const _StartPanel({required this.controller, required this.fieldError, required this.onStart});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return AppPanel(
      padding: const EdgeInsets.all(AppSpacing.lg),
      raised: true,
      borderColor: Colors.transparent,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        mainAxisSize: MainAxisSize.min,
        children: [
          Semantics(header: true, child: Text("Let's practise writing!", style: text.headlineSmall)),
          const SizedBox(height: AppSpacing.xs),
          Text('Take your time. You can take a break whenever you need one.',
              style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
          const SizedBox(height: AppSpacing.lg),
          TextField(
            controller: controller,
            style: const TextStyle(fontSize: 19),
            textInputAction: TextInputAction.go,
            onSubmitted: (_) => onStart(),
            decoration: InputDecoration(
              labelText: 'Child ID',
              helperText: 'For the adult: a code, not a name (e.g. P-0042)',
              errorText: fieldError,
              prefixIcon: const Icon(Icons.badge_outlined),
            ),
          ),
          const SizedBox(height: AppSpacing.lg),
          ElevatedButton(
            key: const Key('start-session'),
            onPressed: onStart,
            child: const Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Flexible(child: Text('Start Session')),
                SizedBox(width: AppSpacing.sm),
                Icon(Icons.arrow_forward_rounded),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HowItWorks extends StatelessWidget {
  const _HowItWorks();

  static const _steps = [
    (Icons.grid_view_rounded, 'Choose', 'Pick a letter, a word or a shape.'),
    (Icons.play_circle_outline_rounded, 'Watch and trace', 'See the strokes in order, then trace over the guide.'),
    (Icons.edit_rounded, 'Write on your own', 'Try it without the guide on lined paper.'),
    (
      Icons.tune_rounded,
      'Next step',
      'After each try the tutor suggests what to do next: a hint, a bigger guide, an easier or harder exercise, '
          'or a short break.'
    ),
  ];

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return LayoutBuilder(builder: (context, c) {
      final cols = c.maxWidth >= 900 ? 4 : (c.maxWidth >= 560 ? 2 : 1);
      final w = (c.maxWidth - AppSpacing.md * (cols - 1)) / cols;
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const AppEyebrow('How it works', color: AppColors.secondary),
          const SizedBox(height: AppSpacing.xs),
          Semantics(header: true, child: Text('How practice works', style: text.headlineSmall)),
          const SizedBox(height: AppSpacing.md),
          // Equal-height cards per row (IntrinsicHeight is cheap for 4 cards).
          for (var row = 0; row * cols < _steps.length; row++) ...[
            if (row > 0) const SizedBox(height: AppSpacing.md),
            IntrinsicHeight(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  for (var i = row * cols; i < (row + 1) * cols && i < _steps.length; i++) ...[
                    if (i > row * cols) const SizedBox(width: AppSpacing.md),
                    SizedBox(
                      width: w,
                      child: _StepCard(index: i, icon: _steps[i].$1, title: _steps[i].$2, body: _steps[i].$3),
                    ),
                  ],
                ],
              ),
            ),
          ],
        ],
      );
    });
  }
}

class _StepCard extends StatelessWidget {
  final int index;
  final IconData icon;
  final String title;
  final String body;

  const _StepCard({required this.index, required this.icon, required this.title, required this.body});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Container(
      padding: const EdgeInsets.all(AppSpacing.md),
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(AppRadii.lg),
        border: Border.all(color: AppColors.hairline),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 44,
            height: 44,
            decoration: BoxDecoration(color: AppColors.secondarySoft, borderRadius: BorderRadius.circular(12)),
            child: Icon(icon, color: AppColors.secondary),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${index + 1}. $title', style: text.titleMedium),
                const SizedBox(height: 2),
                Text(body, style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
