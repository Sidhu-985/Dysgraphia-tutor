import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/alphabet.dart';
import '../models/exercise.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_components.dart';
import '../widgets/letter_glyph.dart';
import '../widgets/letter_guide.dart';
import 'writing_screen.dart';

/// One letter's learning page: look at the letter, learn its stroke order
/// (numbered start markers, direction arrows, "show me how"), then start a
/// real backend session to trace it or to write it without the guide.
/// The learner can switch between the small and capital form of the same
/// letter, and step through the sequence it was opened from (A-Z or a group),
/// but is never forced to.
class LetterDetailScreen extends StatefulWidget {
  final ApiService api;
  final String childId;
  final AlphabetIndex index;
  final Exercise initial;
  final List<String> sequence;
  final PracticeLog log;

  const LetterDetailScreen({
    super.key,
    required this.api,
    required this.childId,
    required this.index,
    required this.initial,
    required this.sequence,
    required this.log,
  });

  @override
  State<LetterDetailScreen> createState() => _LetterDetailScreenState();
}

class _LetterDetailScreenState extends State<LetterDetailScreen> with SingleTickerProviderStateMixin {
  late Exercise _ex;
  late List<String> _sequence;
  late final AnimationController _demo;
  String? _startingMode;
  String? _error;

  bool get _starting => _startingMode != null;

  @override
  void initState() {
    super.initState();
    _ex = widget.initial;
    _sequence = widget.sequence;
    // Only the guide and the stroke legend listen to the animation (through
    // AnimatedBuilder), so the rest of the page does not rebuild every frame.
    _demo = AnimationController(vsync: this, duration: const Duration(milliseconds: 2400));
  }

  @override
  void dispose() {
    _demo.dispose();
    super.dispose();
  }

  String get _char => _ex.target!;

  void _show(Exercise ex) {
    _demo.stop();
    _demo.reset();
    setState(() {
      _ex = ex;
      _error = null;
    });
  }

  /// Switch small <-> capital form of the SAME letter. The sequence switches
  /// to the same letters in the other case, so "next" keeps working.
  void _switchCase() {
    final other = widget.index.counterpart(_char);
    if (other == null) return;
    final toUpper = other.isUppercase;
    _sequence = [for (final s in _sequence) toUpper ? s.toUpperCase() : s.toLowerCase()]
        .where((s) => widget.index[s] != null)
        .toList();
    _show(other);
  }

  void _showMe() {
    if (AppMotion.reduced(context)) {
      _demo.value = 1.0; // no motion: show the finished, numbered letter at once
    } else {
      _demo.forward(from: 0);
    }
  }

  Future<void> _start(String practiceMode) async {
    setState(() {
      _startingMode = practiceMode;
      _error = null;
    });
    try {
      final resp =
          await widget.api.startSession(widget.childId, exerciseId: _ex.exerciseId, practiceMode: practiceMode);
      if (!mounted) return;
      final exercise = Exercise.fromJson(resp['exercise'] as Map<String, dynamic>);
      final mode = resp['practice_mode'] is String ? resp['practice_mode'] as String : 'tracing';
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => WritingScreen(
          api: widget.api,
          sessionId: resp['session_id'] as String,
          childId: widget.childId,
          initialExercise: exercise,
          initialMode: mode,
          onSessionEnded: () {},
          onAttemptAccepted: widget.log.record,
        ),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = describeError(e));
    } finally {
      if (mounted) setState(() => _startingMode = null);
    }
  }

  @override
  Widget build(BuildContext context) {
    final (prev, next) = neighbours(_sequence, _char);
    final other = widget.index.counterpart(_char);

    return Scaffold(
      appBar: AppBar(
        title: Text('Letter ${letterName(_char)}'),
        actions: [
          if (other != null)
            Padding(
              padding: const EdgeInsets.only(right: AppSpacing.md),
              child: Semantics(
                button: true,
                label: 'Switch to ${letterName(other.target!)}',
                child: ExcludeSemantics(
                  child: Tooltip(
                    message: 'Switch to ${letterName(other.target!)}',
                    child: OutlinedButton(
                      key: const Key('switch-case'),
                      onPressed: _switchCase,
                      // Local style for the navy app bar: white text and a light
                      // outline. The global OutlinedButton theme (dark text on
                      // white) is deliberately left unchanged.
                      style: OutlinedButton.styleFrom(
                        foregroundColor: Colors.white,
                        backgroundColor: Colors.white.withValues(alpha: 0.08),
                        side: BorderSide(color: Colors.white.withValues(alpha: 0.75), width: 2),
                        minimumSize: const Size(AppSpacing.touchTarget, 48),
                        padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: AppSpacing.xs),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          const Icon(Icons.swap_horiz_rounded, size: 22, color: Colors.white),
                          const SizedBox(width: 6),
                          Text(
                            '${_ex.isUppercase ? other.target : _char} / ${_ex.isUppercase ? _char : other.target}',
                            style: const TextStyle(
                              fontSize: 22,
                              fontWeight: FontWeight.w800,
                              color: Colors.white,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(builder: (context, c) {
          final wide = c.maxWidth >= AppLayout.wideBreakpoint;
          final guidePanel = _GuidePanel(
            exercise: _ex,
            char: _char,
            demo: _demo,
            onShowMe: _showMe,
          );
          final infoPanel = _InfoPanel(
            exercise: _ex,
            char: _char,
            error: _error,
            startingMode: _startingMode,
            onStart: _starting ? null : _start,
            pinActions: wide,
          );
          final nav = _SequenceNav(
            prev: prev,
            next: next,
            onPrev: prev == null ? null : () => _show(widget.index[prev]!),
            onNext: next == null ? null : () => _show(widget.index[next]!),
          );
          if (wide) {
            return Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1400),
                child: Padding(
                  padding: const EdgeInsets.all(AppSpacing.lg),
                  child: Column(
                    children: [
                      Expanded(
                        child: Row(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Expanded(flex: 3, child: guidePanel),
                            const SizedBox(width: AppSpacing.lg),
                            Expanded(flex: 2, child: infoPanel),
                          ],
                        ),
                      ),
                      const SizedBox(height: AppSpacing.md),
                      nav,
                    ],
                  ),
                ),
              ),
            );
          }
          return SingleChildScrollView(
            padding: const EdgeInsets.all(AppSpacing.md),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                SizedBox(height: (c.maxWidth * 1.3).clamp(400.0, 640.0).toDouble(), child: guidePanel),
                const SizedBox(height: AppSpacing.md),
                infoPanel,
                const SizedBox(height: AppSpacing.md),
                nav,
              ],
            ),
          );
        }),
      ),
    );
  }
}

/// The demonstration: the letter on four-line paper with numbered start
/// markers and arrows, "Show me how", and a legend of the strokes in order.
/// During the demonstration the stroke being drawn is highlighted.
class _GuidePanel extends StatelessWidget {
  final Exercise exercise;
  final String char;
  final AnimationController demo;
  final VoidCallback onShowMe;

  const _GuidePanel({required this.exercise, required this.char, required this.demo, required this.onShowMe});

  /// Index of the stroke being drawn at demo progress [p] (strokes share the
  /// time by length, exactly as LetterGuidePainter draws them). The guide is
  /// scaled uniformly, so template-unit lengths give the same fractions.
  static int activeStroke(List<List<List<double>>> strokes, double p) {
    double len(List<List<double>> s) {
      var l = 0.0;
      for (var i = 1; i < s.length; i++) {
        l += math.sqrt(math.pow(s[i][0] - s[i - 1][0], 2) + math.pow(s[i][1] - s[i - 1][1], 2));
      }
      return l;
    }

    final lengths = [
      for (final s in strokes)
        if (s.length >= 2) len(s)
    ];
    final total = lengths.fold<double>(0, (a, b) => a + b);
    if (total <= 0) return 0;
    var budget = p.clamp(0.0, 1.0) * total;
    for (var i = 0; i < lengths.length; i++) {
      if (budget <= lengths[i]) return i;
      budget -= lengths[i];
    }
    return lengths.length - 1;
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final strokes = exercise.guideStrokes;
    final n = strokes.length;
    return Container(
      decoration: BoxDecoration(
        color: AppColors.paper,
        borderRadius: BorderRadius.circular(AppRadii.xxl),
        border: Border.all(color: AppColors.hairline, width: 1.5),
        boxShadow: AppShadows.card,
      ),
      padding: const EdgeInsets.all(AppSpacing.md),
      child: AnimatedBuilder(
        animation: demo,
        builder: (context, _) {
          final playing = demo.isAnimating;
          final demoProgress = playing || demo.value > 0 ? demo.value : null;
          final active = playing ? activeStroke(strokes, demo.value) : null;
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              LayoutBuilder(builder: (context, hc) {
                final title = Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    AppEyebrow('$n ${n == 1 ? 'stroke' : 'strokes'} · in order',
                        color: AppColors.secondary, icon: Icons.format_list_numbered_rounded),
                    const SizedBox(height: 2),
                    Semantics(
                      header: true,
                      child: Text('How to write it', style: text.titleLarge),
                    ),
                  ],
                );
                final showMe = FilledButton.icon(
                  key: const Key('show-me'),
                  onPressed: playing ? null : onShowMe,
                  style: FilledButton.styleFrom(
                    backgroundColor: AppColors.secondary,
                    foregroundColor: Colors.white,
                    disabledBackgroundColor: AppColors.secondarySoft,
                    disabledForegroundColor: AppColors.ink,
                    minimumSize: const Size(0, 56),
                    padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                    textStyle: const TextStyle(fontFamily: kAppFontFamily, fontSize: 17, fontWeight: FontWeight.w600),
                  ),
                  icon: Icon(demo.value >= 1.0 ? Icons.replay_rounded : Icons.play_arrow_rounded),
                  label: Text(playing ? 'Watch...' : (demo.value >= 1.0 ? 'Show me again' : 'Show me how')),
                );
                // Narrow panels: the button goes under the title, full width.
                if (hc.maxWidth < 440) {
                  return Column(
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [title, const SizedBox(height: AppSpacing.sm), showMe],
                  );
                }
                return Row(children: [Expanded(child: title), const SizedBox(width: AppSpacing.sm), showMe]);
              }),
              const SizedBox(height: AppSpacing.sm),
              Expanded(
                child: LetterGuide(
                  key: const Key('letter-guide'),
                  strokes: strokes,
                  lines: exercise.writingLines,
                  progress: demoProgress,
                  semanticLabel: 'How to write ${letterName(char)}: $n '
                      '${n == 1 ? 'stroke' : 'strokes'}, start at the numbered dots and follow the arrows.',
                ),
              ),
              const SizedBox(height: AppSpacing.sm),
              Text('Start at the green numbered dot. Follow the arrows.',
                  textAlign: TextAlign.center, style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
              const SizedBox(height: AppSpacing.sm),
              Wrap(
                spacing: AppSpacing.sm,
                runSpacing: AppSpacing.sm,
                alignment: WrapAlignment.center,
                children: [
                  for (var i = 0; i < n; i++) _StrokeChip(index: i, active: active == i),
                ],
              ),
            ],
          );
        },
      ),
    );
  }
}

class _StrokeChip extends StatelessWidget {
  final int index;
  final bool active;

  const _StrokeChip({required this.index, required this.active});

  @override
  Widget build(BuildContext context) {
    final color = kStrokeColors[index % kStrokeColors.length];
    return AnimatedContainer(
      duration: AppMotion.duration(context, AppMotion.fast),
      padding: const EdgeInsets.fromLTRB(6, 6, 14, 6),
      decoration: BoxDecoration(
        color: active ? color.withValues(alpha: 0.14) : AppColors.paper,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: active ? color : AppColors.hairline, width: active ? 2.5 : 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircleAvatar(
            radius: 14,
            backgroundColor: color,
            child: Text('${index + 1}',
                style: const TextStyle(color: Colors.white, fontWeight: FontWeight.w800, fontSize: 15)),
          ),
          const SizedBox(width: AppSpacing.sm),
          Text('Stroke ${index + 1}', style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
        ],
      ),
    );
  }
}

class _InfoPanel extends StatelessWidget {
  final Exercise exercise;
  final String char;
  final String? error;
  final String? startingMode;
  final void Function(String mode)? onStart;
  final bool pinActions;

  const _InfoPanel({
    required this.exercise,
    required this.char,
    required this.error,
    required this.startingMode,
    required this.onStart,
    this.pinActions = false,
  });

  static const _journey = [
    (Icons.visibility_rounded, 'Look'),
    (Icons.format_list_numbered_rounded, 'Learn the strokes'),
    (Icons.gesture_rounded, 'Trace'),
    (Icons.edit_rounded, 'Write on your own'),
    (Icons.chat_bubble_outline_rounded, 'Feedback'),
  ];

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final zone = zoneDescription(exercise.letterZone);
    final info = <Widget>[
      AppPanel(
        padding: const EdgeInsets.all(AppSpacing.md),
        child: Row(
          children: [
            Container(
              width: 120,
              height: 120,
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(24)),
              child: AnimatedSwitcher(
                duration: AppMotion.duration(context, AppMotion.medium),
                child: KeyedSubtree(
                  key: ValueKey(char),
                  // The letter exactly as it is taught (from its strokes).
                  child: Semantics(
                    label: letterName(char),
                    child: LetterGlyph.of(exercise, key: const Key('reference-letter'), showLines: true),
                  ),
                ),
              ),
            ),
            const SizedBox(width: AppSpacing.md),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(letterName(char), style: text.headlineMedium),
                  if (zone.isNotEmpty) Text(zone, style: text.bodyLarge?.copyWith(color: AppColors.inkMuted)),
                ],
              ),
            ),
          ],
        ),
      ),
      if (exercise.letterInstruction != null) ...[
        const SizedBox(height: AppSpacing.md),
        AppPanel(
          color: AppColors.secondarySoft,
          borderColor: AppColors.secondarySoft,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Icon(Icons.lightbulb_outline_rounded, color: AppColors.secondary),
              const SizedBox(width: AppSpacing.sm),
              Expanded(child: Text(exercise.letterInstruction!, style: text.bodyLarge)),
            ],
          ),
        ),
      ],
      const SizedBox(height: AppSpacing.lg),
      const AppEyebrow('Your practice steps'),
      const SizedBox(height: AppSpacing.sm),
      const AppStepIndicator(steps: _journey, current: 1),
      const SizedBox(height: AppSpacing.lg),
    ];
    final actions = <Widget>[
      if (error != null) ...[
        AppStatusBanner.error(
          title: 'Practice could not start',
          message: error!,
          icon: Icons.wifi_off_rounded,
        ),
        const SizedBox(height: AppSpacing.md),
      ],
      ElevatedButton.icon(
        key: const Key('start-tracing'),
        onPressed: onStart == null ? null : () => onStart!('tracing'),
        icon: startingMode == 'tracing' ? const _ButtonSpinner() : const Icon(Icons.gesture_rounded),
        label: Text('Trace ${letterName(char)}'),
      ),
      const SizedBox(height: AppSpacing.sm),
      OutlinedButton.icon(
        key: const Key('start-free-writing'),
        onPressed: onStart == null ? null : () => onStart!('free_writing'),
        icon: startingMode == 'free_writing' ? const _ButtonSpinner() : const Icon(Icons.edit_rounded),
        label: const Text('Write it on my own'),
      ),
      if (startingMode != null)
        Padding(
          padding: const EdgeInsets.only(top: AppSpacing.sm),
          child: Semantics(
            liveRegion: true,
            child: Text('Starting practice...',
                textAlign: TextAlign.center, style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
          ),
        ),
    ];
    // Wide layout: the information scrolls and the two start buttons stay
    // pinned at the bottom of the column, always fully visible.
    if (pinActions) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Expanded(
            child: SingleChildScrollView(
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: info),
            ),
          ),
          const SizedBox(height: AppSpacing.md),
          ...actions,
        ],
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      mainAxisSize: MainAxisSize.min,
      children: [...info, ...actions],
    );
  }
}

class _ButtonSpinner extends StatelessWidget {
  const _ButtonSpinner();

  @override
  Widget build(BuildContext context) => const SizedBox(
        width: 22,
        height: 22,
        child: CircularProgressIndicator(strokeWidth: 3, color: AppColors.inkMuted),
      );
}

class _SequenceNav extends StatelessWidget {
  final String? prev;
  final String? next;
  final VoidCallback? onPrev;
  final VoidCallback? onNext;

  const _SequenceNav({required this.prev, required this.next, required this.onPrev, required this.onNext});

  @override
  Widget build(BuildContext context) => Row(
        children: [
          Expanded(
            child: OutlinedButton.icon(
              key: const Key('prev-letter'),
              onPressed: onPrev,
              icon: const Icon(Icons.chevron_left_rounded),
              label: Text(prev == null ? 'Previous' : 'Previous: $prev'),
            ),
          ),
          const SizedBox(width: AppSpacing.md),
          Expanded(
            child: OutlinedButton(
              key: const Key('next-letter'),
              onPressed: onNext,
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Flexible(child: Text(next == null ? 'Next' : 'Next: $next', overflow: TextOverflow.ellipsis)),
                  const SizedBox(width: AppSpacing.xs),
                  const Icon(Icons.chevron_right_rounded),
                ],
              ),
            ),
          ),
        ],
      );
}
