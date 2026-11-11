import 'package:flutter/material.dart';
import 'package:uuid/uuid.dart';

import '../models/exercise.dart';
import '../models/exercise_instruction.dart';
import '../models/stroke_point.dart';
import '../models/attempt_result.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';
import '../widgets/app_components.dart';
import '../widgets/handwriting_canvas.dart';
import '../widgets/exercise_display.dart';
import '../widgets/session_controls.dart';
import '../widgets/feedback_overlay.dart';
import 'break_screen.dart';
import 'completion_screen.dart';
import 'progress_screen.dart';

/// Practice screen. The canvas takes most of the screen. Each submit sends the
/// captured points, and the screen then carries out exactly what the backend's
/// intervention says:
///   hint             -> start dot + direction arrows on the guide
///   stroke_animation -> a dot demonstrates the shape
///   template_size    -> guide redrawn at the new template_scale
///   mode_change      -> guide shown (tracing) or hidden (free writing)
///   difficulty / repeat / continue -> the backend's next_exercise
///   break            -> the break screen. break_complete is only sent if the
///                       break was actually taken (not skipped).
class WritingScreen extends StatefulWidget {
  final ApiService api;
  final String sessionId;
  final String childId;
  final Exercise initialExercise;
  final VoidCallback onSessionEnded;

  /// The session's starting mode as the backend reported it at session start
  /// ("tracing" shows the guide, "free_writing" hides it while writing).
  final String initialMode;

  /// Called with the exercise id after the backend accepted an attempt.
  final ValueChanged<String>? onAttemptAccepted;

  const WritingScreen({
    super.key,
    required this.api,
    required this.sessionId,
    required this.childId,
    required this.initialExercise,
    required this.onSessionEnded,
    this.initialMode = 'tracing',
    this.onAttemptAccepted,
  });

  @override
  State<WritingScreen> createState() => _WritingScreenState();
}

class _WritingScreenState extends State<WritingScreen> {
  final _canvasKey = GlobalKey<HandwritingCanvasState>();
  final _uuid = const Uuid();

  late Exercise _exercise;
  List<List<StrokePoint>> _strokes = [];
  bool _isSubmitting = false;
  AttemptResult? _lastResult;
  String? _errorMessage;
  double _templateScale = 1.0;
  String _mode = 'tracing';
  int _retryIndex = 0;

  @override
  void initState() {
    super.initState();
    _exercise = widget.initialExercise;
    _mode = widget.initialMode == 'free_writing' ? 'free_writing' : 'tracing';
  }

  // One attempt id per captured attempt. A retry after a network error
  // reuses it, so the backend's idempotent submit cannot count the same
  // attempt twice. Any change to the strokes starts a new attempt.
  String? _pendingAttemptId;

  void _onStrokesChanged(List<List<StrokePoint>> strokes) {
    setState(() {
      _strokes = strokes;
      _pendingAttemptId = null;
    });
  }

  Future<void> _submit() async {
    final points = _strokes.expand((s) => s).toList();
    if (points.isEmpty) {
      setState(() => _errorMessage = 'Write in the writing area first, then press Submit.');
      return;
    }
    setState(() {
      _isSubmitting = true;
      _errorMessage = null;
    });
    try {
      final capture = {
        ...?_canvasKey.currentState?.captureMetadata(),
        'retry_index': _retryIndex,
        'completed': true,
        'client_version': 'tablet_app/0.2.0',
      };
      final result = await widget.api.submitAttempt(
        sessionId: widget.sessionId,
        exerciseId: _exercise.exerciseId,
        attemptId: _pendingAttemptId ??= _uuid.v4(),
        points: points,
        capture: capture,
      );
      if (!result.duplicateSubmission) widget.onAttemptAccepted?.call(result.exerciseId);
      if (!mounted) return;
      setState(() {
        _lastResult = result;
        _isSubmitting = false;
        _pendingAttemptId = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isSubmitting = false;
        _errorMessage = describeError(e);
      });
    }
  }

  Future<void> _applyIntervention() async {
    final result = _lastResult;
    if (result == null) return;
    final type = result.intervention.type;
    final params = result.intervention.params;
    final reducedMotion = MediaQuery.maybeOf(context)?.disableAnimations ?? false;

    setState(() {
      _lastResult = null;
      _strokes = [];
      _templateScale = result.templateScale;
      _mode = result.mode;
      final changed = result.nextExercise.exerciseId != _exercise.exerciseId;
      _exercise = result.nextExercise;
      _retryIndex = changed ? 0 : _retryIndex + 1;
    });
    _canvasKey.currentState?.retry();

    if (type == 'break') {
      final seconds = (params['duration_seconds'] as num?)?.toInt() ?? 60;
      final taken = await Navigator.of(context).push<bool>(
        MaterialPageRoute(builder: (_) => BreakScreen(durationSeconds: seconds)),
      );
      if (taken == true) {
        try {
          await widget.api.breakComplete(widget.sessionId);
        } catch (e) {
          if (mounted) setState(() => _errorMessage = describeError(e));
        }
      }
    } else if (type == 'hint') {
      final seconds = (params['display_seconds'] as num?)?.toInt() ?? 8;
      WidgetsBinding.instance.addPostFrameCallback((_) => _canvasKey.currentState?.showHint(seconds: seconds));
    } else if (type == 'stroke_animation') {
      WidgetsBinding.instance
          .addPostFrameCallback((_) => _canvasKey.currentState?.playDemo(reducedMotion: reducedMotion));
    }
    // 'difficulty_adjustment', 'letter_rotation', 'mode_change',
    // 'template_size', 'repeat_exercise' and 'continue' need no special
    // action: setState above already swapped _exercise / _templateScale /
    // _mode to the backend's reported next exercise / scale / mode, so
    // the next build will draw the rotated letter (or the harder /
    // easier exercise) with its own guide at the requested scale.
  }

  Future<void> _openProgress() async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => ProgressScreen(api: widget.api, sessionId: widget.sessionId),
    ));
  }

  Future<void> _finish() async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Finish practice?'),
        content: const Text('This ends the session.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Keep practising')),
          ElevatedButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Finish')),
        ],
      ),
    );
    if (confirm != true || !mounted) return;
    try {
      await widget.api.endSession(widget.sessionId);
    } catch (e) {
      if (mounted) setState(() => _errorMessage = describeError(e));
      return;
    }
    widget.onSessionEnded();
    if (!mounted) return;
    await Navigator.of(context).pushReplacement(MaterialPageRoute(
      builder: (_) => CompletionScreen(api: widget.api, sessionId: widget.sessionId),
    ));
  }

  @override
  Widget build(BuildContext context) {
    // Derived from the current exercise and mode on every build (never stored),
    // so the instruction, the guide and the exercise cannot get out of step.
    final instruction = instructionFor(_exercise, _mode, retryIndex: _retryIndex);
    final narrow = MediaQuery.sizeOf(context).width < 600;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Handwriting Practice'),
        automaticallyImplyLeading: false,
        actions: [
          // White on navy, styled locally (the global TextButton is cobalt,
          // which is unreadable on the navy app bar).
          AppBarActionButton(
              key: const Key('open-progress'),
              label: 'Progress',
              icon: Icons.show_chart_rounded,
              onPressed: _openProgress,
              compact: narrow),
          const SizedBox(width: AppSpacing.xs),
          AppBarActionButton(
              key: const Key('finish-session'),
              label: 'Finish',
              icon: Icons.flag_rounded,
              onPressed: _finish,
              compact: narrow),
          const SizedBox(width: AppSpacing.sm),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(builder: (context, c) {
          final gutter = c.maxWidth < 600 ? AppSpacing.sm + 4 : AppSpacing.md;
          // Wide landscape tablets: tools beside the canvas, so the writing
          // area keeps as much height as possible.
          final landscape = c.maxWidth / c.maxHeight > 1.25;
          final sideControls = landscape && (c.maxWidth >= 1000 || c.maxHeight < 500);
          final controls = SessionControls(
            onClear: () => _canvasKey.currentState?.clear(),
            onUndo: () => _canvasKey.currentState?.undoLastStroke(),
            onRetry: () => _canvasKey.currentState?.retry(),
            onSubmit: _submit,
            submitEnabled: _lastResult == null,
            isSubmitting: _isSubmitting,
            vertical: sideControls,
          );
          final writingArea = _WritingArea(
            canvas: HandwritingCanvas(
              key: _canvasKey,
              referenceTemplate: _exercise.referenceTemplate,
              templateStrokes: _exercise.guideStrokes,
              writingLines: _exercise.writingLines,
              semanticLabel: 'Writing area. ${instruction.headline} ${instruction.guidance}',
              templateScale: _templateScale,
              mode: _mode,
              enabled: _lastResult == null && !_isSubmitting,
              onStrokesChanged: _onStrokesChanged,
            ),
            feedback: _lastResult == null ? null : FeedbackOverlay(result: _lastResult!, onDismiss: _applyIntervention),
            // The instruction is the instrument's header strip, so what to
            // write and where to write it read as one object.
            header: ExerciseDisplay(
              exercise: _exercise,
              templateScale: _templateScale,
              mode: _mode,
              retryIndex: _retryIndex,
              dense: c.maxHeight < 480,
              attached: true,
              trailing: c.maxWidth < 600 ? null : _InkStatus(strokes: _strokes.length),
            ),
          );
          return Padding(
            padding: EdgeInsets.fromLTRB(gutter, gutter, gutter, gutter),
            child: Column(
              children: [
                if (_errorMessage != null)
                  Padding(
                    padding: const EdgeInsets.only(bottom: AppSpacing.sm),
                    child: AppStatusBanner.error(message: _errorMessage!),
                  ),
                Expanded(
                  child: sideControls
                      ? Row(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            Expanded(child: writingArea),
                            const SizedBox(width: AppSpacing.md),
                            SizedBox(
                              width: c.maxWidth >= 1000 ? 220 : 168,
                              child: SingleChildScrollView(child: controls),
                            ),
                          ],
                        )
                      : writingArea,
                ),
                if (!sideControls) ...[const SizedBox(height: AppSpacing.sm + 4), controls],
              ],
            ),
          );
        }),
      ),
    );
  }
}

/// The white writing surface with a quiet border, and the feedback card on
/// top of it after a submit. Nothing decorative is drawn behind or over the
/// canvas while the learner writes. The canvas itself (input capture,
/// coordinates, guide drawing) is HandwritingCanvas, unchanged: its
/// coordinates are local to the canvas, so this frame does not affect them.
class _WritingArea extends StatelessWidget {
  final Widget canvas;
  final Widget? feedback;
  final Widget header;

  const _WritingArea({required this.canvas, required this.feedback, required this.header});

  @override
  Widget build(BuildContext context) {
    return Container(
      key: const Key('writing-instrument'),
      decoration: BoxDecoration(
        color: AppColors.paper,
        border: Border.all(color: AppColors.hairline, width: 1.5),
        borderRadius: BorderRadius.circular(AppRadii.xl),
        boxShadow: AppShadows.card,
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          header,
          const Divider(height: 1, thickness: 1, color: AppColors.hairline),
          Expanded(
            child: Stack(
              children: [
                // Plain white paper: nothing decorative behind or over the
                // canvas while the learner writes.
                Positioned.fill(
                    child: ColoredBox(key: const Key('writing-surface'), color: AppColors.paper, child: canvas)),
                if (feedback != null) ...[
                  // Dim the canvas a little so the feedback reads as the next step.
                  Positioned.fill(
                    child: IgnorePointer(
                      child: ColoredBox(color: AppColors.navy.withValues(alpha: 0.10)),
                    ),
                  ),
                  Positioned.fill(
                    child: Align(alignment: Alignment.bottomCenter, child: feedback!),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// How much ink is on the paper (real canvas state), so the learner and the
/// adult can see the tablet registered the writing before Submit.
class _InkStatus extends StatelessWidget {
  final int strokes;

  const _InkStatus({required this.strokes});

  @override
  Widget build(BuildContext context) {
    final has = strokes > 0;
    return Semantics(
      liveRegion: true,
      label: has ? '$strokes ${strokes == 1 ? 'stroke' : 'strokes'} written' : 'Nothing written yet',
      child: ExcludeSemantics(
        child: AnimatedContainer(
          key: const Key('ink-status'),
          duration: AppMotion.duration(context, AppMotion.fast),
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          decoration: BoxDecoration(
            color: has ? AppColors.primarySoft : AppColors.bg,
            borderRadius: BorderRadius.circular(999),
          ),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(has ? Icons.draw_rounded : Icons.touch_app_outlined,
                size: 18, color: has ? AppColors.primary : AppColors.inkMuted),
            const SizedBox(width: 6),
            Text(has ? '$strokes ${strokes == 1 ? 'stroke' : 'strokes'}' : 'Write in the box',
                style: TextStyle(
                    fontSize: 15, fontWeight: FontWeight.w600, color: has ? AppColors.ink : AppColors.inkMuted)),
          ]),
        ),
      ),
    );
  }
}
