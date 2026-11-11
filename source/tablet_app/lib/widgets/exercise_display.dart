import 'package:flutter/material.dart';

import '../models/exercise.dart';
import '../models/exercise_instruction.dart';
import '../theme/app_theme.dart';
import 'app_components.dart';
import 'letter_glyph.dart';
import 'shape_preview.dart';

/// Instruction card above the canvas: the practice mode (tracing or on your
/// own), WHAT to write ("Trace the letter A."), HOW and WHERE ("Follow the
/// guide in the writing area."), a small model of the target (always
/// visible, so free-writing mode still shows what to write) and the level.
/// The full-size guide is drawn by HandwritingCanvas.
///
/// Everything shown is derived on every build from [exercise] and [mode]
/// (instructionFor), so it updates with the exercise and cannot go stale.
class ExerciseDisplay extends StatelessWidget {
  final Exercise exercise;
  final double templateScale;
  final String mode;

  /// 0 for a first attempt; > 0 when the same exercise is being retried.
  final int retryIndex;

  /// Short screens (landscape phones): one line, no model and no guidance
  /// sentence, so the writing area keeps its height. The guidance is still
  /// part of the canvas's screen-reader label.
  final bool dense;

  /// Drawn as the header strip of the writing instrument (the parent draws
  /// the frame), instead of as a free-standing card.
  final bool attached;

  /// Optional status at the end of the strip (e.g. how much ink is down).
  final Widget? trailing;

  const ExerciseDisplay({
    super.key,
    required this.exercise,
    this.templateScale = 1.0,
    this.mode = 'tracing',
    this.retryIndex = 0,
    this.dense = false,
    this.attached = false,
    this.trailing,
  });

  BoxDecoration _frame(double radius, {bool shadow = false}) => attached
      ? const BoxDecoration(color: AppColors.paper)
      : BoxDecoration(
          color: AppColors.paper,
          borderRadius: BorderRadius.circular(radius),
          border: Border.all(color: AppColors.hairline, width: 1.5),
          boxShadow: shadow ? AppShadows.card : null,
        );

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final instruction = instructionFor(exercise, mode, retryIndex: retryIndex);
    final tracing = guideVisibleFor(mode);
    return LayoutBuilder(builder: (context, c) {
      final narrow = c.maxWidth < 560;
      final preview = Semantics(
        label: instruction.label.isEmpty ? 'Model of the exercise' : 'Model of the ${instruction.label}',
        child: Container(
          width: narrow ? 72 : 104,
          height: narrow ? 64 : 84,
          padding: const EdgeInsets.all(6),
          decoration: BoxDecoration(color: AppColors.bg, borderRadius: BorderRadius.circular(AppRadii.md)),
          // Letters: the taught letter model, drawn from its strokes.
          child: exercise.isLetter
              ? LetterGlyph.of(exercise, showLines: true)
              : ShapePreview.strokes(
                  strokes: exercise.guideStrokes,
                  ascender: exercise.writingLines?.ascender,
                  baseline: exercise.writingLines?.baseline,
                ),
        ),
      );
      if (dense) {
        return Container(
          key: const Key('instruction-card'),
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm + 4, vertical: AppSpacing.xs + 2),
          decoration: _frame(AppRadii.md),
          child: Row(
            children: [
              Icon(tracing ? Icons.gesture_rounded : Icons.edit_rounded, color: AppColors.secondary),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: Semantics(
                  header: true,
                  child: Text(
                    instruction.headline,
                    key: const ValueKey('instruction-headline'),
                    style: text.titleMedium?.copyWith(fontWeight: FontWeight.w800),
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ),
              if (instruction.encouragement != null)
                Flexible(
                  child: Text(instruction.encouragement!,
                      key: const ValueKey('instruction-encouragement'),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: text.bodyMedium?.copyWith(color: AppColors.secondary, fontWeight: FontWeight.w700)),
                ),
              const SizedBox(width: AppSpacing.sm),
              Text('Level ${exercise.difficulty}', style: text.bodyMedium?.copyWith(color: AppColors.inkMuted)),
            ],
          ),
        );
      }
      return Container(
        key: const Key('instruction-card'),
        padding: EdgeInsets.all(narrow ? AppSpacing.sm + 4 : AppSpacing.md),
        decoration: _frame(AppRadii.lg, shadow: true),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            preview,
            SizedBox(width: narrow ? AppSpacing.sm + 4 : AppSpacing.md),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Wrap(
                    spacing: AppSpacing.sm,
                    runSpacing: AppSpacing.xs,
                    children: [
                      Semantics(
                        label: tracing ? 'Practice mode: tracing' : 'Practice mode: writing on your own',
                        child: ExcludeSemantics(
                          child: AppTag(
                            key: const Key('practice-mode'),
                            label: tracing ? 'Tracing' : 'On your own',
                            icon: tracing ? Icons.gesture_rounded : Icons.edit_rounded,
                            background: tracing ? AppColors.secondarySoft : AppColors.primarySoft,
                          ),
                        ),
                      ),
                      AppTag(
                        label: 'Level ${exercise.difficulty}',
                        icon: Icons.signal_cellular_alt_rounded,
                        background: AppColors.bg,
                      ),
                      if (retryIndex > 0)
                        AppTag(
                          key: const Key('try-number'),
                          label: 'Try ${retryIndex + 1}',
                          icon: Icons.replay_rounded,
                          background: AppColors.bg,
                        ),
                    ],
                  ),
                  const SizedBox(height: AppSpacing.xs + 2),
                  Semantics(
                    header: true,
                    child: Text(
                      instruction.headline,
                      key: const ValueKey('instruction-headline'),
                      style: (narrow ? text.titleLarge : text.headlineSmall)
                          ?.copyWith(fontWeight: FontWeight.w800, color: AppColors.ink),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    instruction.guidance,
                    key: const ValueKey('instruction-guidance'),
                    style: (narrow ? text.bodyMedium : text.bodyLarge)?.copyWith(color: AppColors.inkMuted),
                  ),
                  if (instruction.encouragement != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 2),
                      child: Text(
                        instruction.encouragement!,
                        key: const ValueKey('instruction-encouragement'),
                        style: text.titleMedium?.copyWith(color: AppColors.secondary, fontWeight: FontWeight.w700),
                      ),
                    ),
                ],
              ),
            ),
            if (trailing != null) ...[const SizedBox(width: AppSpacing.sm), trailing!],
          ],
        ),
      );
    });
  }
}
