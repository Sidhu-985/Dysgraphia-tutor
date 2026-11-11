import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Bottom toolbar: Clear / Undo / Retry / Submit. Every callback is
/// provided by the parent screen; this widget has no logic of its own.
/// Buttons are at least 64 dp tall, so they are easy for children to hit.
///
/// Layouts: one row (tablets), a compact row with icon-over-label tools
/// (phones), or a column beside the canvas ([vertical], wide landscape).
class SessionControls extends StatelessWidget {
  final VoidCallback onClear;
  final VoidCallback onUndo;
  final VoidCallback onRetry;
  final VoidCallback onSubmit;
  final bool submitEnabled;
  final bool isSubmitting;
  final bool vertical;

  const SessionControls({
    super.key,
    required this.onClear,
    required this.onUndo,
    required this.onRetry,
    required this.onSubmit,
    this.submitEnabled = true,
    this.isSubmitting = false,
    this.vertical = false,
  });

  @override
  Widget build(BuildContext context) {
    // A plain ElevatedButton (not ElevatedButton.icon, which builds a
    // private subclass) so `find.byType(ElevatedButton)` matches it.
    Widget submitButton({bool compact = false}) => ElevatedButton(
          key: const Key('submit'),
          onPressed: submitEnabled && !isSubmitting ? onSubmit : null,
          style: ElevatedButton.styleFrom(
            minimumSize: Size(compact ? 0 : 140, AppSpacing.touchTarget),
            padding: compact ? const EdgeInsets.symmetric(horizontal: AppSpacing.sm) : null,
          ),
          child: FittedBox(
            fit: BoxFit.scaleDown,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                isSubmitting
                    ? const SizedBox(
                        width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 3, color: AppColors.ink))
                    : const Icon(Icons.check_circle_outline, size: 28),
                const SizedBox(width: AppSpacing.sm),
                const Text('Submit'),
              ],
            ),
          ),
        );
    Widget submitLabelled({bool compact = false}) => Semantics(
          label: isSubmitting ? 'Checking your writing' : null,
          child: submitButton(compact: compact),
        );

    if (vertical) {
      return LayoutBuilder(builder: (context, c) {
        // Narrow side column (landscape phones): three small tools in a row,
        // Submit underneath, so all four stay on screen.
        if (c.maxWidth < 200) {
          return Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            mainAxisSize: MainAxisSize.min,
            children: [
              Row(children: [
                Expanded(child: _ToolButton(label: 'Clear', icon: Icons.clear, onPressed: onClear, horizontal: false)),
                const SizedBox(width: 4),
                Expanded(child: _ToolButton(label: 'Undo', icon: Icons.undo, onPressed: onUndo, horizontal: false)),
                const SizedBox(width: 4),
                Expanded(child: _ToolButton(label: 'Retry', icon: Icons.replay, onPressed: onRetry, horizontal: false)),
              ]),
              const SizedBox(height: AppSpacing.sm),
              submitLabelled(compact: true),
            ],
          );
        }
        return Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisSize: MainAxisSize.min,
          children: [
            _ToolButton(label: 'Clear', icon: Icons.clear, onPressed: onClear, horizontal: true),
            const SizedBox(height: AppSpacing.sm),
            _ToolButton(label: 'Undo', icon: Icons.undo, onPressed: onUndo, horizontal: true),
            const SizedBox(height: AppSpacing.sm),
            _ToolButton(label: 'Retry', icon: Icons.replay, onPressed: onRetry, horizontal: true),
            const SizedBox(height: AppSpacing.lg),
            submitLabelled(),
          ],
        );
      });
    }

    return LayoutBuilder(builder: (context, c) {
      final compact = c.maxWidth < 560;
      final tools = [
        _ToolButton(label: 'Clear', icon: Icons.clear, onPressed: onClear, horizontal: !compact),
        _ToolButton(label: 'Undo', icon: Icons.undo, onPressed: onUndo, horizontal: !compact),
        _ToolButton(label: 'Retry', icon: Icons.replay, onPressed: onRetry, horizontal: !compact),
      ];
      if (compact) {
        return Row(
          children: [
            for (final t in tools) ...[Expanded(child: t), const SizedBox(width: AppSpacing.sm)],
            Expanded(flex: 2, child: submitLabelled(compact: true)),
          ],
        );
      }
      return Row(
        children: [
          for (final t in tools) ...[t, const SizedBox(width: AppSpacing.sm)],
          const Spacer(),
          ConstrainedBox(constraints: const BoxConstraints(minWidth: 200), child: submitLabelled()),
        ],
      );
    });
  }
}

/// Secondary tool: icon and label side by side, or stacked on phones so the
/// four controls fit in one 64 dp row at 360 dp.
class _ToolButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback onPressed;
  final bool horizontal;

  const _ToolButton({required this.label, required this.icon, required this.onPressed, required this.horizontal});

  @override
  Widget build(BuildContext context) {
    if (horizontal) {
      return OutlinedButton.icon(
        onPressed: onPressed,
        icon: Icon(icon, size: 26),
        label: Text(label),
      );
    }
    return OutlinedButton(
      onPressed: onPressed,
      style: OutlinedButton.styleFrom(
        minimumSize: const Size(0, AppSpacing.touchTarget),
        padding: const EdgeInsets.symmetric(horizontal: 2, vertical: 6),
        textStyle: const TextStyle(fontSize: 15, fontWeight: FontWeight.w700),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 24),
          const SizedBox(height: 2),
          FittedBox(child: Text(label)),
        ],
      ),
    );
  }
}
