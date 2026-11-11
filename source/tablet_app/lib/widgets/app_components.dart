import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Small, shared building blocks used across screens so headings, panels and
/// status messages look and behave the same everywhere. None of these hold
/// business logic.

/// Section title with an optional one-line explanation and trailing widget.
class AppSectionHeader extends StatelessWidget {
  final String title;
  final String? subtitle;
  final IconData? icon;
  final Widget? trailing;
  final Color? color;
  final Color? subtitleColor;

  const AppSectionHeader({
    super.key,
    required this.title,
    this.subtitle,
    this.icon,
    this.trailing,
    this.color,
    this.subtitleColor,
  });

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final fg = color ?? AppColors.ink;
    return Semantics(
      header: true,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          if (icon != null) ...[
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(color: AppColors.primarySoft, borderRadius: BorderRadius.circular(14)),
              child: Icon(icon, color: AppColors.primary, size: AppIconSize.md),
            ),
            const SizedBox(width: AppSpacing.md),
          ],
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(title, style: text.titleLarge?.copyWith(color: fg)),
                if (subtitle != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 2),
                    child:
                        Text(subtitle!, style: text.bodyMedium?.copyWith(color: subtitleColor ?? AppColors.inkMuted)),
                  ),
              ],
            ),
          ),
          if (trailing != null) ...[const SizedBox(width: AppSpacing.sm), trailing!],
        ],
      ),
    );
  }
}

/// White rounded panel used for grouped content.
class AppPanel extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry padding;
  final Color color;
  final Color borderColor;
  final double radius;
  final bool raised;

  const AppPanel({
    super.key,
    required this.child,
    this.padding = const EdgeInsets.all(AppSpacing.md),
    this.color = AppColors.paper,
    this.borderColor = AppColors.hairline,
    this.radius = AppRadii.xl,
    this.raised = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: padding,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(radius),
        border: Border.all(color: borderColor, width: 1.5),
        boxShadow: raised ? AppShadows.raised : AppShadows.card,
      ),
      child: child,
    );
  }
}

enum AppStatusKind { error, info, success }

/// A message with an icon and words, never colour alone. Errors explain how
/// to recover; [actionLabel] adds a button that does it.
class AppStatusBanner extends StatelessWidget {
  final AppStatusKind kind;
  final String message;
  final String? title;
  final String? actionLabel;
  final VoidCallback? onAction;
  final IconData? icon;

  const AppStatusBanner({
    super.key,
    required this.kind,
    required this.message,
    this.title,
    this.actionLabel,
    this.onAction,
    this.icon,
  });

  const AppStatusBanner.error({
    super.key,
    required this.message,
    this.title,
    this.actionLabel,
    this.onAction,
    this.icon,
  }) : kind = AppStatusKind.error;

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final (Color bg, Color fg, IconData defaultIcon) = switch (kind) {
      AppStatusKind.error => (AppColors.errorSoft, AppColors.errorInk, Icons.error_outline_rounded),
      AppStatusKind.info => (AppColors.infoSoft, AppColors.ink, Icons.info_outline_rounded),
      AppStatusKind.success => (AppColors.successSoft, AppColors.ink, Icons.check_circle_outline_rounded),
    };
    final action = actionLabel == null
        ? null
        : TextButton(
            onPressed: onAction,
            style: TextButton.styleFrom(foregroundColor: kind == AppStatusKind.error ? AppColors.errorInk : null),
            child: Text(actionLabel!),
          );
    return Semantics(
      liveRegion: kind == AppStatusKind.error,
      container: true,
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.fromLTRB(AppSpacing.md, AppSpacing.sm, AppSpacing.sm, AppSpacing.sm),
        decoration: BoxDecoration(
          color: bg,
          borderRadius: BorderRadius.circular(AppRadii.md),
          border: Border.all(color: fg.withValues(alpha: 0.25)),
        ),
        child: LayoutBuilder(builder: (context, c) {
          final body = Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Icon(icon ?? defaultIcon, color: fg, size: AppIconSize.md),
              ),
              const SizedBox(width: AppSpacing.sm),
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      if (title != null) Text(title!, style: text.titleSmall?.copyWith(color: fg)),
                      Text(message, style: text.bodyMedium?.copyWith(color: fg)),
                    ],
                  ),
                ),
              ),
            ],
          );
          if (action == null) return body;
          // Narrow: the button goes under the text so neither is squeezed.
          if (c.maxWidth < 420) {
            return Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              mainAxisSize: MainAxisSize.min,
              children: [body, action],
            );
          }
          return Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [Expanded(child: body), action],
          );
        }),
      ),
    );
  }
}

/// Centred spinner with a short sentence about what is loading.
class AppLoadingState extends StatelessWidget {
  final String message;

  const AppLoadingState({super.key, required this.message});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Semantics(
          liveRegion: true,
          label: message,
          child: ExcludeSemantics(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(width: 40, height: 40, child: CircularProgressIndicator(strokeWidth: 4)),
                const SizedBox(height: AppSpacing.md),
                Text(message, style: text.bodyLarge?.copyWith(color: AppColors.inkMuted), textAlign: TextAlign.center),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// Icon, title and one sentence for "nothing here yet" situations.
class AppEmptyState extends StatelessWidget {
  final IconData icon;
  final String title;
  final String? message;
  final Widget? action;

  const AppEmptyState({super.key, required this.icon, required this.title, this.message, this.action});

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    // Scrolls rather than overflowing when the space is short (landscape phones).
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(AppSpacing.lg),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 72,
              height: 72,
              decoration: const BoxDecoration(color: AppColors.primarySoft, shape: BoxShape.circle),
              child: Icon(icon, size: 36, color: AppColors.primary),
            ),
            const SizedBox(height: AppSpacing.md),
            Text(title, style: text.titleMedium, textAlign: TextAlign.center),
            if (message != null) ...[
              const SizedBox(height: AppSpacing.xs),
              Text(message!, style: text.bodyMedium?.copyWith(color: AppColors.inkMuted), textAlign: TextAlign.center),
            ],
            if (action != null) ...[const SizedBox(height: AppSpacing.md), action!],
          ],
        ),
      ),
    );
  }
}

/// Small rounded label with an icon, e.g. the practice mode or "For the adult".
class AppTag extends StatelessWidget {
  final String label;
  final IconData? icon;
  final Color background;
  final Color foreground;

  const AppTag({
    super.key,
    required this.label,
    this.icon,
    this.background = AppColors.primarySoft,
    this.foreground = AppColors.ink,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(color: background, borderRadius: BorderRadius.circular(999)),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (icon != null) ...[Icon(icon, size: 18, color: foreground), const SizedBox(width: 6)],
          Flexible(
            child: Text(
              label,
              style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: foreground, height: 1.2),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }
}

/// A labelled number (attempts, breaks, level). Value comes from the backend.
class AppStatTile extends StatelessWidget {
  final String label;
  final String value;
  final IconData icon;

  /// Icon above the number, for equal-width tiles in a row on phones.
  final bool compact;

  const AppStatTile({super.key, required this.label, required this.value, required this.icon, this.compact = false});

  @override
  Widget build(BuildContext context) {
    if (compact) {
      return Semantics(
        label: '$label: $value',
        child: ExcludeSemantics(
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: AppSpacing.sm, vertical: AppSpacing.md),
            decoration: BoxDecoration(
              color: AppColors.paper,
              borderRadius: BorderRadius.circular(AppRadii.lg),
              border: Border.all(color: AppColors.hairline, width: 1.5),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(icon, color: AppColors.secondary),
                const SizedBox(height: AppSpacing.xs),
                Text(value,
                    style:
                        const TextStyle(fontSize: 28, fontWeight: FontWeight.w800, color: AppColors.ink, height: 1.1)),
                FittedBox(child: Text(label, style: const TextStyle(fontSize: 16, color: AppColors.inkMuted))),
              ],
            ),
          ),
        ),
      );
    }
    return Semantics(
      label: '$label: $value',
      child: ExcludeSemantics(
        child: Container(
          constraints: const BoxConstraints(minWidth: 140),
          padding: const EdgeInsets.symmetric(horizontal: AppSpacing.md, vertical: AppSpacing.md),
          decoration: BoxDecoration(
            color: AppColors.paper,
            borderRadius: BorderRadius.circular(AppRadii.lg),
            border: Border.all(color: AppColors.hairline, width: 1.5),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(color: AppColors.secondarySoft, borderRadius: BorderRadius.circular(12)),
                child: Icon(icon, color: AppColors.secondary),
              ),
              const SizedBox(width: AppSpacing.md),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(value,
                      style: const TextStyle(
                          fontSize: 28, fontWeight: FontWeight.w800, color: AppColors.ink, height: 1.1)),
                  Text(label, style: const TextStyle(fontSize: 16, color: AppColors.inkMuted)),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

/// A row of stats: equal-width compact tiles on phones, a wrap on tablets.
class AppStatRow extends StatelessWidget {
  final List<(String label, String value, IconData icon)> stats;
  final WrapAlignment alignment;

  const AppStatRow({super.key, required this.stats, this.alignment = WrapAlignment.start});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, c) {
      if (c.maxWidth < 560) {
        return Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (var i = 0; i < stats.length; i++) ...[
              if (i > 0) const SizedBox(width: AppSpacing.sm),
              Expanded(child: AppStatTile(label: stats[i].$1, value: stats[i].$2, icon: stats[i].$3, compact: true)),
            ],
          ],
        );
      }
      return Wrap(
        alignment: alignment,
        spacing: AppSpacing.md,
        runSpacing: AppSpacing.md,
        children: [for (final s in stats) AppStatTile(label: s.$1, value: s.$2, icon: s.$3)],
      );
    });
  }
}

/// Button for navy app bars: white text and icon, applied locally so the
/// global TextButton style (cobalt on white) is unchanged elsewhere.
class AppBarActionButton extends StatelessWidget {
  final String label;
  final IconData icon;
  final VoidCallback? onPressed;

  /// Show only the icon (with a tooltip and semantic label) on narrow screens.
  final bool compact;

  const AppBarActionButton({
    super.key,
    required this.label,
    required this.icon,
    required this.onPressed,
    this.compact = false,
  });

  @override
  Widget build(BuildContext context) {
    if (compact) {
      return IconButton(
        tooltip: label,
        onPressed: onPressed,
        icon: Icon(icon, color: Colors.white),
        iconSize: 28,
        constraints: const BoxConstraints(minWidth: 48, minHeight: 48),
      );
    }
    return TextButton.icon(
      onPressed: onPressed,
      style: TextButton.styleFrom(
        foregroundColor: Colors.white,
        minimumSize: const Size(48, 48),
        textStyle: const TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
      ),
      icon: Icon(icon),
      label: Text(label),
    );
  }
}

/// A numbered step indicator ("1 Look  2 Learn  3 Trace ..."). [current] is
/// highlighted; earlier steps show a tick. Purely presentational.
class AppStepIndicator extends StatelessWidget {
  final List<(IconData, String)> steps;
  final int current;

  const AppStepIndicator({super.key, required this.steps, required this.current});

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Steps: ${[for (var i = 0; i < steps.length; i++) '${i + 1}. ${steps[i].$2}'].join(', ')}. '
          'You are on step ${current + 1}.',
      child: ExcludeSemantics(
        child: Wrap(
          spacing: AppSpacing.xs,
          runSpacing: AppSpacing.xs,
          children: [
            for (var i = 0; i < steps.length; i++)
              _Step(index: i, icon: steps[i].$1, label: steps[i].$2, state: i < current ? -1 : (i == current ? 0 : 1)),
          ],
        ),
      ),
    );
  }
}

class _Step extends StatelessWidget {
  final int index;
  final IconData icon;
  final String label;

  /// -1 done, 0 current, 1 upcoming.
  final int state;

  const _Step({required this.index, required this.icon, required this.label, required this.state});

  @override
  Widget build(BuildContext context) {
    final current = state == 0;
    final bg = current ? AppColors.secondary : (state < 0 ? AppColors.secondarySoft : AppColors.paper);
    final fg = current ? Colors.white : AppColors.ink;
    return Container(
      padding: const EdgeInsets.fromLTRB(6, 6, 12, 6),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: current ? AppColors.secondary : AppColors.hairline, width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 26,
            height: 26,
            alignment: Alignment.center,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: current ? Colors.white : (state < 0 ? AppColors.secondary : AppColors.bg),
            ),
            child: state < 0
                ? const Icon(Icons.check_rounded, size: 18, color: Colors.white)
                : Text('${index + 1}',
                    style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                        color: current ? AppColors.secondary : AppColors.inkMuted)),
          ),
          const SizedBox(width: 6),
          Icon(icon, size: 18, color: fg),
          const SizedBox(width: 4),
          Text(label, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w600, color: fg)),
        ],
      ),
    );
  }
}

/// The product's signature surface for header regions: deep navy, a quiet
/// glow from the brand gradient, and faint four-line writing paper (top
/// line, dashed middle line, baseline, descender line), the same lines the
/// learner writes on. Decorative only: never placed behind the canvas.
class AppHeroBand extends StatelessWidget {
  final Widget child;

  const AppHeroBand({super.key, required this.child});

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: const BoxDecoration(color: AppColors.navy),
      child: CustomPaint(
        painter: const _HeroPainter(),
        child: child,
      ),
    );
  }
}

class _HeroPainter extends CustomPainter {
  const _HeroPainter();

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;
    final rect = Offset.zero & size;
    // Brand gradient as a low-opacity wash, strongest at the top right.
    canvas.drawRect(
      rect,
      Paint()
        ..shader = RadialGradient(
          center: const Alignment(0.9, -1.1),
          radius: 1.4,
          colors: [
            AppColors.gradientBlue.withValues(alpha: 0.55),
            AppColors.gradientTeal.withValues(alpha: 0.30),
            AppColors.gradientGreen.withValues(alpha: 0.10),
            AppColors.navy.withValues(alpha: 0),
          ],
          stops: const [0, 0.35, 0.7, 1],
        ).createShader(rect),
    );
    // Four-line writing paper, repeated down the band.
    const gap = 22.0; // between lines inside one set
    const setH = gap * 3 + 40;
    final solid = Paint()
      ..color = Colors.white.withValues(alpha: 0.07)
      ..strokeWidth = 1.2;
    final strong = Paint()
      ..color = Colors.white.withValues(alpha: 0.11)
      ..strokeWidth = 1.6;
    for (var top = 18.0; top < size.height; top += setH) {
      canvas.drawLine(Offset(0, top), Offset(size.width, top), solid);
      for (var x = 0.0; x < size.width; x += 18) {
        canvas.drawLine(Offset(x, top + gap), Offset(x + 9, top + gap), solid);
      }
      canvas.drawLine(Offset(0, top + gap * 2), Offset(size.width, top + gap * 2), strong);
      canvas.drawLine(Offset(0, top + gap * 3), Offset(size.width, top + gap * 3), solid);
    }
  }

  @override
  bool shouldRepaint(covariant _HeroPainter oldDelegate) => false;
}

/// Gentle press feedback: the child scales to 97% while pressed. Instant
/// (no scaling) when the platform asks for reduced motion. It listens to
/// pointer events only to animate; taps are still handled by the child.
class AppPressable extends StatefulWidget {
  final Widget child;
  final bool enabled;

  const AppPressable({super.key, required this.child, this.enabled = true});

  @override
  State<AppPressable> createState() => _AppPressableState();
}

class _AppPressableState extends State<AppPressable> {
  bool _down = false;

  void _set(bool v) {
    if (_down != v && mounted) setState(() => _down = v);
  }

  @override
  Widget build(BuildContext context) {
    final reduced = AppMotion.reduced(context);
    return Listener(
      onPointerDown: widget.enabled ? (_) => _set(true) : null,
      onPointerUp: (_) => _set(false),
      onPointerCancel: (_) => _set(false),
      child: AnimatedScale(
        scale: _down && !reduced ? 0.97 : 1.0,
        duration: AppMotion.fast,
        curve: AppMotion.curve,
        child: widget.child,
      ),
    );
  }
}

/// Small uppercase label above a group ("YOUR WRITING", "NEXT STEP").
class AppEyebrow extends StatelessWidget {
  final String text;
  final Color color;
  final IconData? icon;

  const AppEyebrow(this.text, {super.key, this.color = AppColors.inkMuted, this.icon});

  @override
  Widget build(BuildContext context) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (icon != null) ...[Icon(icon, size: 16, color: color), const SizedBox(width: 6)],
        Flexible(
          child: Text(text.toUpperCase(),
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, letterSpacing: 1.1, color: color)),
        ),
      ],
    );
  }
}
