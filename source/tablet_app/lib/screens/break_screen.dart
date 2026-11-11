import 'dart:async';
import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Rest screen shown when the tutor suggests a break. It pops `true` when the
/// break was taken ("I'm ready" after the countdown) and `false` when it was
/// skipped. The caller sends break_complete only for `true`, so the backend
/// applies fatigue recovery only for breaks that actually happened.
///
/// Calm by design: soft colours, one slow breathing circle (static under
/// reduced motion), a countdown ring and two clear choices.
class BreakScreen extends StatefulWidget {
  final int durationSeconds;

  const BreakScreen({super.key, required this.durationSeconds});

  @override
  State<BreakScreen> createState() => _BreakScreenState();
}

class _BreakScreenState extends State<BreakScreen> with SingleTickerProviderStateMixin {
  late int _remaining = widget.durationSeconds;
  Timer? _timer;
  late final AnimationController _breath = AnimationController(vsync: this, duration: const Duration(seconds: 4));

  @override
  void initState() {
    super.initState();
    _timer = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) return;
      setState(() => _remaining = (_remaining - 1).clamp(0, widget.durationSeconds).toInt());
      if (_remaining == 0) t.cancel();
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final reduce = MediaQuery.maybeOf(context)?.disableAnimations ?? false;
    if (reduce) {
      _breath.stop();
      _breath.value = 0.5;
    } else if (!_breath.isAnimating) {
      _breath.repeat(reverse: true);
    }
  }

  @override
  void dispose() {
    _timer?.cancel();
    _breath.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final text = Theme.of(context).textTheme;
    final done = _remaining == 0;
    final reduced = AppMotion.reduced(context);
    final total = math.max(1, widget.durationSeconds);
    return Scaffold(
      body: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AppColors.secondarySoft, AppColors.bg],
          ),
        ),
        child: SafeArea(
          child: LayoutBuilder(builder: (context, c) {
            // The circle shrinks on short screens so nothing is pushed off.
            final circle = (math.min(c.maxWidth, c.maxHeight) * 0.42).clamp(120.0, 260.0).toDouble();
            return Center(
              child: SingleChildScrollView(
                padding: const EdgeInsets.symmetric(horizontal: AppSpacing.lg, vertical: AppSpacing.md),
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 520),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Semantics(header: true, child: Text('Break time', style: text.displaySmall)),
                      const SizedBox(height: AppSpacing.sm),
                      Text('Shake out your hands and breathe with the circle.',
                          style: text.bodyLarge?.copyWith(color: AppColors.inkMuted), textAlign: TextAlign.center),
                      const SizedBox(height: AppSpacing.lg),
                      ExcludeSemantics(
                        child: SizedBox(
                          width: circle,
                          height: circle,
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              // Countdown ring: how much of the break is left.
                              SizedBox.expand(
                                child: CircularProgressIndicator(
                                  value: _remaining / total,
                                  strokeWidth: 6,
                                  backgroundColor: AppColors.paper,
                                  color: AppColors.secondary,
                                ),
                              ),
                              AnimatedBuilder(
                                animation: _breath,
                                builder: (_, __) {
                                  final d = circle * (0.55 + 0.30 * _breath.value);
                                  final label = reduced
                                      ? null
                                      : (_breath.status == AnimationStatus.reverse ? 'Breathe out' : 'Breathe in');
                                  return Container(
                                    width: d,
                                    height: d,
                                    alignment: Alignment.center,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: AppColors.secondary.withAlpha(40 + (60 * _breath.value).round()),
                                    ),
                                    child: label == null
                                        ? null
                                        : Text(label,
                                            style: const TextStyle(
                                                fontSize: 18, fontWeight: FontWeight.w700, color: AppColors.ink)),
                                  );
                                },
                              ),
                            ],
                          ),
                        ),
                      ),
                      const SizedBox(height: AppSpacing.lg),
                      Semantics(
                        liveRegion: true,
                        child:
                            Text(done ? 'Ready when you are!' : '$_remaining seconds left', style: text.headlineSmall),
                      ),
                      const SizedBox(height: AppSpacing.lg),
                      SizedBox(
                        width: double.infinity,
                        child: ElevatedButton(
                          onPressed: done ? () => Navigator.of(context).pop(true) : null,
                          child: const Text("I'm ready"),
                        ),
                      ),
                      const SizedBox(height: AppSpacing.sm),
                      TextButton(
                        onPressed: () => Navigator.of(context).pop(false),
                        child: const Text('Skip break', style: TextStyle(fontSize: 18)),
                      ),
                    ],
                  ),
                ),
              ),
            );
          }),
        ),
      ),
    );
  }
}
