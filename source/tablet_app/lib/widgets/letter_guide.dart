import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/exercise.dart';
import '../theme/app_theme.dart';
import 'template_geometry.dart';

/// Colours used to tell the strokes of one letter apart (stroke 1, 2, 3, 4).
/// Each has at least 3:1 contrast on the paper colour for graphics.
const List<Color> kStrokeColors = [
  Color(0xFF1E6F68), // teal
  Color(0xFF6E5BA8), // lavender
  Color(0xFFB4532A), // terracotta
  Color(0xFF2F6CA8), // blue
];

/// The teaching view of a letter: four-line paper, the guide strokes in
/// drawing order, a numbered start marker on every stroke and arrows that show
/// the direction. With [progress] (0-1) the strokes are drawn one after the
/// other up to that point ("show me how").
class LetterGuide extends StatelessWidget {
  final List<List<List<double>>> strokes;
  final WritingLines? lines;
  final bool showNumbers;
  final bool showArrows;
  final double? progress;
  final double templateScale;
  final String semanticLabel;

  const LetterGuide({
    super.key,
    required this.strokes,
    this.lines,
    this.showNumbers = true,
    this.showArrows = true,
    this.progress,
    this.templateScale = 1.6,
    this.semanticLabel = '',
  });

  @override
  Widget build(BuildContext context) => Semantics(
        image: true,
        label: semanticLabel,
        child: CustomPaint(
          painter: LetterGuidePainter(
            strokes: strokes,
            lines: lines,
            showNumbers: showNumbers,
            showArrows: showArrows,
            progress: progress,
            templateScale: templateScale,
          ),
          size: Size.infinite,
        ),
      );
}

class LetterGuidePainter extends CustomPainter {
  final List<List<List<double>>> strokes;
  final WritingLines? lines;
  final bool showNumbers;
  final bool showArrows;
  final double? progress;
  final double templateScale;

  LetterGuidePainter({
    required this.strokes,
    required this.lines,
    required this.showNumbers,
    required this.showArrows,
    required this.progress,
    required this.templateScale,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final valid = strokes.where((s) => s.length >= 2).toList();
    final all = [for (final s in valid) ...s];
    if (all.length < 2 || size.isEmpty) return;
    final wl = lines;
    final fit = TemplateFit.compute(fitPointsFor(all, wl?.ascender, wl?.baseline, descender: wl?.descender), size,
        templateScale: templateScale);

    if (wl != null) _paintLines(canvas, size, fit, all.first[0], wl);

    final mapped = [for (final s in valid) fit.mapAll(s)];
    final width = math.max(6.0, size.shortestSide / 28);
    final walkers = [for (final s in mapped) PolylineWalker(s)];
    final total = walkers.fold<double>(0, (a, w) => a + w.length);
    var budget = progress == null ? double.infinity : progress!.clamp(0.0, 1.0).toDouble() * total;

    for (var i = 0; i < mapped.length; i++) {
      final color = kStrokeColors[i % kStrokeColors.length];
      final w = walkers[i];
      if (budget <= 0) break;
      final frac = w.length <= 0 ? 1.0 : math.min(1.0, budget / w.length);
      budget -= w.length;
      final pts = frac >= 1.0 ? mapped[i] : w.prefix(frac);
      if (pts.length >= 2) {
        final path = Path()..moveTo(pts.first.dx, pts.first.dy);
        for (final p in pts.skip(1)) {
          path.lineTo(p.dx, p.dy);
        }
        canvas.drawPath(
            path,
            Paint()
              ..color = color.withAlpha(progress == null ? 200 : 235)
              ..strokeWidth = width
              ..style = PaintingStyle.stroke
              ..strokeCap = StrokeCap.round
              ..strokeJoin = StrokeJoin.round);
      }
      if (showArrows && frac >= 1.0 && w.length > width * 4) _paintArrows(canvas, w, color, width);
    }
    if (showNumbers) {
      final r = math.max(13.0, width * 1.3);
      final placed = <Offset>[];
      for (var i = 0; i < mapped.length; i++) {
        // Two strokes often start at the same point (A, B, E, M ...). A later
        // marker is then moved a little way along its own stroke, so both
        // numbers stay visible and each still sits on its own stroke.
        var at = mapped[i].first;
        if (placed.any((q) => (q - at).distance < 2 * r) && walkers[i].length > 0) {
          at = walkers[i].at(math.min(0.45, 2.4 * r / walkers[i].length)).$1;
        }
        placed.add(at);
        _paintStartMarker(canvas, at, i + 1, kStrokeColors[i % kStrokeColors.length], width);
      }
    }
  }

  void _paintLines(Canvas canvas, Size size, TemplateFit fit, double x, WritingLines wl) {
    const pad = 12.0;
    double y(double v) => fit.map([x, v]).dy;
    final solid = Paint()
      ..color = AppColors.lavender.withAlpha(120)
      ..strokeWidth = 2.5;
    canvas.drawLine(Offset(pad, y(wl.ascender)), Offset(size.width - pad, y(wl.ascender)), solid);
    canvas.drawLine(
        Offset(pad, y(wl.baseline)),
        Offset(size.width - pad, y(wl.baseline)),
        Paint()
          ..color = AppColors.lavender.withAlpha(170)
          ..strokeWidth = 3.5);
    final dash = Paint()
      ..color = AppColors.lavender.withAlpha(120)
      ..strokeWidth = 2;
    for (var dx = pad; dx < size.width - pad; dx += 20) {
      canvas.drawLine(Offset(dx, y(wl.xHeight)), Offset(math.min(dx + 11, size.width - pad), y(wl.xHeight)), dash);
    }
    if (wl.descender != null) {
      canvas.drawLine(
          Offset(pad, y(wl.descender!)),
          Offset(size.width - pad, y(wl.descender!)),
          Paint()
            ..color = AppColors.lavender.withAlpha(80)
            ..strokeWidth = 2);
    }
  }

  void _paintArrows(Canvas canvas, PolylineWalker w, Color color, double width) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = math.max(2.5, width / 2.6)
      ..strokeCap = StrokeCap.round;
    final fractions = w.length > 240 ? const [0.3, 0.7] : const [0.55];
    for (final f in fractions) {
      final (p, heading) = w.at(f);
      final len = width * 1.1;
      final left = p - Offset(math.cos(heading - 0.6), math.sin(heading - 0.6)) * len;
      final right = p - Offset(math.cos(heading + 0.6), math.sin(heading + 0.6)) * len;
      canvas.drawLine(left, p, paint);
      canvas.drawLine(right, p, paint);
    }
  }

  void _paintStartMarker(Canvas canvas, Offset at, int number, Color color, double width) {
    final r = math.max(13.0, width * 1.3);
    canvas.drawCircle(at, r + 2, Paint()..color = Colors.white);
    canvas.drawCircle(at, r, Paint()..color = AppColors.startDot);
    final tp = TextPainter(
      text: TextSpan(
          text: '$number', style: TextStyle(color: Colors.white, fontSize: r * 1.15, fontWeight: FontWeight.w800)),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, at - Offset(tp.width / 2, tp.height / 2));
  }

  @override
  bool shouldRepaint(covariant LetterGuidePainter old) =>
      old.strokes != strokes ||
      old.lines != lines ||
      old.progress != progress ||
      old.showNumbers != showNumbers ||
      old.showArrows != showArrows ||
      old.templateScale != templateScale;
}
