import 'package:flutter/material.dart';

import '../theme/app_theme.dart';
import 'template_geometry.dart';

/// Small static drawing of an exercise guide (selection cards, prompt bar).
/// Each pen stroke is drawn separately, so a letter such as "A" is not joined
/// by false connecting lines. The start dot marks where the first stroke begins.
class ShapePreview extends StatelessWidget {
  final List<List<List<double>>> guideStrokes;
  final Color color;

  /// Ascender and baseline (letters and words), so "c" previews smaller than "C".
  final double? ascender;
  final double? baseline;

  /// Descender line (letters), so letters with tails keep their size too.
  final double? descender;

  /// Single-polyline template (original constructor, kept for compatibility).
  ShapePreview({super.key, required List<List<double>> template, this.color = AppColors.teal})
      : guideStrokes = template.length >= 2 ? [template] : const [],
        ascender = null,
        baseline = null,
        descender = null;

  /// Multi-stroke guide.
  const ShapePreview.strokes(
      {super.key,
      required List<List<List<double>>> strokes,
      this.color = AppColors.teal,
      this.ascender,
      this.baseline,
      this.descender})
      : guideStrokes = strokes;

  @override
  Widget build(BuildContext context) =>
      CustomPaint(painter: _ShapePainter(guideStrokes, color, ascender, baseline, descender), size: Size.infinite);
}

class _ShapePainter extends CustomPainter {
  final List<List<List<double>>> strokes;
  final Color color;
  final double? ascender;
  final double? baseline;
  final double? descender;

  _ShapePainter(this.strokes, this.color, this.ascender, this.baseline, this.descender);

  @override
  void paint(Canvas canvas, Size size) {
    final all = [for (final s in strokes) ...s];
    if (all.length < 2 || size.isEmpty) return;
    final fit =
        TemplateFit.compute(fitPointsFor(all, ascender, baseline, descender: descender), size, templateScale: 1.6);
    if (ascender != null && baseline != null) {
      final line = Paint()
        ..color = AppColors.lavender.withAlpha(90)
        ..strokeWidth = 1.5;
      for (final y in [ascender!, baseline!]) {
        final dy = fit.map([all.first[0], y]).dy;
        canvas.drawLine(Offset(0, dy), Offset(size.width, dy), line);
      }
    }
    final paint = Paint()
      ..color = color
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    for (final s in strokes) {
      if (s.length < 2) continue;
      final pts = fit.mapAll(s);
      final path = Path()..moveTo(pts.first.dx, pts.first.dy);
      for (final p in pts.skip(1)) {
        path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(path, paint);
    }
    canvas.drawCircle(fit.map(all.first), 5, Paint()..color = AppColors.startDot);
  }

  @override
  bool shouldRepaint(covariant _ShapePainter old) =>
      old.strokes != strokes ||
      old.color != color ||
      old.ascender != ascender ||
      old.baseline != baseline ||
      old.descender != descender;
}
