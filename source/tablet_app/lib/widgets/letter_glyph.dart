import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/exercise.dart';
import '../theme/app_theme.dart';
import 'template_geometry.dart';

/// A letter drawn from the backend's own guide strokes rather than from a
/// font. The learner therefore always sees the exact letter model the tutor
/// teaches (single-storey a and g, barred capital I, ...), on tiles, on the
/// letter page and above the canvas.
///
/// Every letter is framed on the same four-line box (top line to descender
/// line), so a small letter sits on the baseline and is smaller than a
/// capital, exactly as on the writing paper. Purely visual: it reads the
/// stroke data and never changes it.
class LetterGlyph extends StatelessWidget {
  final List<List<List<double>>> strokes;
  final WritingLines? lines;
  final Color color;

  /// Draw a faint baseline (and dashed middle line) behind the letter.
  final bool showLines;

  /// Used when the backend sent no usable strokes.
  final String fallbackText;

  /// Stroke weight as a fraction of the glyph box height.
  final double weight;

  const LetterGlyph({
    super.key,
    required this.strokes,
    required this.lines,
    required this.fallbackText,
    this.color = AppColors.ink,
    this.showLines = false,
    this.weight = 0.085,
  });

  factory LetterGlyph.of(Exercise ex,
          {Key? key, Color color = AppColors.ink, bool showLines = false, double weight = 0.085}) =>
      LetterGlyph(
        key: key,
        strokes: ex.guideStrokes,
        lines: ex.writingLines,
        fallbackText: ex.target ?? '',
        color: color,
        showLines: showLines,
        weight: weight,
      );

  bool get _drawable => strokes.isNotEmpty && strokes.every((s) => s.length >= 2);

  @override
  Widget build(BuildContext context) {
    if (!_drawable) {
      return FittedBox(
        child: Text(fallbackText, style: TextStyle(fontSize: 64, fontWeight: FontWeight.w600, color: color)),
      );
    }
    return CustomPaint(
      painter: _GlyphPainter(strokes, lines, color, showLines, weight),
      size: Size.infinite,
    );
  }
}

class _GlyphPainter extends CustomPainter {
  final List<List<List<double>>> strokes;
  final WritingLines? lines;
  final Color color;
  final bool showLines;
  final double weight;

  _GlyphPainter(this.strokes, this.lines, this.color, this.showLines, this.weight);

  @override
  void paint(Canvas canvas, Size size) {
    if (size.isEmpty) return;
    final all = [for (final s in strokes) ...s];
    final wl = lines;
    // A large templateScale makes the fit hit its "fits the box" clamp, so the
    // shared four-line frame fills the box and every letter keeps its size.
    final fit = TemplateFit.compute(fitPointsFor(all, wl?.ascender, wl?.baseline, descender: wl?.descender), size,
        templateScale: 10);
    final frameH = wl == null ? size.shortestSide : (fit.map([0, wl.bottom]).dy - fit.map([0, wl.ascender]).dy).abs();
    final w = math.max(2.0, frameH * weight);

    if (showLines && wl != null) {
      final y = fit.map([all.first[0], wl.baseline]).dy;
      final xh = fit.map([all.first[0], wl.xHeight]).dy;
      final base = Paint()
        ..color = AppColors.primary.withValues(alpha: 0.28)
        ..strokeWidth = math.max(1.0, w / 4);
      canvas.drawLine(Offset(0, y), Offset(size.width, y), base);
      final dash = Paint()
        ..color = AppColors.primary.withValues(alpha: 0.18)
        ..strokeWidth = math.max(1.0, w / 5);
      for (var x = 0.0; x < size.width; x += w * 1.6) {
        canvas.drawLine(Offset(x, xh), Offset(math.min(x + w * 0.8, size.width), xh), dash);
      }
    }

    final ink = Paint()
      ..color = color
      ..strokeWidth = w
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    for (final s in strokes) {
      final pts = fit.mapAll(s);
      final path = Path()..moveTo(pts.first.dx, pts.first.dy);
      for (final p in pts.skip(1)) {
        path.lineTo(p.dx, p.dy);
      }
      canvas.drawPath(path, ink);
    }
  }

  @override
  bool shouldRepaint(covariant _GlyphPainter old) =>
      old.strokes != strokes ||
      old.lines != lines ||
      old.color != color ||
      old.showLines != showLines ||
      old.weight != weight;
}
