import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../theme/app_theme.dart';

/// Line chart of the practice scores the backend recorded in this session.
/// Draws only the values it is given. It never interpolates, smooths or
/// extrapolates.
class ScoreChart extends StatelessWidget {
  final List<double> scores;

  const ScoreChart({super.key, required this.scores});

  @override
  Widget build(BuildContext context) {
    final pcts = scores.map((s) => (s.clamp(0.0, 1.0) * 100).round()).join(', ');
    return Semantics(
      label: 'Practice scores for each attempt: $pcts percent',
      child: CustomPaint(painter: _ScorePainter(scores), size: Size.infinite),
    );
  }
}

class _ScorePainter extends CustomPainter {
  final List<double> scores;

  _ScorePainter(this.scores);

  @override
  void paint(Canvas canvas, Size size) {
    const left = 44.0, bottom = 28.0, top = 12.0, right = 12.0;
    final w = size.width - left - right, h = size.height - top - bottom;
    if (w <= 0 || h <= 0) return;
    final grid = Paint()
      ..color = AppColors.creamDeep
      ..strokeWidth = 1.5;
    final tp = TextPainter(textDirection: TextDirection.ltr);
    for (final v in const [0, 50, 100]) {
      final y = top + h * (1 - v / 100);
      canvas.drawLine(Offset(left, y), Offset(left + w, y), grid);
      tp.text = TextSpan(text: '$v%', style: const TextStyle(fontSize: 13, color: AppColors.inkMuted));
      tp.layout();
      tp.paint(canvas, Offset(left - tp.width - 6, y - tp.height / 2));
    }
    if (scores.isEmpty) return;
    final n = scores.length;
    Offset at(int i) => Offset(left + (n == 1 ? w / 2 : w * i / (n - 1)), top + h * (1 - scores[i].clamp(0.0, 1.0)));
    final line = Paint()
      ..color = AppColors.teal
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeJoin = StrokeJoin.round;
    final path = Path()..moveTo(at(0).dx, at(0).dy);
    for (var i = 1; i < n; i++) {
      path.lineTo(at(i).dx, at(i).dy);
    }
    canvas.drawPath(path, line);
    final step = math.max(1, (n / 10).ceil());
    for (var i = 0; i < n; i++) {
      canvas.drawCircle(at(i), 6, Paint()..color = AppColors.teal);
      canvas.drawCircle(at(i), 3, Paint()..color = Colors.white);
      if (i % step == 0 || i == n - 1) {
        tp.text = TextSpan(text: '${i + 1}', style: const TextStyle(fontSize: 13, color: AppColors.inkMuted));
        tp.layout();
        tp.paint(canvas, Offset(at(i).dx - tp.width / 2, top + h + 6));
      }
    }
  }

  @override
  bool shouldRepaint(covariant _ScorePainter old) => old.scores != scores;
}
