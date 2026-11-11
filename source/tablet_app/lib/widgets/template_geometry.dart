import 'dart:math' as math;
import 'dart:ui';

/// Maps an exercise template, given in its own coordinate system (0-100 units
/// for the bundled exercises), into canvas pixels. The template keeps its
/// aspect ratio and is centred. Its size is a fraction of the canvas's shorter
/// side, multiplied by the backend's `template_scale` and clamped so the
/// guide always fits on the canvas.
///
/// Before this, the template was painted at its raw 0-100 coordinates. On a
/// tablet it was a tiny shape in the top-left corner that did not grow with
/// the canvas.
class TemplateFit {
  final double scale; // canvas px per template unit
  final Offset offset; // canvas position of template point (minX, minY)
  final double minX;
  final double minY;

  const TemplateFit._(this.scale, this.offset, this.minX, this.minY);

  static const double baseFraction = 0.55;
  static const double maxFraction = 0.92;

  factory TemplateFit.compute(List<List<double>> template, Size canvas, {double templateScale = 1.0}) {
    if (template.length < 2 || canvas.isEmpty) {
      return const TemplateFit._(1.0, Offset.zero, 0, 0);
    }
    double minX = double.infinity, minY = double.infinity;
    double maxX = -double.infinity, maxY = -double.infinity;
    for (final p in template) {
      minX = math.min(minX, p[0]);
      maxX = math.max(maxX, p[0]);
      minY = math.min(minY, p[1]);
      maxY = math.max(maxY, p[1]);
    }
    final w = math.max(maxX - minX, 1e-6);
    final h = maxY - minY; // may be 0 for a straight horizontal line
    final shortSide = math.min(canvas.width, canvas.height);
    final targetLongest = shortSide * baseFraction * templateScale;
    final longest = math.max(w, h);
    var s = targetLongest / longest;
    // Clamp so the whole template stays inside the canvas.
    final maxS = math.min(canvas.width * maxFraction / w, h > 1e-6 ? canvas.height * maxFraction / h : double.infinity);
    s = math.min(s, maxS);
    final drawnW = w * s, drawnH = h * s;
    final off = Offset((canvas.width - drawnW) / 2, (canvas.height - drawnH) / 2);
    return TemplateFit._(s, off, minX, minY);
  }

  Offset map(List<double> p) => Offset(offset.dx + (p[0] - minX) * scale, offset.dy + (p[1] - minY) * scale);

  List<Offset> mapAll(List<List<double>> template) => template.map(map).toList();
}

/// Points used to size and centre a guide: the guide itself plus, for letters
/// and words, the top and bottom writing lines, so the guide is placed on
/// lined paper instead of being stretched to its own ink. With a descender
/// line every letter shares the same frame, so a capital, a small letter and
/// a letter with a tail all keep their true relative sizes.
List<List<double>> fitPointsFor(List<List<double>> guidePoints, double? ascender, double? baseline,
    {double? descender}) {
  if (guidePoints.isEmpty || ascender == null || baseline == null) return guidePoints;
  final x = guidePoints.first[0];
  return [
    ...guidePoints,
    [x, ascender],
    [x, baseline],
    if (descender != null) [x, descender],
  ];
}

/// Cumulative arc length helper, used to place hint arrows and the demo
/// animation along the guide at evenly spaced positions.
class PolylineWalker {
  final List<Offset> points;
  final List<double> _cum;

  PolylineWalker(this.points) : _cum = _cumulative(points);

  static List<double> _cumulative(List<Offset> pts) {
    final out = <double>[0];
    for (var i = 1; i < pts.length; i++) {
      out.add(out.last + (pts[i] - pts[i - 1]).distance);
    }
    return out;
  }

  double get length => _cum.isEmpty ? 0 : _cum.last;

  /// Point and heading (radians) at fraction `f` in [0, 1] of the length.
  (Offset, double) at(double f) {
    if (points.length < 2 || length <= 0) return (points.isEmpty ? Offset.zero : points.first, 0.0);
    final target = f.clamp(0.0, 1.0) * length;
    var i = 1;
    while (i < _cum.length - 1 && _cum[i] < target) {
      i++;
    }
    final segLen = _cum[i] - _cum[i - 1];
    final t = segLen <= 0 ? 0.0 : (target - _cum[i - 1]) / segLen;
    final a = points[i - 1], b = points[i];
    final d = b - a;
    return (Offset.lerp(a, b, t)!, math.atan2(d.dy, d.dx));
  }

  /// Sub-path from the start up to fraction `f`.
  List<Offset> prefix(double f) {
    if (points.length < 2) return points;
    final target = f.clamp(0.0, 1.0) * length;
    final out = <Offset>[points.first];
    for (var i = 1; i < points.length; i++) {
      if (_cum[i] <= target) {
        out.add(points[i]);
      } else {
        out.add(at(f).$1);
        break;
      }
    }
    return out;
  }
}
