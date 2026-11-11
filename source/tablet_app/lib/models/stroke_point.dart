/// A single raw sample captured from the stylus/touch input.
///
/// `t` is seconds elapsed since the current attempt started (NOT
/// wall-clock time) so the backend's dynamic-feature extractor doesn't
/// need to know anything about device clocks.
class StrokePoint {
  final double x;
  final double y;
  final double t;
  final double? pressure; // null when the device/input doesn't report pressure
  final int strokeId;
  final bool isStrokeStart;
  final bool isStrokeEnd;

  const StrokePoint({
    required this.x,
    required this.y,
    required this.t,
    required this.strokeId,
    this.pressure,
    this.isStrokeStart = false,
    this.isStrokeEnd = false,
  });

  Map<String, dynamic> toJson() => {
        'x': x,
        'y': y,
        't': t,
        'pressure': pressure,
        'stroke_id': strokeId,
        'is_stroke_start': isStrokeStart,
        'is_stroke_end': isStrokeEnd,
      };
}
