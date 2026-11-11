import 'dart:math' as math;

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';

import '../models/exercise.dart';
import '../models/exercise_instruction.dart';
import '../models/stroke_point.dart';
import '../theme/app_theme.dart';
import 'template_geometry.dart';

/// Captures raw stylus/touch input as strokes of [StrokePoint]s and draws
/// them over the exercise guide.
///
/// Capture rules (Phase B/C data-integrity fixes):
///  * Uses raw [Listener] pointer events, not a pan gesture recogniser. The
///    old GestureDetector dropped the first few pixels of every stroke
///    (pan slop) and ignored taps entirely, so dots and short strokes were lost.
///  * `t` comes from the **device event timestamp** ([PointerEvent.timeStamp]),
///    relative to the first pointer-down of the attempt. It is not taken from
///    a UI-thread stopwatch, which adds frame-scheduling jitter.
///  * `pressure` is normalised to [0, 1] only when the device reports a real
///    range (pressureMax > pressureMin). Otherwise it is null. No value is invented.
///  * Only one pointer draws at a time. If a stylus touches down while a
///    finger or palm is drawing, the touch stroke is dropped and the stylus
///    takes over (simple palm rejection).
///
/// This widget does no scoring and makes no network calls.
class HandwritingCanvas extends StatefulWidget {
  final List<List<double>>? referenceTemplate; // exercise template coords (0-100)

  /// The guide as separate pen strokes (letters and words). When given, it
  /// is drawn instead of [referenceTemplate], one stroke at a time, so no
  /// false connecting lines appear between strokes.
  final List<List<List<double>>>? templateStrokes;

  /// Letters and words: ascender / x-height / baseline drawn as lined paper,
  /// and used to size the guide (see fitPointsFor).
  final WritingLines? writingLines;

  /// Screen-reader label for the writing area (built from the instruction).
  final String semanticLabel;
  final ValueChanged<List<List<StrokePoint>>> onStrokesChanged;
  final double templateScale;

  /// "tracing": guide drawn on the canvas. "free_writing": guide hidden.
  final String mode;
  final bool enabled;

  const HandwritingCanvas({
    super.key,
    this.referenceTemplate,
    this.templateStrokes,
    this.writingLines,
    this.semanticLabel = 'Writing area. Trace the shape with your finger or stylus.',
    required this.onStrokesChanged,
    this.templateScale = 1.0,
    this.mode = 'tracing',
    this.enabled = true,
  });

  @override
  State<HandwritingCanvas> createState() => HandwritingCanvasState();
}

class HandwritingCanvasState extends State<HandwritingCanvas> with TickerProviderStateMixin {
  final List<List<StrokePoint>> _strokes = [];
  List<StrokePoint> _currentStroke = [];
  int _nextStrokeId = 0;
  int? _activePointer;
  PointerDeviceKind? _activeKind;
  Duration? _attemptOrigin;
  final Set<String> _pointerKindsSeen = {};
  Size _canvasSize = Size.zero;
  bool _demoStatic = false;

  late final AnimationController _demo = AnimationController(vsync: this, duration: const Duration(seconds: 3))
    ..addListener(() => setState(() {}));
  late final AnimationController _hint = AnimationController(vsync: this, duration: const Duration(seconds: 8))
    ..addListener(() => setState(() {}));

  @override
  void dispose() {
    _demo.dispose();
    _hint.dispose();
    super.dispose();
  }

  List<List<StrokePoint>> get strokes => List.unmodifiable(_strokes);
  List<StrokePoint> get flattenedPoints => _strokes.expand((s) => s).toList();
  bool get isDemoPlaying => _demo.isAnimating || _demoStatic;
  bool get isHintVisible => _hint.isAnimating;

  /// Facts about how this attempt was captured (sent as `capture` metadata).
  Map<String, dynamic> captureMetadata() {
    final kinds = _pointerKindsSeen;
    final kind = kinds.length == 1 ? kinds.first : 'unknown';
    return {
      'pointer_kind': kind,
      'pressure_supported': _strokes.expand((s) => s).any((p) => p.pressure != null),
      'canvas_width': _canvasSize.width,
      'canvas_height': _canvasSize.height,
    };
  }

  void clear() {
    setState(() {
      _strokes.clear();
      _currentStroke = [];
      _activePointer = null;
    });
    widget.onStrokesChanged(strokes);
  }

  void undoLastStroke() {
    setState(() {
      if (_currentStroke.isNotEmpty) {
        _currentStroke = [];
        _activePointer = null;
      } else if (_strokes.isNotEmpty) {
        _strokes.removeLast();
      }
    });
    widget.onStrokesChanged(strokes);
  }

  /// Full reset for a new attempt: clears strokes and the time origin, so
  /// `t` restarts at 0 on the next pointer-down.
  void retry() {
    setState(() {
      _strokes.clear();
      _currentStroke = [];
      _nextStrokeId = 0;
      _activePointer = null;
      _attemptOrigin = null;
      _pointerKindsSeen.clear();
      _demo.value = 0;
      _hint.value = 0;
      _demoStatic = false;
    });
    widget.onStrokesChanged(strokes);
  }

  /// REPLAY_STROKE_ANIMATION: a dot traces the guide once (3 s).
  void playDemo({bool reducedMotion = false}) {
    if (reducedMotion) {
      // No moving dot: show the whole demonstrated path for 3 s instead.
      setState(() => _demoStatic = true);
      Future.delayed(const Duration(seconds: 3), () {
        if (mounted) setState(() => _demoStatic = false);
      });
    } else {
      _demo.forward(from: 0);
    }
  }

  /// SHOW_HINT: start point and direction arrows for [seconds].
  void showHint({int seconds = 8}) {
    _hint.duration = Duration(seconds: seconds);
    _hint.forward(from: 0);
  }

  static String _kindName(PointerDeviceKind k) {
    switch (k) {
      case PointerDeviceKind.stylus:
      case PointerDeviceKind.invertedStylus:
        return 'stylus';
      case PointerDeviceKind.touch:
        return 'touch';
      case PointerDeviceKind.mouse:
        return 'mouse';
      default:
        return 'unknown';
    }
  }

  static bool _isStylus(PointerDeviceKind k) => k == PointerDeviceKind.stylus || k == PointerDeviceKind.invertedStylus;

  /// Normalised pressure, or null when the device reports no pressure range.
  static double? pressureOf(PointerEvent e) {
    if (e.pressureMax <= e.pressureMin) return null;
    return ((e.pressure - e.pressureMin) / (e.pressureMax - e.pressureMin)).clamp(0.0, 1.0).toDouble();
  }

  double _t(PointerEvent e) {
    _attemptOrigin ??= e.timeStamp;
    final us = (e.timeStamp - _attemptOrigin!).inMicroseconds;
    return math.max(0, us) / 1e6;
  }

  StrokePoint _point(PointerEvent e, {bool start = false}) => StrokePoint(
        x: e.localPosition.dx,
        y: e.localPosition.dy,
        t: _t(e),
        pressure: pressureOf(e),
        strokeId: _nextStrokeId,
        isStrokeStart: start,
      );

  void _onDown(PointerDownEvent e) {
    if (!widget.enabled) return;
    if (_activePointer != null) {
      // Palm rejection: a stylus wins over an in-progress touch stroke.
      if (_isStylus(e.kind) && _activeKind != null && !_isStylus(_activeKind!)) {
        _currentStroke = [];
      } else {
        return; // ignore extra fingers
      }
    }
    _activePointer = e.pointer;
    _activeKind = e.kind;
    _pointerKindsSeen.add(_kindName(e.kind));
    setState(() => _currentStroke = [_point(e, start: true)]);
  }

  void _onMove(PointerMoveEvent e) {
    if (e.pointer != _activePointer) return;
    setState(() => _currentStroke = [..._currentStroke, _point(e)]);
  }

  void _finishStroke() {
    if (_currentStroke.isEmpty) return;
    final last = _currentStroke.removeLast();
    final closed = StrokePoint(
      x: last.x,
      y: last.y,
      t: last.t,
      pressure: last.pressure,
      strokeId: last.strokeId,
      isStrokeStart: last.isStrokeStart,
      isStrokeEnd: true,
    );
    setState(() {
      _strokes.add([..._currentStroke, closed]);
      _currentStroke = [];
      _nextStrokeId += 1;
    });
    widget.onStrokesChanged(strokes);
  }

  void _onUp(PointerUpEvent e) {
    if (e.pointer != _activePointer) return;
    _activePointer = null;
    _finishStroke();
  }

  void _onCancel(PointerCancelEvent e) {
    if (e.pointer != _activePointer) return;
    _activePointer = null;
    setState(() => _currentStroke = []); // an OS-cancelled stroke is discarded, not completed
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      _canvasSize = Size(constraints.maxWidth, constraints.maxHeight);
      final strokes = guideStrokesFor(widget.templateStrokes, widget.referenceTemplate);
      final all = [for (final s in strokes) ...s];
      List<List<Offset>>? guide;
      List<double>? lineYs;
      if (all.length >= 2 && !_canvasSize.isEmpty) {
        final wl = widget.writingLines;
        final fit = TemplateFit.compute(
            fitPointsFor(all, wl?.ascender, wl?.baseline, descender: wl?.descender), _canvasSize,
            templateScale: widget.templateScale);
        guide = [for (final s in strokes) fit.mapAll(s)];
        if (wl != null) {
          lineYs = [
            for (final y in wl.all) fit.map([all.first[0], y]).dy
          ];
        }
      }
      return Semantics(
        label: widget.semanticLabel,
        child: Listener(
          behavior: HitTestBehavior.opaque,
          onPointerDown: _onDown,
          onPointerMove: _onMove,
          onPointerUp: _onUp,
          onPointerCancel: _onCancel,
          child: CustomPaint(
            size: Size.infinite,
            painter: HandwritingPainter(
              strokes: _strokes,
              currentStroke: _currentStroke,
              guideStrokes: guide,
              writingLineYs: lineYs,
              showGuide: guideVisibleFor(widget.mode),
              hintProgress: _hint.isAnimating ? _hint.value : null,
              demoProgress: _demoStatic ? 1.0 : (_demo.isAnimating ? _demo.value : null),
            ),
          ),
        ),
      );
    });
  }
}

/// Strokes to use as the guide: [templateStrokes] when every stroke has at
/// least two points, otherwise the single [referenceTemplate] polyline.
List<List<List<double>>> guideStrokesFor(
    List<List<List<double>>>? templateStrokes, List<List<double>>? referenceTemplate) {
  if (templateStrokes != null && templateStrokes.isNotEmpty && templateStrokes.every((s) => s.length >= 2)) {
    return templateStrokes;
  }
  if (referenceTemplate != null && referenceTemplate.length >= 2) return [referenceTemplate];
  return const [];
}

class HandwritingPainter extends CustomPainter {
  final List<List<StrokePoint>> strokes;
  final List<StrokePoint> currentStroke;

  /// Guide in canvas pixels, one list per pen stroke, in drawing order.
  final List<List<Offset>>? guideStrokes;

  /// Canvas y of the ascender, x-height and baseline (letters and words).
  /// Drawn in both modes: they are writing lines, not the tracing guide.
  final List<double>? writingLineYs;
  final bool showGuide;
  final double? hintProgress;
  final double? demoProgress;

  HandwritingPainter({
    required this.strokes,
    required this.currentStroke,
    required this.guideStrokes,
    this.writingLineYs,
    required this.showGuide,
    this.hintProgress,
    this.demoProgress,
  });

  @override
  void paint(Canvas canvas, Size size) {
    canvas.drawRect(Offset.zero & size, Paint()..color = AppColors.paper);
    final lines = writingLineYs;
    if (lines == null) {
      // Faint ruled lines give a writing-paper feel without competing with the
      // guide (shapes). Letters and words get real writing lines instead.
      final rule = Paint()
        ..color = AppColors.creamDeep
        ..strokeWidth = 1.5;
      for (var y = size.height / 6; y < size.height; y += size.height / 6) {
        canvas.drawLine(Offset(24, y), Offset(size.width - 24, y), rule);
      }
    }
    if (lines != null && lines.length >= 3) {
      // lined paper: solid ascender and baseline, dashed x-height (midline),
      // and a lighter descender line when the backend sends one
      final solid = Paint()
        ..color = AppColors.lavender.withAlpha(110)
        ..strokeWidth = 2.5;
      canvas.drawLine(Offset(16, lines[0]), Offset(size.width - 16, lines[0]), solid);
      canvas.drawLine(Offset(16, lines[2]), Offset(size.width - 16, lines[2]), solid..strokeWidth = 3.5);
      final dash = Paint()
        ..color = AppColors.lavender.withAlpha(110)
        ..strokeWidth = 2;
      for (var x = 16.0; x < size.width - 16; x += 22) {
        canvas.drawLine(Offset(x, lines[1]), Offset(math.min(x + 12, size.width - 16), lines[1]), dash);
      }
      if (lines.length >= 4) {
        final tail = Paint()
          ..color = AppColors.lavender.withAlpha(70)
          ..strokeWidth = 2;
        canvas.drawLine(Offset(16, lines[3]), Offset(size.width - 16, lines[3]), tail);
      }
    }

    final g = guideStrokes?.where((s) => s.length >= 2).toList();
    if (g != null && g.isNotEmpty) {
      if (showGuide || demoProgress != null || hintProgress != null) {
        final guidePaint = Paint()
          ..color = showGuide ? AppColors.guide : AppColors.guide.withAlpha(90)
          ..strokeWidth = 14
          ..style = PaintingStyle.stroke
          ..strokeCap = StrokeCap.round
          ..strokeJoin = StrokeJoin.round;
        for (final s in g) {
          canvas.drawPath(_path(s), guidePaint);
        }
      }
      if (hintProgress != null) _paintHint(canvas, g);
      if (demoProgress != null) _paintDemo(canvas, g, demoProgress!);
    }

    final ink = Paint()
      ..color = AppColors.ink
      ..strokeWidth = 6
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    for (final stroke in [...strokes, currentStroke]) {
      if (stroke.isEmpty) continue;
      if (stroke.length == 1) {
        canvas.drawCircle(Offset(stroke[0].x, stroke[0].y), 3.5, Paint()..color = AppColors.ink);
        continue;
      }
      canvas.drawPath(_path(stroke.map((p) => Offset(p.x, p.y)).toList()), ink);
    }
  }

  Path _path(List<Offset> pts) {
    final path = Path()..moveTo(pts.first.dx, pts.first.dy);
    for (final p in pts.skip(1)) {
      path.lineTo(p.dx, p.dy);
    }
    return path;
  }

  /// SHOW_HINT: a start dot on every stroke (numbered in drawing order when
  /// there is more than one stroke) and direction arrows along each stroke.
  void _paintHint(Canvas canvas, List<List<Offset>> g) {
    final arrow = Paint()
      ..color = AppColors.lavender
      ..strokeWidth = 4
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    final fractions = g.length == 1 ? const [0.2, 0.4, 0.6, 0.8] : const [0.35, 0.75];
    for (final s in g) {
      final walker = PolylineWalker(s);
      for (final f in fractions) {
        final (p, heading) = walker.at(f);
        const len = 18.0;
        final back = p - Offset(math.cos(heading), math.sin(heading)) * len;
        final left = p - Offset(math.cos(heading - 0.5), math.sin(heading - 0.5)) * 12;
        final right = p - Offset(math.cos(heading + 0.5), math.sin(heading + 0.5)) * 12;
        canvas.drawLine(back, p, arrow);
        canvas.drawLine(left, p, arrow);
        canvas.drawLine(right, p, arrow);
      }
    }
    for (var i = 0; i < g.length; i++) {
      final start = g[i].first;
      canvas.drawCircle(start, 16, Paint()..color = AppColors.startDot);
      if (g.length == 1) {
        canvas.drawCircle(start, 6, Paint()..color = Colors.white);
      } else {
        final tp = TextPainter(
          text: TextSpan(
              text: '${i + 1}', style: const TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.w800)),
          textDirection: TextDirection.ltr,
        )..layout();
        tp.paint(canvas, start - Offset(tp.width / 2, tp.height / 2));
      }
    }
  }

  /// REPLAY_STROKE_ANIMATION: the dot draws the strokes one after another,
  /// in order. Time is shared out by stroke length.
  void _paintDemo(Canvas canvas, List<List<Offset>> g, double f) {
    final walkers = [for (final s in g) PolylineWalker(s)];
    final total = walkers.fold<double>(0, (a, w) => a + w.length);
    var remaining = f.clamp(0.0, 1.0) * total;
    final trailPaint = Paint()
      ..color = AppColors.teal.withAlpha(140)
      ..strokeWidth = 8
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    Offset dot = g.first.first;
    for (final w in walkers) {
      if (w.length <= 0) continue;
      final frac = (remaining / w.length).clamp(0.0, 1.0);
      final trail = w.prefix(frac);
      if (trail.length >= 2) canvas.drawPath(_path(trail), trailPaint);
      dot = w.at(frac).$1;
      remaining -= w.length;
      if (remaining <= 0) break;
    }
    canvas.drawCircle(dot, 12, Paint()..color = AppColors.teal);
  }

  @override
  bool shouldRepaint(covariant HandwritingPainter old) => true;
}
