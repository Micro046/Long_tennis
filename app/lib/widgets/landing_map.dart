import 'package:flutter/material.dart';

/// Court geometry in metres (same model as pipeline/court.py).
class CourtGeom {
  static const width = 10.97, length = 23.77, singles = 8.23, net = length / 2, service = 6.40;
  static const sl = (width - singles) / 2, cx = width / 2;
  static const regions = ['deep_left', 'deep_centre', 'deep_right', 'short_left', 'short_centre', 'short_right'];
}

class LandingPoint {
  LandingPoint(this.x, this.y, this.hitterSide, this.out);
  final double x, y; // court metres (camera frame)
  final String hitterSide;
  final bool out;
}

/// The opponent's half seen from the hitter: baseline at the top, net at the bottom.
class LandingMap extends StatelessWidget {
  const LandingMap({super.key, required this.points, required this.regionPercent, required this.color});
  final List<LandingPoint> points;
  final Map<String, num?> regionPercent;
  final Color color;

  @override
  Widget build(BuildContext context) => AspectRatio(
        aspectRatio: 0.72,
        child: CustomPaint(painter: _Painter(points, regionPercent, color)),
      );
}

class _Painter extends CustomPainter {
  _Painter(this.points, this.pct, this.color);
  final List<LandingPoint> points;
  final Map<String, num?> pct;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    const margin = 3.0;
    const wM = CourtGeom.width + 2 * margin, hM = CourtGeom.net + margin;
    final s = (size.width / wM) < (size.height / hM) ? size.width / wM : size.height / hM;
    Offset o(double x, double depth) => Offset((x + margin) * s, (margin + CourtGeom.net - depth) * s);
    canvas.drawRect(Rect.fromPoints(o(0, CourtGeom.net), o(CourtGeom.width, 0)), Paint()..color = const Color(0xFF2E5E9E));
    final w = Paint()..color = Colors.white..strokeWidth = 2;
    canvas.drawLine(o(0, 0), o(CourtGeom.width, 0), w..strokeWidth = 3);
    canvas.drawLine(o(0, CourtGeom.net), o(CourtGeom.width, CourtGeom.net), w..strokeWidth = 2);
    for (final x in [0.0, CourtGeom.sl, CourtGeom.width - CourtGeom.sl, CourtGeom.width]) {
      canvas.drawLine(o(x, 0), o(x, CourtGeom.net), w);
    }
    canvas.drawLine(o(CourtGeom.sl, CourtGeom.service), o(CourtGeom.width - CourtGeom.sl, CourtGeom.service), w);
    canvas.drawLine(o(CourtGeom.cx, 0), o(CourtGeom.cx, CourtGeom.service), w);
    for (var i = 0; i < 6; i++) {
      final col = i % 3;
      final deep = i < 3;
      final cx = CourtGeom.sl + (col + 0.5) * CourtGeom.singles / 3;
      final cd = deep ? (CourtGeom.service + CourtGeom.net) / 2 : CourtGeom.service / 2;
      final v = pct[CourtGeom.regions[i]];
      final tp = TextPainter(
          text: TextSpan(text: v == null ? '–' : '${v.round()}%', style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 18)),
          textDirection: TextDirection.ltr)
        ..layout();
      tp.paint(canvas, o(cx, cd) - Offset(tp.width / 2, tp.height / 2));
    }
    for (final p in points) {
      // map camera-frame metres into "depth from net, lateral from hitter's left"
      final (vx, depth) = p.hitterSide == 'near' ? (p.x, CourtGeom.net - p.y) : (CourtGeom.width - p.x, p.y - CourtGeom.net);
      if (depth < -1) continue;
      canvas.drawCircle(o(vx, depth), 5, Paint()..color = p.out ? Colors.redAccent : color);
      canvas.drawCircle(o(vx, depth), 5, Paint()..color = Colors.black..style = PaintingStyle.stroke..strokeWidth = 1);
    }
  }

  @override
  bool shouldRepaint(covariant _Painter old) => true;
}
