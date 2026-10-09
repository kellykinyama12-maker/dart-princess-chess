import 'dart:typed_data';

import 'lc0_v6.dart' show kV6RecordSize, toAbsoluteUci;
import 'policy_index.dart';

/// A fully-decoded lc0 V6 training record (one position). Unlike [movesFromChunk]
/// in lc0_v6.dart — which keeps only the played move — this exposes every target
/// lc0 trains on: the board planes (input), the MCTS policy distribution, the
/// value (WDL), and the moves-left target.
///
/// Field offsets follow LeelaChessZero/lczero-training chunkparser.py.
class Lc0Record {
  final ByteData _bd;
  final int base;

  Lc0Record(this._bd, this.base);

  int get version => _bd.getUint32(base, Endian.little);
  int get inputFormat => _bd.getUint32(base + 4, Endian.little);

  /// 0 = White to move, 1 = Black to move.
  int get sideToMove => inputFormat == 1
      ? _bd.getUint8(base + 8276)
      : (_bd.getUint8(base + 8278) >> 7) & 1;

  int get playedIdx => _bd.getUint16(base + 8344, Endian.little);
  int get bestIdx => _bd.getUint16(base + 8346, Endian.little);

  double get bestQ => _bd.getFloat32(base + 8284, Endian.little);
  double get bestD => _bd.getFloat32(base + 8292, Endian.little);
  double get resultQ => _bd.getFloat32(base + 8308, Endian.little);
  double get resultD => _bd.getFloat32(base + 8312, Endian.little);
  double get pliesLeft => _bd.getFloat32(base + 8304, Endian.little);

  /// Raw policy weight for move index [i] (0..1857). Illegal moves are stored
  /// as a negative sentinel; legal weights are MCTS visit fractions.
  double policyAt(int i) => _bd.getFloat32(base + 8 + i * 4, Endian.little);

  /// The move actually played, as absolute-board UCI.
  String playedUci() => toAbsoluteUci(kPolicyIndex[playedIdx], sideToMove);

  /// Game-result value from the side-to-move's POV, as [win, draw, loss].
  List<double> resultWdl() {
    final q = resultQ, d = resultD;
    return [0.5 * (1 - d + q), d, 0.5 * (1 - d - q)];
  }

  /// Search (best) value from the side-to-move's POV, as [win, draw, loss].
  List<double> bestWdl() {
    final q = bestQ, d = bestD;
    return [0.5 * (1 - d + q), d, 0.5 * (1 - d - q)];
  }

  /// Top-[k] legal moves by policy weight, as (absolute UCI, probability).
  List<MapEntry<String, double>> topPolicy(int k) {
    final hits = <MapEntry<int, double>>[];
    for (var i = 0; i < kPolicyIndex.length; i++) {
      final p = policyAt(i);
      if (p > 0) hits.add(MapEntry(i, p));
    }
    hits.sort((a, b) => b.value.compareTo(a.value));
    return hits
        .take(k)
        .map((e) =>
            MapEntry(toAbsoluteUci(kPolicyIndex[e.key], sideToMove), e.value))
        .toList();
  }

  /// One input plane (0..103) as 64 floats; bit s maps to square s. Planes are
  /// oriented from the side-to-move's POV, exactly as lc0 stores them.
  List<double> plane(int p) {
    final off = base + 7440 + p * 8;
    final lo = _bd.getUint32(off, Endian.little);
    final hi = _bd.getUint32(off + 4, Endian.little);
    final out = List<double>.filled(64, 0.0);
    for (var s = 0; s < 32; s++) {
      if ((lo >> s) & 1 == 1) out[s] = 1.0;
      if ((hi >> s) & 1 == 1) out[s + 32] = 1.0;
    }
    return out;
  }
}

/// Iterates the V6 records in a decompressed game chunk.
Iterable<Lc0Record> recordsOf(Uint8List chunk) sync* {
  final bd = ByteData.sublistView(chunk);
  final n = chunk.length ~/ kV6RecordSize;
  for (var r = 0; r < n; r++) {
    final rec = Lc0Record(bd, r * kV6RecordSize);
    if (rec.version == 6) yield rec;
  }
}
