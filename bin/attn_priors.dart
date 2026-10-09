// M4 gate: decode start-position policy priors from the attention net, using
// the chess move tables. Sane priors (e4/d4/Nf3/c4 on top) prove the whole
// attention body + policy head is correct end-to-end.
import 'dart:io';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

import 'package:chess_moves/policy_index.dart';
import 'package:chess_moves/lc0_v6.dart';

const _net = 'models/lc0/t1-256x10-distilled.pb.gz';
const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

void main() {
  if (!File(_net).existsSync()) {
    stderr.writeln('missing $_net');
    exit(64);
  }
  final w = Lc0AttnReader.readFile(_net);
  final net = Lc0AttnNet(w);
  final input = Lc0Input.fromFen(_startFen).toFloat32List();

  final sw = Stopwatch()..start();
  final out = net.forward(input);
  sw.stop();

  final idx = <String, int>{
    for (var i = 0; i < kPolicyIndex.length; i++) kPolicyIndex[i]: i
  };
  final board = ch.Chess.fromFEN(_startFen); // white to move
  final logits = <String, double>{};
  for (final mv in board.moves({'verbose': true}) as List) {
    final m = mv as Map;
    final uci = (m['from'] as String) +
        (m['to'] as String) +
        (m['promotion'] != null ? m['promotion'].toString().toLowerCase() : '');
    final mover = toAbsoluteUci(uci, 0); // white => identity
    final id = idx[mover];
    logits[uci] = id != null ? out.policy[id] : -1e9;
  }
  var mx = double.negativeInfinity;
  logits.forEach((_, v) => mx = v > mx ? v : mx);
  var denom = 0.0;
  final pri = <String, double>{};
  logits.forEach((k, v) {
    final e = math.exp(v - mx);
    pri[k] = e;
    denom += e;
  });
  pri.updateAll((k, v) => v / denom);
  final sorted = pri.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));

  stdout.writeln('wdl = W=${out.wdl[0].toStringAsFixed(3)} '
      'D=${out.wdl[1].toStringAsFixed(3)} L=${out.wdl[2].toStringAsFixed(3)}  '
      'mlh=${out.movesLeft.toStringAsFixed(1)}  (${sw.elapsedMilliseconds} ms)');
  stdout.writeln('top priors:');
  for (final e in sorted.take(8)) {
    stdout.writeln('  ${e.key}  ${(e.value * 100).toStringAsFixed(1)}%');
  }

  // Diagnostics: did squares collapse? what's the raw policy spread?
  final emb = net.embed(input);
  final body = net.encode(input);
  final e = w.embDim;
  double crossSqVar(List<double> x) {
    var total = 0.0;
    for (var c = 0; c < e; c++) {
      var m = 0.0;
      for (var s = 0; s < 64; s++) m += x[s * e + c];
      m /= 64;
      var v = 0.0;
      for (var s = 0; s < 64; s++) {
        final d = x[s * e + c] - m;
        v += d * d;
      }
      total += v / 64;
    }
    return total / e;
  }

  final vals = logits.values.where((v) => v > -1e8).toList()..sort();
  stdout.writeln('diag: emb cross-square var=${crossSqVar(emb).toStringAsFixed(4)} '
      'body cross-square var=${crossSqVar(body).toStringAsFixed(4)} '
      '(0 => squares identical)');
  stdout.writeln('diag: raw policy logit range over legal = '
      '${vals.first.toStringAsFixed(3)} .. ${vals.last.toStringAsFixed(3)}');
  // Top-5 over the FULL 1858 policy (any move, legal or not).
  final allIdx = List<int>.generate(1858, (i) => i)
    ..sort((a, b) => out.policy[b].compareTo(out.policy[a]));
  stdout.writeln('diag: top-5 full policy: ' +
      [for (var i = 0; i < 5; i++) '${kPolicyIndex[allIdx[i]]}=${out.policy[allIdx[i]].toStringAsFixed(2)}']
          .join(' '));
  var nz = 0;
  for (final v in input) {
    if (v != 0) nz++;
  }
  List<int> planesAt(int s) =>
      [for (var p = 0; p < 12; p++) if (input[p * 64 + s] != 0) p];
  stdout.writeln('diag: input nonzero=$nz; '
      'sq0=${planesAt(0)} sq8=${planesAt(8)} sq24=${planesAt(24)}');
  stdout.writeln('diag: emb sq0[0..4]='
      '${[for (var c = 0; c < 5; c++) emb[c].toStringAsFixed(3)]}');
  stdout.writeln('diag: emb sq24[0..4]='
      '${[for (var c = 0; c < 5; c++) emb[24 * e + c].toStringAsFixed(3)]}');
}
