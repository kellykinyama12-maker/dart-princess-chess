// Standalone tactical check: run the classical alpha-beta searcher (no net, no
// MCTS) over an EPD suite and report solves. Proves the AB worker's tactical
// strength independent of the MCTS integration.
//
//   dart run bin/ab_solve.dart --movetime 2000 [suite.epd]
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/alphabeta.dart';
import 'package:chess_moves/nnue.dart';

void main(List<String> args) {
  var movetime = 2000;
  var maxDepth = 64;
  String? nnuePath;
  var complexity = false;
  final files = <String>[];
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--movetime':
        movetime = int.parse(args[++i]);
      case '--depth':
        maxDepth = int.parse(args[++i]);
      case '--nnue':
        nnuePath = args[++i];
      case '--complexity':
        complexity = true;
      default:
        files.add(args[i]);
    }
  }
  if (files.isEmpty) files.add('test/tactics.epd');

  final ab = AlphaBeta(
      nnue: nnuePath != null ? Nnue.load(nnuePath) : null,
      complexity: complexity);
  var solved = 0, total = 0, totalMs = 0, totalNodes = 0;
  for (final f in files) {
    if (!File(f).existsSync()) {
      stderr.writeln('skip (missing): $f');
      continue;
    }
    for (final line in File(f).readAsLinesSync()) {
      final p = _parseEpd(line);
      if (p == null) continue;

      final board = ch.Chess.fromFEN(p.fen);
      final uciToSan = <String, String>{};
      final sanToUci = <String, String>{};
      for (final mv in board.moves({'asObjects': true}) as List) {
        final m = mv as ch.Move;
        final uci = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
        final san = _norm(board.move_to_san(m));
        uciToSan[uci] = san;
        sanToUci[san] = uci;
      }
      final want = {
        for (final bm in p.bm)
          if (sanToUci[_norm(bm)] != null) sanToUci[_norm(bm)]!
      };
      if (want.isEmpty) continue;
      total++;

      final sw = Stopwatch()..start();
      final r = ab.search(p.fen, maxDepth: maxDepth,
          time: Duration(milliseconds: movetime));
      sw.stop();
      totalMs += sw.elapsedMilliseconds;
      totalNodes += r.nodes;

      final ok = r.bestUci != null && want.contains(r.bestUci);
      if (ok) solved++;
      final san = uciToSan[r.bestUci] ?? r.bestUci ?? '----';
      final sc = r.isMate
          ? '#${((AlphaBeta.mateScore - r.scoreCp.abs() + 1) ~/ 2) * r.scoreCp.sign}'
          : '${(r.scoreCp / 100).toStringAsFixed(2)}';
      stdout.writeln('${ok ? 'OK  ' : 'FAIL'} ${p.id.padRight(10)} '
          'ab=${san.padRight(7)} bm ${p.bm.join('/').padRight(8)} '
          'd=${r.depth} sc=$sc ${r.nodes} nodes (${sw.elapsedMilliseconds} ms)');
    }
  }
  final nps = totalMs > 0 ? (totalNodes * 1000 / totalMs).round() : 0;
  stdout.writeln('\nsolved $solved / $total  '
      '(${total > 0 ? (100 * solved / total).toStringAsFixed(1) : '0'}%)  '
      '${nps} nodes/s');
}

String _norm(String san) => san.replaceAll(RegExp(r'[+#!?]'), '');

class _Epd {
  final String fen;
  final List<String> bm;
  final String id;
  _Epd(this.fen, this.bm, this.id);
}

_Epd? _parseEpd(String line) {
  final t = line.trim();
  if (t.isEmpty || t.startsWith('#')) return null;
  final parts = t.split(RegExp(r'\s+'));
  if (parts.length < 4) return null;
  var fen = parts.sublist(0, 4).join(' ');
  if (!RegExp(r'\s\d+\s\d+$').hasMatch(fen)) fen = '$fen 0 1';
  final rest = parts.length > 4 ? parts.sublist(4).join(' ') : '';
  final bm = <String>[];
  var id = '';
  final bmM = RegExp(r'bm\s+([^;]+);').firstMatch(rest);
  if (bmM != null) bm.addAll(bmM.group(1)!.trim().split(RegExp(r'\s+')));
  final idM = RegExp(r'id\s+"?([^";]+)"?;?').firstMatch(rest);
  if (idM != null) id = idM.group(1)!.trim();
  if (bm.isEmpty) return null;
  return _Epd(fen, bm, id);
}
