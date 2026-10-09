// Tactics harness: run the engine over EPD positions and report how many it
// solves (engine best move matches a `bm` best-move / avoids an `am` move).
//
//   dart run bin/solve_epd.dart --nodes 2000 --workers 8
//   dart run bin/solve_epd.dart --movetime 1000 path/to/suite.epd
//
// With no file argument it uses the built-in WAC subset in test/tactics.epd.
import 'dart:io';

import 'package:chess/chess.dart' as ch;

import 'package:chess_moves/eval_pool.dart';
import 'package:chess_moves/lc0_evaluator.dart';
import 'package:chess_moves/mcts.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';
const _defaultEpd = 'test/tactics.epd';

Future<void> main(List<String> args) async {
  int? nodes = 2000;
  int? movetime;
  var workers = 1;
  final files = <String>[];
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--nodes':
        nodes = int.parse(args[++i]);
        movetime = null;
      case '--movetime':
        movetime = int.parse(args[++i]);
        nodes = null;
      case '--workers':
        workers = int.parse(args[++i]);
      default:
        files.add(args[i]);
    }
  }
  if (files.isEmpty) files.add(_defaultEpd);

  if (!File(_weightsPath).existsSync()) {
    stderr.writeln('Missing $_weightsPath. Run bin/lc0_eval_smoke.dart first.');
    exit(64);
  }

  final puzzles = <_Epd>[];
  for (final f in files) {
    if (!File(f).existsSync()) {
      stderr.writeln('skip (missing): $f');
      continue;
    }
    for (final line in File(f).readAsLinesSync()) {
      final p = _parseEpd(line);
      if (p != null) puzzles.add(p);
    }
  }
  if (puzzles.isEmpty) {
    stderr.writeln('no puzzles parsed');
    exit(65);
  }

  EvalPool? pool;
  Mcts mcts;
  if (workers > 1) {
    pool = await EvalPool.start(_weightsPath, workers);
    mcts = Mcts.pooled(pool, batchSize: workers * 2, addRootNoise: false);
  } else {
    mcts = Mcts(Lc0Evaluator.load(_weightsPath), addRootNoise: false);
  }

  final budget = movetime != null
      ? 'movetime ${movetime}ms'
      : 'nodes $nodes';
  stdout.writeln('suite: ${puzzles.length} positions  budget: $budget  '
      'workers: $workers\n');

  var solved = 0;
  var totalMs = 0;
  for (final p in puzzles) {
    mcts.reset();
    final board = ch.Chess.fromFEN(p.fen);
    final uciToSan = <String, String>{};
    final sanToUci = <String, String>{};
    for (final mv in board.moves({'verbose': true}) as List) {
      final m = mv as Map;
      final uci = (m['from'] as String) +
          (m['to'] as String) +
          (m['promotion'] != null
              ? (m['promotion']).toString().toLowerCase()
              : '');
      final san = _norm(m['san'] as String);
      uciToSan[uci] = san;
      sanToUci[san] = uci;
    }
    final wantUci = {
      for (final bm in p.bm)
        if (sanToUci[_norm(bm)] != null) sanToUci[_norm(bm)]!
    };
    final avoidUci = {
      for (final am in p.am)
        if (sanToUci[_norm(am)] != null) sanToUci[_norm(am)]!
    };
    if (wantUci.isEmpty && avoidUci.isEmpty) {
      stdout.writeln('?? ${p.id}: bm/am not legal here, skipping');
      continue;
    }

    final sw = Stopwatch()..start();
    final stats = pool != null
        ? await mcts.searchBudgetAsync(p.fen,
            nodes: nodes, time: _dur(movetime))
        : mcts.searchBudget(p.fen, nodes: nodes, time: _dur(movetime));
    sw.stop();
    totalMs += sw.elapsedMilliseconds;

    final engUci = stats.isEmpty ? '----' : stats.first.uci;
    final ok = p.bm.isNotEmpty
        ? wantUci.contains(engUci)
        : !avoidUci.contains(engUci);
    if (ok) solved++;
    final tag = ok ? 'OK  ' : 'FAIL';
    final want = p.bm.isNotEmpty ? 'bm ${p.bm.join('/')}' : 'am ${p.am.join('/')}';
    stdout.writeln('$tag ${p.id.padRight(10)} '
        'engine=${(uciToSan[engUci] ?? engUci).padRight(7)} $want  '
        '(${sw.elapsedMilliseconds} ms)');
  }

  stdout.writeln('\nsolved $solved / ${puzzles.length}  '
      '(${(100 * solved / puzzles.length).toStringAsFixed(1)}%)  '
      'avg ${(totalMs / puzzles.length).toStringAsFixed(0)} ms/pos');
  pool?.dispose();
}

Duration? _dur(int? ms) => ms == null ? null : Duration(milliseconds: ms);

String _norm(String san) => san.replaceAll(RegExp(r'[+#!?]'), '');

class _Epd {
  final String fen;
  final List<String> bm;
  final List<String> am;
  final String id;
  _Epd(this.fen, this.bm, this.am, this.id);
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
  final am = <String>[];
  var id = '';
  for (final op in rest.split(';')) {
    final o = op.trim();
    if (o.startsWith('bm ')) {
      bm.addAll(o.substring(3).trim().split(RegExp(r'\s+')));
    } else if (o.startsWith('am ')) {
      am.addAll(o.substring(3).trim().split(RegExp(r'\s+')));
    } else if (o.startsWith('id ')) {
      id = o.substring(3).trim().replaceAll('"', '');
    }
  }
  return _Epd(fen, bm, am, id.isEmpty ? '(no id)' : id);
}
