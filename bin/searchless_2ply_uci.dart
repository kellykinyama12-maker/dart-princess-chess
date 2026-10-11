// UCI engine: 2-ply minimax with the searchless-chess net as the leaf
// evaluator. The net gives a win probability for every legal move in one pass;
// a depth-1 lookahead (make our move, let the net pick the opponent's best
// reply) turns the amortized value into a short search — the refinement the
// paper shows improves play. Pure Dart. Fast with OpenBLAS (DART_PYTORCH_BLAS).
//   dart run bin/searchless_2ply_uci.dart
import 'dart:convert';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/searchless_pool.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/// Board after playing [uci] from [fen] (opponent to move), or null if illegal.
ch.Chess? _childBoard(String fen, String uci) {
  final board = ch.Chess.fromFEN(fen);
  final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
  for (final m in moves) {
    final mu = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
    if (mu == uci) {
      board.make_move(m);
      return board;
    }
  }
  return null;
}

Future<void> main(List<String> args) async {
  var npzPath = 'models/searchless/params.npz';
  var workers = (Platform.numberOfProcessors - 1).clamp(1, 15);
  var topK = 10; // number of 1-ply candidates refined with the 2-ply lookahead
  SearchlessPool? pool;
  var board = ch.Chess.fromFEN(_startFen);

  Future<void> ensure() async {
    pool ??= await SearchlessPool.start(npzPath, workers);
  }

  void applyMoves(List<String> ucis) {
    for (final uci in ucis) {
      final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
      for (final m in moves) {
        final mu = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
        if (mu == uci) {
          board.make_move(m);
          break;
        }
      }
    }
  }

  Future<String> pickMove() async {
    final fen = board.generate_fen();
    final q = await pool!.evaluate(fen);
    if (q.isEmpty) return '0000';
    final ranked = q.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    final k = topK < ranked.length ? topK : ranked.length;

    var best = ranked.first.key;
    var bestVal = double.negativeInfinity;
    for (var i = 0; i < k; i++) {
      final a = ranked[i].key;
      final child = _childBoard(fen, a);
      double val;
      if (child == null) {
        continue;
      } else if (child.in_checkmate) {
        val = 1.0; // our move delivers mate
      } else if (child.in_stalemate || child.in_draw) {
        val = 0.5;
      } else {
        final cq = await pool!.evaluate(child.generate_fen());
        var oppBest = double.negativeInfinity;
        cq.forEach((_, w) {
          if (w > oppBest) oppBest = w;
        });
        // Opponent plays their best reply; our win prob is the complement.
        val = oppBest.isFinite ? 1.0 - oppBest : 0.5;
      }
      // Higher 2-ply value wins; ties fall to the better 1-ply rank (loop order).
      if (val > bestVal + 1e-9) {
        bestVal = val;
        best = a;
      }
    }
    stdout.writeln('info string pick=$best 2ply=${(bestVal * 100).toStringAsFixed(1)}%'
        ' net1ply=${(ranked.first.value * 100).toStringAsFixed(1)}% (${ranked.first.key})');
    return best;
  }

  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    final s = line.trim();
    if (s == 'uci') {
      stdout.writeln('id name Searchless2ply');
      stdout.writeln('id author chess_moves');
      stdout.writeln('option name WeightsFile type string default $npzPath');
      stdout.writeln('option name Workers type spin default $workers min 1 max 64');
      stdout.writeln('option name TopK type spin default $topK min 1 max 64');
      stdout.writeln('uciok');
    } else if (s == 'isready') {
      await ensure();
      stdout.writeln('readyok');
    } else if (s.startsWith('setoption')) {
      final m = RegExp(r'name\s+(.+?)\s+value\s+(.+)$').firstMatch(s);
      if (m != null) {
        final name = m.group(1)!.trim().toLowerCase();
        final value = m.group(2)!.trim();
        switch (name) {
          case 'weightsfile':
            npzPath = value;
            pool = null;
            break;
          case 'workers':
            workers = int.tryParse(value) ?? workers;
            pool = null;
            break;
          case 'topk':
            topK = int.tryParse(value) ?? topK;
            break;
        }
      }
    } else if (s == 'ucinewgame') {
      board = ch.Chess.fromFEN(_startFen);
    } else if (s.startsWith('position')) {
      final movesIdx = s.indexOf(' moves ');
      final head = movesIdx >= 0 ? s.substring(0, movesIdx) : s;
      if (head.contains('startpos')) {
        board = ch.Chess.fromFEN(_startFen);
      } else {
        final fenIdx = head.indexOf('fen ');
        if (fenIdx >= 0) board = ch.Chess.fromFEN(head.substring(fenIdx + 4).trim());
      }
      if (movesIdx >= 0) {
        applyMoves(s.substring(movesIdx + 7).trim().split(RegExp(r'\s+')));
      }
    } else if (s.startsWith('go')) {
      await ensure();
      stdout.writeln('bestmove ${await pickMove()}');
    } else if (s == 'quit') {
      break;
    }
  }
}
