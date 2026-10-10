// Minimal UCI engine: classical alpha-beta with NNUE eval (no MCTS, no net).
//   dart run bin/ab_uci.dart
import 'dart:convert';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/alphabeta.dart';
import 'package:chess_moves/nnue.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main() async {
  var nnuePath = 'models/nnue/nn-62ef826d1a6d.nnue';
  AlphaBeta? ab;
  var board = ch.Chess.fromFEN(_startFen);

  void ensure() {
    ab ??= AlphaBeta(
        nnue: File(nnuePath).existsSync() ? Nnue.load(nnuePath) : null);
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

  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    final s = line.trim();
    if (s == 'uci') {
      stdout.writeln('id name NnueAB');
      stdout.writeln('id author chess_moves');
      stdout.writeln('option name AbNnue type string default $nnuePath');
      stdout.writeln('uciok');
    } else if (s == 'isready') {
      ensure();
      stdout.writeln('readyok');
    } else if (s.startsWith('setoption')) {
      final vi = s.indexOf('value');
      if (s.contains('AbNnue') && vi >= 0) {
        nnuePath = s.substring(vi + 5).trim();
        ab = null;
      }
    } else if (s.startsWith('position')) {
      final tok = s.split(RegExp(r'\s+'));
      final movesIdx = tok.indexOf('moves');
      if (tok.length > 1 && tok[1] == 'startpos') {
        board = ch.Chess.fromFEN(_startFen);
      } else if (tok.length > 1 && tok[1] == 'fen') {
        final end = movesIdx >= 0 ? movesIdx : tok.length;
        board = ch.Chess.fromFEN(tok.sublist(2, end).join(' '));
      }
      if (movesIdx >= 0) applyMoves(tok.sublist(movesIdx + 1));
    } else if (s.startsWith('go')) {
      ensure();
      final tok = s.split(RegExp(r'\s+'));
      var movetime = 1000;
      final white = board.turn == ch.Chess.WHITE;
      for (var i = 1; i < tok.length - 1; i++) {
        switch (tok[i]) {
          case 'movetime':
            movetime = int.parse(tok[i + 1]);
          case 'wtime':
            if (white) movetime = int.parse(tok[i + 1]) ~/ 30;
          case 'btime':
            if (!white) movetime = int.parse(tok[i + 1]) ~/ 30;
        }
      }
      final r = ab!.search(board.generate_fen(),
          time: Duration(milliseconds: movetime));
      stdout.writeln('bestmove ${r.bestUci ?? '0000'}');
    } else if (s == 'quit') {
      break;
    }
  }
}
