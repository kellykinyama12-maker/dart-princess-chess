// Validates incremental Zobrist (applyMove) against the full FEN hash over many
// random games, so the make-style key updates are provably correct before the
// MCTS relies on them.
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/zobrist.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

void main() {
  final zob = Zobrist();
  final rng = math.Random(12345);
  var plies = 0, promos = 0, castles = 0, eps = 0, games = 0, mismatches = 0;

  for (var g = 0; g < 400; g++) {
    games++;
    final board = ch.Chess.fromFEN(_startFen);
    var key = zob.ofFen(board.fen);
    for (var ply = 0; ply < 160; ply++) {
      if (board.game_over) break;
      final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
      if (moves.isEmpty) break;
      final m = moves[rng.nextInt(moves.length)];
      if (m.promotion != null) promos++;
      if ((m.flags &
              (ch.Chess.BITS_KSIDE_CASTLE | ch.Chess.BITS_QSIDE_CASTLE)) !=
          0) {
        castles++;
      }
      if ((m.flags & ch.Chess.BITS_EP_CAPTURE) != 0) eps++;

      final before = board.copy();
      board.move(m);
      key = zob.applyMove(key, before, m, board);
      final full = zob.ofFen(board.fen);
      if (key != full) {
        mismatches++;
        if (mismatches <= 5) {
          print('MISMATCH after ${m.fromAlgebraic}${m.toAlgebraic}'
              '${m.promotion?.name ?? ''} flags=${m.flags}');
          print('  fen=${board.fen}');
          print('  incremental=$key full=$full');
        }
      }
      plies++;
    }
  }

  print('games=$games plies=$plies promos=$promos castles=$castles eps=$eps');
  print(mismatches == 0
      ? 'ZOBRIST-INCREMENTAL-OK (all keys match)'
      : 'ZOBRIST-FAIL: $mismatches mismatches');
}
