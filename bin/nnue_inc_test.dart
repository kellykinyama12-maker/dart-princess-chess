// Validates the incremental NNUE accumulator against full recompute over a
// random game (exercises captures, promotions, castling, en passant, king
// moves).  dart run bin/nnue_inc_test.dart
import 'dart:math';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/nnue.dart';

void main() {
  final net = Nnue.load('models/nnue/nn-62ef826d1a6d.nnue');
  final rng = Random(12345);
  var mismatches = 0, total = 0, games = 0;
  for (var g = 0; g < 40; g++) {
    final b = ch.Chess();
    final acc = NnueAccumulator(net);
    acc.reset(b);
    games++;
    for (var ply = 0; ply < 160; ply++) {
      final moves = b.generate_moves();
      if (moves.isEmpty) break;
      final m = moves[rng.nextInt(moves.length)];
      b.make_move(m);
      acc.push(m, b);
      final inc = acc.eval(b.turn == ch.Chess.WHITE);
      final full = net.evaluate(b);
      total++;
      if (inc != full) {
        mismatches++;
        if (mismatches <= 8) {
          print('mismatch g$g ply$ply: inc=$inc full=$full '
              'move=${m.fromAlgebraic}${m.toAlgebraic} '
              'piece=${m.piece.name} flags=${m.flags}');
        }
      }
      // also test pop/push round-trip by undoing and redoing occasionally
    }
  }
  print('$mismatches / $total mismatches over $games games');
  print(mismatches == 0 ? 'INCREMENTAL OK' : 'INCREMENTAL BROKEN');
}
