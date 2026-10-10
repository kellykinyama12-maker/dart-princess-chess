import 'package:chess/chess.dart' as ch;

import 'zobrist.dart';

/// Shared Zobrist (fixed seed -> identical tables across isolates, so keys
/// computed on workers match the main isolate).
final Zobrist boardZobrist = Zobrist();

/// Legal-move expansion of a position: the chess work (move generation +
/// resulting child FENs, or terminal detection) with no network involved.
/// Shared by the sync MCTS path and the worker isolates so board work can run
/// off the main isolate.
class BoardExpansion {
  final bool gameOver;
  final bool checkmate;
  final List<String> legal; // absolute UCI moves
  final List<String> childFens; // aligned with [legal]
  final List<int> childKeys; // Zobrist keys, aligned with [legal]

  BoardExpansion(
      this.gameOver, this.checkmate, this.legal, this.childFens, this.childKeys);
}

/// Expands [fen] (whose Zobrist key is [parentKey]). Child keys are derived
/// incrementally from [parentKey] with one make per move.
BoardExpansion expandBoard(String fen, int parentKey) {
  final board = ch.Chess.fromFEN(fen);
  if (board.game_over) {
    return BoardExpansion(
        true, board.in_checkmate, const [], const [], const []);
  }
  final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
  final legal = <String>[];
  final childFens = <String>[];
  final childKeys = <int>[];
  for (final m in moves) {
    // chess 0.8.1's verbose map omits the promotion field, so read it off the
    // Move object (PieceType.name is 'q'/'r'/'b'/'n').
    final uci = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
    final child = ch.Chess.fromFEN(fen);
    child.move(m);
    legal.add(uci);
    childFens.add(child.fen);
    childKeys.add(boardZobrist.applyMove(parentKey, board, m, child));
  }
  return BoardExpansion(false, false, legal, childFens, childKeys);
}
