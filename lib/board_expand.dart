import 'package:chess/chess.dart' as ch;

/// Legal-move expansion of a position: the chess work (move generation +
/// resulting child FENs, or terminal detection) with no network involved.
/// Shared by the sync MCTS path and the worker isolates so board work can run
/// off the main isolate.
class BoardExpansion {
  final bool gameOver;
  final bool checkmate;
  final List<String> legal; // absolute UCI moves
  final List<String> childFens; // aligned with [legal]

  BoardExpansion(this.gameOver, this.checkmate, this.legal, this.childFens);
}

BoardExpansion expandBoard(String fen) {
  final board = ch.Chess.fromFEN(fen);
  if (board.game_over) {
    return BoardExpansion(true, board.in_checkmate, const [], const []);
  }
  final verbose = board.moves({'asObjects': true}) as List;
  final legal = <String>[];
  final childFens = <String>[];
  for (final mv in verbose) {
    final m = mv as ch.Move;
    // chess 0.8.1's verbose map omits the promotion field, so read it off the
    // Move object (PieceType.name is 'q'/'r'/'b'/'n').
    final uci = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
    final child = ch.Chess.fromFEN(fen);
    child.move(m);
    legal.add(uci);
    childFens.add(child.fen);
  }
  return BoardExpansion(false, false, legal, childFens);
}
