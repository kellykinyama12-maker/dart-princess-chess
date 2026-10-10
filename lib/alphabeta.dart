import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

/// Result of an alpha-beta search of one position.
class AbResult {
  final String? bestUci; // best move (absolute UCI), null if none
  final int scoreCp; // centipawns from side-to-move POV (mate = +-mateScore)
  final int depth; // deepest ply fully completed
  final List<String> pv; // principal variation (absolute UCI)
  final int nodes;
  AbResult(this.bestUci, this.scoreCp, this.depth, this.pv, this.nodes);

  bool get isMate => scoreCp.abs() >= AlphaBeta.mateScore - 1000;
}

class _Tt {
  int depth;
  int score;
  int flag; // 0 exact, 1 lower bound, 2 upper bound
  int from;
  int to;
  _Tt(this.depth, this.score, this.flag, this.from, this.to);
}

/// Classical negamax alpha-beta with iterative deepening, a transposition
/// table, MVV-LVA move ordering and a capture quiescence search. Uses a
/// material + piece-square-table evaluation. This is the tactical counterpart
/// to the MCTS/net search: it searches far deeper on sharp lines.
class AlphaBeta {
  static const int mateScore = 30000;
  static const int _inf = 1 << 30;

  // Centipawn piece values indexed by PieceType.shift (p,n,b,r,q,k).
  static const List<int> _pieceVal = [100, 320, 330, 500, 900, 0];

  final ch.Chess _b = ch.Chess();
  final Map<int, _Tt> tt;
  final List<int> _zPiece; // [12*64] zobrist keys
  late final int _zTurn;
  final List<int> _zCastle; // [4]
  final List<int> _zEp; // [8]

  int _nodes = 0;
  DateTime? _deadline;
  bool _aborted = false;

  AlphaBeta({Map<int, _Tt>? sharedTt, int seed = 0x9e3779b9})
      : tt = sharedTt ?? <int, _Tt>{},
        _zPiece = List<int>.filled(12 * 64, 0),
        _zCastle = List<int>.filled(4, 0),
        _zEp = List<int>.filled(8, 0) {
    final r = math.Random(seed);
    int rnd() => (r.nextInt(1 << 32) << 32) ^ r.nextInt(1 << 32);
    for (var i = 0; i < _zPiece.length; i++) {
      _zPiece[i] = rnd();
    }
    _zTurn = rnd();
    for (var i = 0; i < 4; i++) {
      _zCastle[i] = rnd();
    }
    for (var i = 0; i < 8; i++) {
      _zEp[i] = rnd();
    }
  }

  int get nodes => _nodes;

  /// Iterative-deepening search of [fen]. Stops at [maxDepth] or when [time]
  /// elapses (whichever first). A found mate ends the search early. When
  /// [rootPriors] (move UCI -> net policy prior, typically from the MCTS tree)
  /// is given, the root moves are ordered by prior; deeper nodes use MVV-LVA.
  AbResult search(String fen,
      {int maxDepth = 64, Duration? time, Map<String, double>? rootPriors}) {
    _b.load(fen, check_validity: false);
    _deadline = time == null ? null : DateTime.now().add(time);
    _nodes = 0;

    ch.Move? bestMove;
    var bestScore = 0;
    var reached = 0;

    for (var depth = 1; depth <= maxDepth; depth++) {
      _aborted = false;
      final rootMoves = _b.generate_moves();
      if (rootMoves.isEmpty) break;
      _order(rootMoves, bestMove, rootPriors);

      var alpha = -_inf;
      const beta = _inf;
      var localBest = -_inf;
      ch.Move? localBestMove;
      final inCheck = _b.king_attacked(_b.turn);
      var moveCount = 0;
      for (final m in rootMoves) {
        final isQuiet = (m.flags &
                (ch.Chess.BITS_CAPTURE |
                    ch.Chess.BITS_EP_CAPTURE |
                    ch.Chess.BITS_PROMOTION)) ==
            0;
        final liked = _netLikes(m, rootPriors); // never reduce a move MCTS likes
        _b.make_move(m);
        final givesCheck = _b.king_attacked(_b.turn);
        int score;
        if (moveCount == 0) {
          score = -_negamax(depth - 1, -beta, -alpha, 1);
        } else {
          var r = 0;
          if (depth >= 3 &&
              moveCount >= 4 &&
              isQuiet &&
              !liked &&
              !inCheck &&
              !givesCheck) {
            r = 1;
          }
          score = -_negamax(depth - 1 - r, -alpha - 1, -alpha, 1);
          if (!_aborted && score > alpha) {
            score = -_negamax(depth - 1, -beta, -alpha, 1);
          }
        }
        _b.undo_move();
        if (_aborted) break;
        moveCount++;
        if (score > localBest) {
          localBest = score;
          localBestMove = m;
        }
        if (score > alpha) alpha = score;
      }
      if (_aborted) break; // discard the half-finished depth
      bestScore = localBest;
      bestMove = localBestMove;
      reached = depth;
      _store(_zobrist(), depth, bestScore, 0, bestMove);
      if (bestScore.abs() >= mateScore - 1000) break;
    }

    return AbResult(
        bestMove == null ? null : _uci(bestMove), bestScore, reached,
        _extractPv(16), _nodes);
  }

  int _negamax(int depth, int alpha, int beta, int ply) {
    if ((_nodes & 2047) == 0 && _timeUp()) {
      _aborted = true;
      return 0;
    }
    _nodes++;

    final key = _zobrist();
    final tte = tt[key];
    ch.Move? ttMove;
    if (tte != null) {
      if (tte.depth >= depth) {
        if (tte.flag == 0) return tte.score;
        if (tte.flag == 1 && tte.score >= beta) return tte.score;
        if (tte.flag == 2 && tte.score <= alpha) return tte.score;
      }
      ttMove = _moveFromSquares(tte.from, tte.to);
    }

    if (depth <= 0) return _quiesce(alpha, beta, ply);

    final moves = _b.generate_moves();
    if (moves.isEmpty) {
      // No legal moves: checkmate (bad) or stalemate (draw).
      return _b.king_attacked(_b.turn) ? -(mateScore - ply) : 0;
    }
    _order(moves, ttMove);

    final inCheck = _b.king_attacked(_b.turn);
    final origAlpha = alpha;
    var best = -_inf;
    ch.Move? bestMove;
    var moveCount = 0;
    for (final m in moves) {
      final isQuiet = (m.flags &
              (ch.Chess.BITS_CAPTURE |
                  ch.Chess.BITS_EP_CAPTURE |
                  ch.Chess.BITS_PROMOTION)) ==
          0;
      _b.make_move(m);
      final givesCheck = _b.king_attacked(_b.turn);
      int score;
      if (moveCount == 0) {
        score = -_negamax(depth - 1, -beta, -alpha, ply + 1);
      } else {
        var r = 0;
        if (depth >= 3 &&
            ply >= 3 &&
            moveCount >= 4 &&
            isQuiet &&
            !inCheck &&
            !givesCheck) {
          r = 1;
        }
        score = -_negamax(depth - 1 - r, -alpha - 1, -alpha, ply + 1);
        if (!_aborted && score > alpha) {
          score = -_negamax(depth - 1, -beta, -alpha, ply + 1);
        }
      }
      _b.undo_move();
      if (_aborted) return 0;
      moveCount++;
      if (score > best) {
        best = score;
        bestMove = m;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }

    final flag = best <= origAlpha ? 2 : (best >= beta ? 1 : 0);
    _store(key, depth, best, flag, bestMove);
    return best;
  }

  /// Whether the net policy ([priors], move UCI -> prior) rates [m] highly
  /// enough that it should never be depth-reduced (a move MCTS likes).
  bool _netLikes(ch.Move m, Map<String, double>? priors) {
    if (priors == null) return false;
    final uci = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
    final p = priors[uci];
    return p != null && p >= 0.05;
  }

  /// Capture/promotion-only search to a quiet position (avoids horizon tactics).
  int _quiesce(int alpha, int beta, int ply) {
    _nodes++;
    final standPat = _evalStm();
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;

    final moves = _b.generate_moves();
    final caps = <ch.Move>[];
    for (final m in moves) {
      if ((m.flags & (ch.Chess.BITS_CAPTURE | ch.Chess.BITS_EP_CAPTURE)) != 0 ||
          (m.flags & ch.Chess.BITS_PROMOTION) != 0) {
        caps.add(m);
      }
    }
    _order(caps, null);

    for (final m in caps) {
      _b.make_move(m);
      final score = -_quiesce(-beta, -alpha, ply + 1);
      _b.undo_move();
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  /// Evaluation (material + piece-square tables) from side-to-move POV.
  int _evalStm() {
    var white = 0;
    for (var i = ch.Chess.SQUARES_A8; i <= ch.Chess.SQUARES_H1; i++) {
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }
      final p = _b.board[i];
      if (p == null) continue;
      final sq = (i >> 4) * 8 + (i & 7);
      final t = p.type.shift;
      if (p.color == ch.Chess.WHITE) {
        white += _pieceVal[t] + _pst[t][sq];
      } else {
        white -= _pieceVal[t] + _pst[t][sq ^ 56];
      }
    }
    return _b.turn == ch.Chess.WHITE ? white : -white;
  }

  void _order(List<ch.Move> moves, ch.Move? first,
      [Map<String, double>? priors]) {
    int score(ch.Move m) {
      if (first != null && m.from == first.from && m.to == first.to) {
        return 1 << 28;
      }
      if (priors != null) {
        final uci =
            m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
        final p = priors[uci];
        // Net prior outranks MVV-LVA but stays below the TT move.
        if (p != null) return (1 << 20) + (p * 1000000).round();
      }
      var s = 0;
      final cap = m.captured;
      if (cap != null) {
        s += 100000 + _pieceVal[cap.shift] * 10 - _pieceVal[m.piece.shift];
      }
      final promo = m.promotion;
      if (promo != null) s += 90000 + _pieceVal[promo.shift];
      return s;
    }

    moves.sort((a, b) => score(b).compareTo(score(a)));
  }

  void _store(int key, int depth, int score, int flag, ch.Move? m) {
    final prev = tt[key];
    if (prev != null && prev.depth > depth && prev.flag == 0 && flag != 0) {
      return; // keep the deeper exact entry
    }
    tt[key] = _Tt(depth, score, flag, m?.from ?? -1, m?.to ?? -1);
  }

  ch.Move? _moveFromSquares(int from, int to) {
    if (from < 0) return null;
    for (final m in _b.generate_moves()) {
      if (m.from == from && m.to == to) return m;
    }
    return null;
  }

  List<String> _extractPv(int maxLen) {
    final pv = <String>[];
    final made = <ch.Move>[];
    while (pv.length < maxLen) {
      final tte = tt[_zobrist()];
      if (tte == null || tte.from < 0) break;
      final m = _moveFromSquares(tte.from, tte.to);
      if (m == null) break;
      pv.add(_uci(m));
      _b.make_move(m);
      made.add(m);
    }
    for (var i = made.length - 1; i >= 0; i--) {
      _b.undo_move();
    }
    return pv;
  }

  bool _timeUp() => _deadline != null && !DateTime.now().isBefore(_deadline!);

  String _uci(ch.Move m) =>
      m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');

  int _zobrist() {
    var h = 0;
    for (var i = ch.Chess.SQUARES_A8; i <= ch.Chess.SQUARES_H1; i++) {
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }
      final p = _b.board[i];
      if (p == null) continue;
      final sq = (i >> 4) * 8 + (i & 7);
      final idx = (p.color == ch.Chess.WHITE ? 0 : 6) + p.type.shift;
      h ^= _zPiece[idx * 64 + sq];
    }
    if (_b.turn == ch.Chess.BLACK) h ^= _zTurn;
    final cw = _b.castling[ch.Chess.WHITE];
    final cb = _b.castling[ch.Chess.BLACK];
    if ((cw & ch.Chess.BITS_KSIDE_CASTLE) != 0) h ^= _zCastle[0];
    if ((cw & ch.Chess.BITS_QSIDE_CASTLE) != 0) h ^= _zCastle[1];
    if ((cb & ch.Chess.BITS_KSIDE_CASTLE) != 0) h ^= _zCastle[2];
    if ((cb & ch.Chess.BITS_QSIDE_CASTLE) != 0) h ^= _zCastle[3];
    final ep = _b.ep_square;
    if (ep != null && ep >= 0) h ^= _zEp[ep & 7];
    return h;
  }

  // Piece-square tables (middlegame), white POV, a8..h1 reading order.
  static const List<List<int>> _pst = [
    // pawn
    [
      0, 0, 0, 0, 0, 0, 0, 0, //
      50, 50, 50, 50, 50, 50, 50, 50,
      10, 10, 20, 30, 30, 20, 10, 10,
      5, 5, 10, 25, 25, 10, 5, 5,
      0, 0, 0, 20, 20, 0, 0, 0,
      5, -5, -10, 0, 0, -10, -5, 5,
      5, 10, 10, -20, -20, 10, 10, 5,
      0, 0, 0, 0, 0, 0, 0, 0,
    ],
    // knight
    [
      -50, -40, -30, -30, -30, -30, -40, -50, //
      -40, -20, 0, 0, 0, 0, -20, -40,
      -30, 0, 10, 15, 15, 10, 0, -30,
      -30, 5, 15, 20, 20, 15, 5, -30,
      -30, 0, 15, 20, 20, 15, 0, -30,
      -30, 5, 10, 15, 15, 10, 5, -30,
      -40, -20, 0, 5, 5, 0, -20, -40,
      -50, -40, -30, -30, -30, -30, -40, -50,
    ],
    // bishop
    [
      -20, -10, -10, -10, -10, -10, -10, -20, //
      -10, 0, 0, 0, 0, 0, 0, -10,
      -10, 0, 5, 10, 10, 5, 0, -10,
      -10, 5, 5, 10, 10, 5, 5, -10,
      -10, 0, 10, 10, 10, 10, 0, -10,
      -10, 10, 10, 10, 10, 10, 10, -10,
      -10, 5, 0, 0, 0, 0, 5, -10,
      -20, -10, -10, -10, -10, -10, -10, -20,
    ],
    // rook
    [
      0, 0, 0, 0, 0, 0, 0, 0, //
      5, 10, 10, 10, 10, 10, 10, 5,
      -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5,
      -5, 0, 0, 0, 0, 0, 0, -5,
      0, 0, 0, 5, 5, 0, 0, 0,
    ],
    // queen
    [
      -20, -10, -10, -5, -5, -10, -10, -20, //
      -10, 0, 0, 0, 0, 0, 0, -10,
      -10, 0, 5, 5, 5, 5, 0, -10,
      -5, 0, 5, 5, 5, 5, 0, -5,
      0, 0, 5, 5, 5, 5, 0, -5,
      -10, 5, 5, 5, 5, 5, 0, -10,
      -10, 0, 5, 0, 0, 0, 0, -10,
      -20, -10, -10, -5, -5, -10, -10, -20,
    ],
    // king (middlegame)
    [
      -30, -40, -40, -50, -50, -40, -40, -30, //
      -30, -40, -40, -50, -50, -40, -40, -30,
      -30, -40, -40, -50, -50, -40, -40, -30,
      -30, -40, -40, -50, -50, -40, -40, -30,
      -20, -30, -30, -40, -40, -30, -30, -20,
      -10, -20, -20, -20, -20, -20, -20, -10,
      20, 20, 0, 0, 0, 0, 20, 20,
      20, 30, 10, 0, 0, 10, 30, 20,
    ],
  ];
}
