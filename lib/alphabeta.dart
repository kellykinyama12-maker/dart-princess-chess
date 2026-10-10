import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

import 'nnue.dart';

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

  // Optional NNUE evaluator; when set it replaces the material+PST eval. Its
  // output (~3x classical centipawns) is scaled down to keep mate/thresholds
  // calibrated.
  final Nnue? nnue;
  static const int _nnueScale = 3;
  // Incremental NNUE accumulator (push on make, pop on undo) when [nnue] is set.
  late final NnueAccumulator? _acc =
      nnue != null ? NnueAccumulator(nnue!) : null;

  // Complexity-budget search: "depth" is the number of quiet moves in a line;
  // forcing moves (checks, replies to check, captures, promotions, castling)
  // cost 0 and extend for free, bounded only by [_maxPly].
  final bool complexity;
  static const int _maxPly = 64;
  int _plyCap = _maxPly; // forcing-extension depth cap for the current budget
  bool cxVerbose = false; // log the complexity ID progression
  // Quiet-move cost grows with depth: 1 near the root, more deeper, so long
  // quiet sequences are budget-prohibitive (loose at root, tight with depth).
  static const int _quietStep = 2;
  // Net policy per PV position for interior move ordering; probed between
  // iterations via [policyProbe] and cached by zobrist key.
  Map<String, double>? Function(String fen)? policyProbe;
  final Map<int, Map<String, double>> nodePriors = {};
  static const int _pvProbeLen = 8;

  AlphaBeta(
      {Map<int, _Tt>? sharedTt,
      int seed = 0x9e3779b9,
      this.nnue,
      this.complexity = false})
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

  /// make_move + incremental accumulator push.
  void _makeMove(ch.Move m) {
    _b.make_move(m);
    _acc?.push(m, _b);
  }

  /// undo_move + incremental accumulator pop.
  void _undoMove() {
    _b.undo_move();
    _acc?.pop();
  }

  /// Iterative-deepening search of [fen]. Stops at [maxDepth] or when [time]
  /// elapses (whichever first). A found mate ends the search early. When
  /// [rootPriors] (move UCI -> net policy prior, typically from the MCTS tree)
  /// is given, the root moves are ordered by prior; deeper nodes use MVV-LVA.
  AbResult search(String fen,
      {int maxDepth = 64, Duration? time, Map<String, double>? rootPriors}) {
    _b.load(fen, check_validity: false);
    _deadline = time == null ? null : DateTime.now().add(time);
    _nodes = 0;
    _acc?.reset(_b);
    if (complexity) return _searchComplexity(maxDepth, rootPriors);

    ch.Move? bestMove;
    var bestScore = 0;
    var reached = 0;

    for (var depth = 1; depth <= maxDepth; depth++) {
      _aborted = false;
      final rootMoves = _b.generate_moves();
      if (rootMoves.isEmpty) break;
      _order(rootMoves, bestMove, rootPriors);

      // Aspiration window: search a narrow band around the previous score and
      // widen on a fail-high/low, falling back to a full window.
      var alpha = -_inf;
      var beta = _inf;
      var delta = 30;
      if (depth >= 4) {
        alpha = bestScore - delta;
        beta = bestScore + delta;
      }
      while (true) {
        final (score, move) =
            _searchRoot(depth, alpha, beta, rootMoves, rootPriors);
        if (_aborted) break;
        if (score <= alpha) {
          delta *= 2;
          alpha = bestScore - delta <= -(mateScore - 1000)
              ? -_inf
              : bestScore - delta;
        } else if (score >= beta) {
          delta *= 2;
          beta = bestScore + delta >= mateScore - 1000
              ? _inf
              : bestScore + delta;
          if (move != null) bestMove = move;
        } else {
          bestScore = score;
          if (move != null) bestMove = move;
          break;
        }
      }
      if (_aborted) break; // discard the half-finished depth
      reached = depth;
      _store(_zobrist(), depth, bestScore, 0, bestMove);
      if (bestScore.abs() >= mateScore - 1000) break;
      _probePv();
    }

    return AbResult(
        bestMove == null ? null : _uci(bestMove), bestScore, reached,
        _extractPv(16), _nodes);
  }

  /// Searches all root moves within [alpha]..[beta] and returns the best score
  /// and move. Applies the net-like-protected LMR; a fail-high (alpha >= beta)
  /// stops early so the aspiration loop can widen.
  (int, ch.Move?) _searchRoot(int depth, int alpha, int beta,
      List<ch.Move> rootMoves, Map<String, double>? priors) {
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
      final liked = _netLikes(m, priors); // never reduce a move MCTS likes
      _makeMove(m);
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
      _undoMove();
      if (_aborted) break;
      moveCount++;
      if (score > localBest) {
        localBest = score;
        localBestMove = m;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break; // fail-high beyond the aspiration window
    }
    return (localBest, localBestMove);
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

    final inCheck = _b.king_attacked(_b.turn);

    // Null move pruning: if handing the opponent a free move still fails high,
    // the position is good enough to cut off. Skipped when in check or with no
    // non-pawn material (zugzwang risk), and only for real (non-mate) bounds.
    if (!inCheck &&
        depth >= 3 &&
        beta.abs() < mateScore - 1000 &&
        _hasNonPawnMaterial(_b.turn)) {
      final r = 2 + (depth >= 6 ? 1 : 0);
      final savedTurn = _b.turn;
      final savedEp = _b.ep_square;
      _b.turn = savedTurn == ch.Chess.WHITE ? ch.Chess.BLACK : ch.Chess.WHITE;
      _b.ep_square = ch.Chess.EMPTY;
      final score = -_negamax(depth - 1 - r, -beta, -beta + 1, ply + 1);
      _b.turn = savedTurn;
      _b.ep_square = savedEp;
      if (_aborted) return 0;
      if (score >= beta) return beta;
    }

    final moves = _b.generate_moves();
    if (moves.isEmpty) {
      // No legal moves: checkmate (bad) or stalemate (draw).
      return _b.king_attacked(_b.turn) ? -(mateScore - ply) : 0;
    }
    _order(moves, ttMove, null, key);

    final origAlpha = alpha;
    var best = -_inf;
    ch.Move? bestMove;
    var moveCount = 0;
    // Futility pruning: at frontier nodes, quiet moves that can't lift a
    // generous static eval to alpha are skipped.
    final canFutility =
        depth <= 2 && !inCheck && alpha.abs() < mateScore - 1000;
    final futilityBase = canFutility ? _evalStm() : 0;
    for (final m in moves) {
      final isQuiet = (m.flags &
              (ch.Chess.BITS_CAPTURE |
                  ch.Chess.BITS_EP_CAPTURE |
                  ch.Chess.BITS_PROMOTION)) ==
          0;
      _makeMove(m);
      final givesCheck = _b.king_attacked(_b.turn);
      if (canFutility &&
          moveCount > 0 &&
          isQuiet &&
          !givesCheck &&
          futilityBase + depth * 150 <= alpha) {
        _undoMove();
        continue;
      }
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
      _undoMove();
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

  /// Iterative-deepening over a complexity budget (number of quiet moves in a
  /// line); forcing moves cost 0. The reported depth is the budget reached.
  AbResult _searchComplexity(int maxBudget, Map<String, double>? rootPriors) {
    ch.Move? bestMove;
    var bestScore = 0;
    var reached = 0;
    for (var budget = 1; budget <= maxBudget; budget++) {
      _aborted = false;
      // Bound forcing extensions relative to the budget so each iteration
      // terminates; the cap grows as iterative deepening proceeds.
      _plyCap = budget + 12;
      final rootMoves = _b.generate_moves();
      if (rootMoves.isEmpty) break;
      _order(rootMoves, bestMove, rootPriors);
      final inCheck = _b.king_attacked(_b.turn);
      var alpha = -_inf;
      const beta = _inf;
      var localBest = -_inf;
      ch.Move? localBestMove;
      for (final m in rootMoves) {
        _makeMove(m);
        final cost =
            (inCheck || _isForcing(m, _b.king_attacked(_b.turn))) ? 0 : 1;
        final score = -_negamaxC(budget - cost, -beta, -alpha, 1);
        _undoMove();
        if (_aborted) break;
        if (score > localBest) {
          localBest = score;
          localBestMove = m;
        }
        if (score > alpha) alpha = score;
      }
      if (_aborted) break;
      bestScore = localBest;
      bestMove = localBestMove;
      reached = budget;
      _store(_zobrist(), budget, bestScore, 0, bestMove);
      final info = _pvInfo(32);
      if (cxVerbose) {
        print('  budget=$budget score=$bestScore pvLen=${info.$1} '
            'pvCx=${info.$2} nodes=$_nodes');
      }
      if (bestScore.abs() >= mateScore - 1000) break;
      // PV resolved within budget (used fewer quiet moves than allowed): the
      // position's complexity is known, no need to deepen further.
      if (bestMove != null && info.$2 < budget) break;
      _probePv();
    }
    return AbResult(bestMove == null ? null : _uci(bestMove), bestScore,
        reached, _extractPv(16), _nodes);
  }

  /// A move is forcing (complexity 0) if it is a capture, promotion or castle,
  /// or gives check ([givesCheck] = opponent in check after the move).
  bool _isForcing(ch.Move m, bool givesCheck) =>
      givesCheck ||
      (m.flags &
              (ch.Chess.BITS_CAPTURE |
                  ch.Chess.BITS_EP_CAPTURE |
                  ch.Chess.BITS_PROMOTION |
                  ch.Chess.BITS_KSIDE_CASTLE |
                  ch.Chess.BITS_QSIDE_CASTLE)) !=
          0;

  /// SEE-lite: a capture is losing if the capturer outvalues the captured piece
  /// and the target square is still defended by the opponent.
  bool _losingCapture(ch.Move m) {
    final captured = m.captured;
    if (captured == null) return false;
    if (_pieceVal[m.piece.shift] <= _pieceVal[captured.shift]) return false;
    final them =
        m.color == ch.Chess.WHITE ? ch.Chess.BLACK : ch.Chess.WHITE;
    return _attackedBy(m.to, them);
  }

  /// Whether [byColor] attacks 0x88 square [sq] on the current board.
  bool _attackedBy(int sq, ch.Color byColor) {
    for (final d in const [-18, -33, -31, -14, 18, 33, 31, 14]) {
      final t = sq + d;
      if ((t & 0x88) == 0) {
        final p = _b.board[t];
        if (p != null && p.color == byColor && p.type == ch.Chess.KNIGHT) {
          return true;
        }
      }
    }
    for (final d in const [-17, -16, -15, -1, 1, 15, 16, 17]) {
      final t = sq + d;
      if ((t & 0x88) == 0) {
        final p = _b.board[t];
        if (p != null && p.color == byColor && p.type == ch.Chess.KING) {
          return true;
        }
      }
    }
    final pawnDirs =
        byColor == ch.Chess.WHITE ? const [15, 17] : const [-15, -17];
    for (final d in pawnDirs) {
      final t = sq + d;
      if ((t & 0x88) == 0) {
        final p = _b.board[t];
        if (p != null && p.color == byColor && p.type == ch.Chess.PAWN) {
          return true;
        }
      }
    }
    for (final d in const [-1, 1, -16, 16]) {
      var t = sq + d;
      while ((t & 0x88) == 0) {
        final p = _b.board[t];
        if (p != null) {
          if (p.color == byColor &&
              (p.type == ch.Chess.ROOK || p.type == ch.Chess.QUEEN)) {
            return true;
          }
          break;
        }
        t += d;
      }
    }
    for (final d in const [-15, 15, -17, 17]) {
      var t = sq + d;
      while ((t & 0x88) == 0) {
        final p = _b.board[t];
        if (p != null) {
          if (p.color == byColor &&
              (p.type == ch.Chess.BISHOP || p.type == ch.Chess.QUEEN)) {
            return true;
          }
          break;
        }
        t += d;
      }
    }
    return false;
  }

  /// Negamax on a complexity budget. Quiet moves cost 1, forcing moves cost 0
  /// and extend for free. At budget <= 0 (and not in check) the node is a
  /// quiescence leaf: stand-pat, then search only forcing moves.
  int _negamaxC(int budget, int alpha, int beta, int ply) {
    if ((_nodes & 2047) == 0 && _timeUp()) {
      _aborted = true;
      return 0;
    }
    _nodes++;
    if (ply >= _plyCap) return _evalStm();

    final key = _zobrist();
    final tte = tt[key];
    ch.Move? ttMove;
    if (tte != null) {
      if (tte.depth >= budget) {
        if (tte.flag == 0) return tte.score;
        if (tte.flag == 1 && tte.score >= beta) return tte.score;
        if (tte.flag == 2 && tte.score <= alpha) return tte.score;
      }
      ttMove = _moveFromSquares(tte.from, tte.to);
    }

    final inCheck = _b.king_attacked(_b.turn);
    final moves = _b.generate_moves();
    if (moves.isEmpty) {
      return inCheck ? -(mateScore - ply) : 0;
    }

    final atLeaf = budget <= 0 && !inCheck;
    final origAlpha = alpha;
    var best = -_inf;
    if (atLeaf) {
      final standPat = _evalStm();
      if (standPat >= beta) return beta;
      best = standPat;
      if (standPat > alpha) alpha = standPat;
    }

    _order(moves, ttMove, null, key);
    ch.Move? bestMove;
    const tacticalMask = ch.Chess.BITS_CAPTURE |
        ch.Chess.BITS_EP_CAPTURE |
        ch.Chess.BITS_PROMOTION |
        ch.Chess.BITS_KSIDE_CASTLE |
        ch.Chess.BITS_QSIDE_CASTLE;

    // Singular detection: if the TT move is much better than every alternative
    // (an exclusion search of the others fails low), it is effectively forced,
    // so it costs 0 (extends for free) like a reply to check.
    ch.Move? singular;
    if (ttMove != null &&
        tte != null &&
        budget >= 3 &&
        tte.depth >= budget - 1 &&
        tte.flag != 2 &&
        tte.score.abs() < mateScore - 1000) {
      final target = tte.score - (40 + budget * 8);
      final singBudget = (budget - 1) ~/ 2;
      var bestExcl = -_inf;
      for (final m in moves) {
        if (m.from == ttMove.from && m.to == ttMove.to) continue;
        _makeMove(m);
        final forcing = inCheck || _isForcing(m, _b.king_attacked(_b.turn));
        final cost = forcing ? 0 : (1 + ply ~/ _quietStep);
        final s =
            -_negamaxC(singBudget - cost, -target, -(target - 1), ply + 1);
        _undoMove();
        if (_aborted) return 0;
        if (s > bestExcl) bestExcl = s;
        if (bestExcl >= target) break;
      }
      if (bestExcl < target) singular = ttMove; // only the TT move holds
    }

    const captureMask = ch.Chess.BITS_CAPTURE | ch.Chess.BITS_EP_CAPTURE;
    const freeMask = ch.Chess.BITS_PROMOTION |
        ch.Chess.BITS_KSIDE_CASTLE |
        ch.Chess.BITS_QSIDE_CASTLE;
    for (final m in moves) {
      if (atLeaf && (m.flags & tacticalMask) == 0) continue;
      final losing = (m.flags & captureMask) != 0 && _losingCapture(m);
      if (atLeaf && losing) continue; // quiescence skips losing captures
      _makeMove(m);
      final givesCheck = _b.king_attacked(_b.turn);
      // Forcing (cost 0): checks, replies to check, promotions/castling, and
      // winning/equal captures. Losing captures cost budget so they don't
      // extend for free and blow up the tree.
      var forcing = inCheck ||
          givesCheck ||
          (m.flags & freeMask) != 0 ||
          ((m.flags & captureMask) != 0 && !losing);
      if (singular != null &&
          m.from == singular.from &&
          m.to == singular.to) {
        forcing = true; // singular (forced) move extends for free
      }
      final cost = forcing ? 0 : (1 + ply ~/ _quietStep);
      final score = -_negamaxC(budget - cost, -beta, -alpha, ply + 1);
      _undoMove();
      if (_aborted) return 0;
      if (score > best) {
        best = score;
        bestMove = m;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }

    final flag = best <= origAlpha ? 2 : (best >= beta ? 1 : 0);
    _store(key, budget, best, flag, bestMove);
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

  /// Whether [c] has any non-pawn, non-king material (knight/bishop/rook/queen),
  /// used to gate null move pruning against zugzwang in pawn endings.
  bool _hasNonPawnMaterial(ch.Color c) {
    for (var i = ch.Chess.SQUARES_A8; i <= ch.Chess.SQUARES_H1; i++) {
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }
      final p = _b.board[i];
      if (p == null || p.color != c) continue;
      final t = p.type.shift;
      if (t >= 1 && t <= 4) return true; // knight..queen
    }
    return false;
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
      _makeMove(m);
      final score = -_quiesce(-beta, -alpha, ply + 1);
      _undoMove();
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  /// Evaluation from side-to-move POV. Uses NNUE when available, else a
  /// material + piece-square-table estimate.
  int _evalStm() {
    if (nnue != null) {
      return _acc!.eval(_b.turn == ch.Chess.WHITE) ~/ _nnueScale;
    }
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
      [Map<String, double>? priors, int? key]) {
    priors ??= (key != null && nodePriors.isNotEmpty) ? nodePriors[key] : null;
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

  /// Populates interior move-ordering policy from MCTS `{fen: {uci: prior}}`,
  /// re-keyed to this searcher's own zobrist. Clobbers the working board (the
  /// next search reloads it).
  void setNodePriors(Map<String, Map<String, double>> fenPriors) {
    nodePriors.clear();
    for (final e in fenPriors.entries) {
      _b.load(e.key, check_validity: false);
      nodePriors[_zobrist()] = e.value;
    }
  }

  /// Probes the net policy on the first [_pvProbeLen] PV positions and caches it
  /// by zobrist key, so subsequent iterations order those interior nodes by net
  /// policy. No-op when [policyProbe] is unset.
  void _probePv() {
    final probe = policyProbe;
    if (probe == null) return;
    final made = <ch.Move>[];
    var count = 0;
    while (count < _pvProbeLen) {
      final k = _zobrist();
      if (!nodePriors.containsKey(k)) {
        final pr = probe(_b.generate_fen());
        if (pr != null) nodePriors[k] = pr;
      }
      final tte = tt[k];
      if (tte == null || tte.from < 0) break;
      final m = _moveFromSquares(tte.from, tte.to);
      if (m == null) break;
      _b.make_move(m);
      made.add(m);
      count++;
    }
    for (var i = made.length - 1; i >= 0; i--) {
      _b.undo_move();
    }
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

  /// (pv length in plies, quiet-move count) of the TT principal variation.
  (int, int) _pvInfo(int maxLen) {
    var len = 0, quiet = 0;
    final made = <ch.Move>[];
    while (len < maxLen) {
      final tte = tt[_zobrist()];
      if (tte == null || tte.from < 0) break;
      final m = _moveFromSquares(tte.from, tte.to);
      if (m == null) break;
      final inCheck = _b.king_attacked(_b.turn);
      _b.make_move(m);
      if (!(inCheck || _isForcing(m, _b.king_attacked(_b.turn)))) quiet++;
      made.add(m);
      len++;
    }
    for (var i = made.length - 1; i >= 0; i--) {
      _b.undo_move();
    }
    return (len, quiet);
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
