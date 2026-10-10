import 'dart:io';
import 'dart:typed_data';

import 'package:chess/chess.dart' as ch;

/// Stockfish HalfKP `256x2-32-32-1` NNUE evaluator (the Stockfish-12-era
/// architecture, e.g. `nn-62ef826d1a6d.nnue`). Parses the quantized `.nnue`
/// file and evaluates a position with a full accumulator recompute.
///
/// Layout: int16 feature-transformer (41024 features -> 256 per perspective),
/// then int8 affine layers 512 -> 32 -> 32 -> 1 with clipped-ReLU and a
/// 6-bit weight scale. Output is side-to-move centipawns (value / 16).
class Nnue {
  static const int _kHalfDims = 256;
  static const int _psEnd = 641; // 10*64 + 1
  static const int _ftInDims = 64 * _psEnd; // 41024
  static const int _transformed = 512; // 256 * 2 perspectives
  static const int _weightScaleBits = 6;
  static const int _fvScale = 16;

  // Feature-transformer parameters (quantized int16).
  final Int16List _ftBias; // [256]
  final Int16List _ftWeights; // [41024 * 256], row = feature

  // Affine layers as float (int values stay exact in float32) for SIMD; weights
  // row-major [out][in].
  final Float32List _l1Bias; // [32]
  final Float32List _l1Weights; // [32 * 512]
  final Float32List _l2Bias; // [32]
  final Float32List _l2Weights; // [32 * 32]
  final Float32List _l3Bias; // [1]
  final Float32List _l3Weights; // [32]

  Nnue._(this._ftBias, this._ftWeights, this._l1Bias, this._l1Weights,
      this._l2Bias, this._l2Weights, this._l3Bias, this._l3Weights);

  /// Piece-code (1=WK..6=WP, 7=BK..12=BP) base offset into a perspective's
  /// feature block; index [perspective][code]. Kings (1,7) are excluded.
  static const List<List<int>> _pieceToIndex = [
    // WHITE perspective
    [0, 0, 513, 385, 257, 129, 1, 0, 577, 449, 321, 193, 65, 0],
    // BLACK perspective
    [0, 0, 577, 449, 321, 193, 65, 0, 513, 385, 257, 129, 1, 0],
  ];

  /// Loads and parses an `.nnue` HalfKP file; throws if the size/version
  /// doesn't match the expected architecture.
  static Nnue load(String path) {
    final bytes = File(path).readAsBytesSync();
    final bd = ByteData.sublistView(bytes);
    var o = 0;
    final version = bd.getUint32(o, Endian.little);
    o += 4;
    if (version != 0x7AF32F16) {
      throw StateError('not a HalfKP net (version 0x${version.toRadixString(16)})');
    }
    o += 4; // hash
    final descLen = bd.getUint32(o, Endian.little);
    o += 4 + descLen; // skip architecture description

    o += 4; // feature-transformer section hash
    final ftBias = Int16List(_kHalfDims);
    for (var i = 0; i < _kHalfDims; i++) {
      ftBias[i] = bd.getInt16(o, Endian.little);
      o += 2;
    }
    final ftWeights = Int16List(_ftInDims * _kHalfDims);
    for (var i = 0; i < ftWeights.length; i++) {
      ftWeights[i] = bd.getInt16(o, Endian.little);
      o += 2;
    }

    o += 4; // network section hash
    Float32List readBias(int n) {
      final b = Float32List(n);
      for (var i = 0; i < n; i++) {
        b[i] = bd.getInt32(o, Endian.little).toDouble();
        o += 4;
      }
      return b;
    }

    Float32List readWeights(int n) {
      final w = Float32List(n);
      for (var i = 0; i < n; i++) {
        w[i] = bd.getInt8(o).toDouble();
        o += 1;
      }
      return w;
    }

    final l1Bias = readBias(32);
    final l1Weights = readWeights(32 * 512);
    final l2Bias = readBias(32);
    final l2Weights = readWeights(32 * 32);
    final l3Bias = readBias(1);
    final l3Weights = readWeights(32);
    if (o != bytes.length) {
      throw StateError('NNUE parse ended at $o, expected ${bytes.length}');
    }
    return Nnue._(ftBias, ftWeights, l1Bias, l1Weights, l2Bias, l2Weights,
        l3Bias, l3Weights);
  }

  static int _orient(int perspective, int sq) =>
      perspective == 0 ? sq : sq ^ 63;

  /// Builds a perspective's 256-wide accumulator for the given pieces.
  void _accumulate(Int32List acc, int perspective, int kingSq,
      List<int> pieceSqs, List<int> pieceCodes) {
    for (var i = 0; i < _kHalfDims; i++) {
      acc[i] = _ftBias[i];
    }
    final ko = _orient(perspective, kingSq);
    final base = _psEnd * ko;
    final idxTable = _pieceToIndex[perspective];
    for (var p = 0; p < pieceSqs.length; p++) {
      final code = pieceCodes[p];
      final f = _orient(perspective, pieceSqs[p]) + idxTable[code] + base;
      final wOff = f * _kHalfDims;
      for (var i = 0; i < _kHalfDims; i++) {
        acc[i] += _ftWeights[wOff + i];
      }
    }
  }

  /// Evaluates [board] in centipawns from the side-to-move perspective
  /// (full recompute; searchers use [NnueAccumulator] for incremental updates).
  int evaluate(ch.Chess board) {
    final accW = Int32List(_kHalfDims);
    final accB = Int32List(_kHalfDims);
    refreshAcc(accW, accB, board);
    return _evalFromAcc(accW, accB, board.turn == ch.Chess.WHITE);
  }

  /// Full recompute of both perspective accumulators from [board].
  void refreshAcc(Int32List accW, Int32List accB, ch.Chess board) {
    final sqs = <int>[];
    final codes = <int>[];
    var wKing = 0, bKing = 0;
    for (var i = ch.Chess.SQUARES_A8; i <= ch.Chess.SQUARES_H1; i++) {
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }
      final pc = board.board[i];
      if (pc == null) continue;
      final sq = (i & 7) + 8 * (7 - (i >> 4)); // a1 = 0
      final white = pc.color == ch.Chess.WHITE;
      final code = white ? 6 - pc.type.shift : 12 - pc.type.shift;
      if (pc.type == ch.PieceType.KING) {
        if (white) {
          wKing = sq;
        } else {
          bKing = sq;
        }
      } else {
        sqs.add(sq);
        codes.add(code);
      }
    }
    _accumulate(accW, 0, wKing, sqs, codes);
    _accumulate(accB, 1, bKing, sqs, codes);
  }

  /// King square (a1=0) of [white] on [board].
  int kingSq(ch.Chess board, bool white) {
    final target = white ? ch.Chess.WHITE : ch.Chess.BLACK;
    for (var i = ch.Chess.SQUARES_A8; i <= ch.Chess.SQUARES_H1; i++) {
      if ((i & 0x88) != 0) {
        i += 7;
        continue;
      }
      final pc = board.board[i];
      if (pc != null && pc.type == ch.PieceType.KING && pc.color == target) {
        return (i & 7) + 8 * (7 - (i >> 4));
      }
    }
    return 0;
  }

  /// Incrementally updates both accumulators for a non-king, non-castling move
  /// [m] (kings unchanged at [wKing]/[bKing], a1=0).
  void applyNonKingMove(
      Int32List accW, Int32List accB, int wKing, int bKing, ch.Move m) {
    final white = m.color == ch.Chess.WHITE;
    final movedCode = white ? 6 - m.piece.shift : 12 - m.piece.shift;
    final fromSq = (m.from & 7) + 8 * (7 - (m.from >> 4));
    final toSq = (m.to & 7) + 8 * (7 - (m.to >> 4));
    final finalType = m.promotion ?? m.piece;
    final finalCode = white ? 6 - finalType.shift : 12 - finalType.shift;
    var capSq = -1, capCode = 0;
    final captured = m.captured;
    if (captured != null) {
      capCode = white ? 12 - captured.shift : 6 - captured.shift;
      capSq = (m.flags & ch.Chess.BITS_EP_CAPTURE) != 0
          ? (white ? toSq - 8 : toSq + 8)
          : toSq;
    }
    final wko = wKing; // orient(0, wKing)
    final bko = bKing ^ 63; // orient(1, bKing)
    _delta(accW, 0, fromSq, movedCode, wko, -1);
    _delta(accW, 0, toSq, finalCode, wko, 1);
    if (capSq >= 0) _delta(accW, 0, capSq, capCode, wko, -1);
    _delta(accB, 1, fromSq, movedCode, bko, -1);
    _delta(accB, 1, toSq, finalCode, bko, 1);
    if (capSq >= 0) _delta(accB, 1, capSq, capCode, bko, -1);
  }

  void _delta(Int32List acc, int perspective, int sq, int code, int kingOriented,
      int sign) {
    final f = _orient(perspective, sq) +
        _pieceToIndex[perspective][code] +
        _psEnd * kingOriented;
    final wOff = f * _kHalfDims;
    if (sign > 0) {
      for (var i = 0; i < _kHalfDims; i++) {
        acc[i] += _ftWeights[wOff + i];
      }
    } else {
      for (var i = 0; i < _kHalfDims; i++) {
        acc[i] -= _ftWeights[wOff + i];
      }
    }
  }

  /// Forward pass from prebuilt accumulators -> side-to-move centipawns.
  int _evalFromAcc(Int32List accW, Int32List accB, bool whiteToMove) {
    final own = whiteToMove ? accW : accB;
    final opp = whiteToMove ? accB : accW;
    final input = Float32List(_transformed);
    for (var i = 0; i < _kHalfDims; i++) {
      input[i] = _clamp8(own[i]).toDouble();
      input[_kHalfDims + i] = _clamp8(opp[i]).toDouble();
    }
    final h1 = _affineRelu(input, _l1Bias, _l1Weights, 32, 512);
    final h2 = _affineRelu(h1, _l2Bias, _l2Weights, 32, 32);
    final wv =
        Float32x4List.view(_l3Weights.buffer, _l3Weights.offsetInBytes, 8);
    final xv = Float32x4List.view(h2.buffer, h2.offsetInBytes, 8);
    var acc = Float32x4.zero();
    for (var i = 0; i < 8; i++) {
      acc += wv[i] * xv[i];
    }
    final out = (acc.x + acc.y + acc.z + acc.w + _l3Bias[0]).toInt();
    return out ~/ _fvScale;
  }

  static int _clamp8(int v) => v < 0 ? 0 : (v > 127 ? 127 : v);

  /// Affine transform (SIMD) followed by clipped-ReLU with the 6-bit weight
  /// scale. [weights] is row-major [out][in]; products stay exact in float32.
  Float32List _affineRelu(Float32List input, Float32List bias,
      Float32List weights, int outDim, int inDim) {
    final out = Float32List(outDim);
    final n4 = inDim >> 2;
    final xv = Float32x4List.view(input.buffer, input.offsetInBytes, n4);
    for (var j = 0; j < outDim; j++) {
      final wv = Float32x4List.view(
          weights.buffer, weights.offsetInBytes + j * inDim * 4, n4);
      var acc = Float32x4.zero();
      for (var i = 0; i < n4; i++) {
        acc += wv[i] * xv[i];
      }
      final sum = (acc.x + acc.y + acc.z + acc.w + bias[j]).toInt();
      out[j] = (sum >> _weightScaleBits).clamp(0, 127).toDouble();
    }
    return out;
  }
}

/// Incremental HalfKP accumulator stack for a search: push on make_move, pop on
/// undo_move. King/castling moves trigger a full refresh (the king square
/// reindexes every feature of that perspective); other moves apply piece deltas.
class NnueAccumulator {
  final Nnue net;
  final List<Int32List> _w = [];
  final List<Int32List> _b = [];
  final List<int> _wk = [];
  final List<int> _bk = [];
  int _sp = -1;

  NnueAccumulator(this.net);

  void _ensure(int level) {
    while (_w.length <= level) {
      _w.add(Int32List(256));
      _b.add(Int32List(256));
      _wk.add(0);
      _bk.add(0);
    }
  }

  /// Resets the stack to [board] at level 0.
  void reset(ch.Chess board) {
    _ensure(0);
    _sp = 0;
    net.refreshAcc(_w[0], _b[0], board);
    _wk[0] = net.kingSq(board, true);
    _bk[0] = net.kingSq(board, false);
  }

  /// Pushes the accumulator after making [m] ([boardAfter] is the post-move
  /// position).
  void push(ch.Move m, ch.Chess boardAfter) {
    final sp = _sp;
    _ensure(sp + 1);
    _sp = sp + 1;
    final nw = _w[_sp], nb = _b[_sp];
    final kingMove = m.piece == ch.PieceType.KING ||
        (m.flags &
                (ch.Chess.BITS_KSIDE_CASTLE | ch.Chess.BITS_QSIDE_CASTLE)) !=
            0;
    if (kingMove) {
      net.refreshAcc(nw, nb, boardAfter);
      _wk[_sp] = net.kingSq(boardAfter, true);
      _bk[_sp] = net.kingSq(boardAfter, false);
    } else {
      nw.setAll(0, _w[sp]);
      nb.setAll(0, _b[sp]);
      _wk[_sp] = _wk[sp];
      _bk[_sp] = _bk[sp];
      net.applyNonKingMove(nw, nb, _wk[_sp], _bk[_sp], m);
    }
  }

  void pop() => _sp--;

  int eval(bool whiteToMove) => net._evalFromAcc(_w[_sp], _b[_sp], whiteToMove);
}

