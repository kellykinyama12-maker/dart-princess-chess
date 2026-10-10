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

  // Affine layers: biases int32, weights int8 (row-major [out][in]).
  final Int32List _l1Bias; // [32]
  final Int8List _l1Weights; // [32 * 512]
  final Int32List _l2Bias; // [32]
  final Int8List _l2Weights; // [32 * 32]
  final Int32List _l3Bias; // [1]
  final Int8List _l3Weights; // [32]

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
    Int32List readBias(int n) {
      final b = Int32List(n);
      for (var i = 0; i < n; i++) {
        b[i] = bd.getInt32(o, Endian.little);
        o += 4;
      }
      return b;
    }

    Int8List readWeights(int n) {
      final w = Int8List(n);
      for (var i = 0; i < n; i++) {
        w[i] = bd.getInt8(o);
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

  /// Evaluates [board] in centipawns from the side-to-move perspective.
  int evaluate(ch.Chess board) {
    // Collect pieces (a1=0 numbering) and king squares.
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

    final accW = Int32List(_kHalfDims);
    final accB = Int32List(_kHalfDims);
    _accumulate(accW, 0, wKing, sqs, codes);
    _accumulate(accB, 1, bKing, sqs, codes);

    // Own perspective first in the transformed input.
    final whiteToMove = board.turn == ch.Chess.WHITE;
    final own = whiteToMove ? accW : accB;
    final opp = whiteToMove ? accB : accW;

    final input = Uint8List(_transformed);
    for (var i = 0; i < _kHalfDims; i++) {
      input[i] = _clamp8(own[i]);
      input[_kHalfDims + i] = _clamp8(opp[i]);
    }

    final h1 = _affineClipped(input, _l1Bias, _l1Weights, 32, 512);
    final h2 = _affineClipped(h1, _l2Bias, _l2Weights, 32, 32);

    var out = _l3Bias[0];
    for (var i = 0; i < 32; i++) {
      out += _l3Weights[i] * h2[i];
    }
    return out ~/ _fvScale;
  }

  static int _clamp8(int v) => v < 0 ? 0 : (v > 127 ? 127 : v);

  /// Affine transform (int8 weights row-major [out][in]) followed by a
  /// clipped-ReLU with the 6-bit weight scale.
  Uint8List _affineClipped(
      Uint8List input, Int32List bias, Int8List weights, int outDim, int inDim) {
    final out = Uint8List(outDim);
    for (var j = 0; j < outDim; j++) {
      var sum = bias[j];
      final wOff = j * inDim;
      for (var i = 0; i < inDim; i++) {
        sum += weights[wOff + i] * input[i];
      }
      out[j] = _clamp8(sum >> _weightScaleBits);
    }
    return out;
  }
}
