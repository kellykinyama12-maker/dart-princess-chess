/// Pure-Dart loader + CPU forward for DeepMind's "searchless chess" 9M
/// action-value transformer (NeurIPS 2024, "Amortized Planning with Large-Scale
/// Transformers"). Weights come from the HuggingFace mirror
/// `dbest-isi/searchless-chess-9M-selfplay` as a plain `params.npz` (a ZIP of
/// `.npy` arrays), so no Python/JAX is needed — we parse the npz with the
/// `archive` package and run a hand-rolled LLaMa-style decoder on the CPU,
/// mirroring `src/transformer.py` from the reference repo.
library;

import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:chess/chess.dart' as ch;

// ---------------------------------------------------------------------------
// npz / npy parsing
// ---------------------------------------------------------------------------

class _Npy {
  final List<int> shape;
  final Float32List data;
  _Npy(this.shape, this.data);
  int get rows => shape.isEmpty ? 1 : shape[0];
  int get cols => shape.length > 1 ? shape[1] : 1;
}

/// Parses a single little-endian float32 `.npy` payload.
_Npy _parseNpy(Uint8List bytes) {
  // Magic: \x93NUMPY, version (2 bytes), then header length.
  if (bytes.length < 10 || bytes[0] != 0x93) {
    throw FormatException('not a .npy file');
  }
  final major = bytes[6];
  int headerLen;
  int dataStart;
  if (major == 1) {
    headerLen = bytes[8] | (bytes[9] << 8);
    dataStart = 10 + headerLen;
  } else {
    headerLen = bytes[8] |
        (bytes[9] << 8) |
        (bytes[10] << 16) |
        (bytes[11] << 24);
    dataStart = 12 + headerLen;
  }
  final header = String.fromCharCodes(
      bytes.sublist(major == 1 ? 10 : 12, dataStart));
  if (!header.contains("'<f4'")) {
    throw FormatException('expected <f4 dtype, got: $header');
  }
  if (header.contains("'fortran_order': True")) {
    throw FormatException('fortran-order arrays not supported');
  }
  final shapeMatch = RegExp(r"'shape':\s*\(([^)]*)\)").firstMatch(header);
  final shape = <int>[];
  for (final part in shapeMatch!.group(1)!.split(',')) {
    final t = part.trim();
    if (t.isNotEmpty) shape.add(int.parse(t));
  }
  // Reinterpret the raw bytes as little-endian float32.
  final bd = ByteData.sublistView(bytes, dataStart);
  var n = 1;
  for (final s in shape) {
    n *= s;
  }
  final out = Float32List(n);
  for (var i = 0; i < n; i++) {
    out[i] = bd.getFloat32(i * 4, Endian.little);
  }
  return _Npy(shape, out);
}

/// Loads `params.npz` and returns its `arr_0`.. arrays in index order.
List<_Npy> _loadNpz(String path) {
  final archive = ZipDecoder().decodeBytes(File(path).readAsBytesSync());
  final byIndex = <int, _Npy>{};
  for (final f in archive.files) {
    if (!f.isFile) continue;
    final m = RegExp(r'arr_(\d+)\.npy').firstMatch(f.name);
    if (m == null) continue;
    byIndex[int.parse(m.group(1)!)] = _parseNpy(f.content as Uint8List);
  }
  final out = <_Npy>[];
  for (var i = 0; i < byIndex.length; i++) {
    final a = byIndex[i];
    if (a == null) throw StateError('missing arr_$i in $path');
    out.add(a);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Tokenizer (mirrors src/tokenizer.py)
// ---------------------------------------------------------------------------

const List<String> _characters = [
  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', //
  'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', //
  'p', 'n', 'r', 'k', 'q', //
  'P', 'B', 'N', 'R', 'Q', 'K', //
  'w', '.', //
];

final Map<String, int> _charIndex = {
  for (var i = 0; i < _characters.length; i++) _characters[i]: i
};

const int seqLength = 77;
final int _dot = _charIndex['.']!;

/// Tokenizes a FEN into exactly [seqLength] integer tokens.
Int32List tokenizeFen(String fen) {
  final parts = fen.split(' ');
  final boardPart = parts[0].replaceAll('/', '');
  final side = parts[1];
  final castling = parts[2];
  final enPassant = parts[3];
  var halfmoves = parts.length > 4 ? parts[4] : '0';
  var fullmoves = parts.length > 5 ? parts[5] : '1';

  final indices = <int>[];
  for (final ch in (side + boardPart).split('')) {
    final code = ch.codeUnitAt(0);
    if (code >= 0x31 && code <= 0x38) {
      // '1'..'8' => that many empty squares.
      indices.addAll(List.filled(int.parse(ch), _dot));
    } else {
      indices.add(_charIndex[ch]!);
    }
  }
  if (castling == '-') {
    indices.addAll(List.filled(4, _dot));
  } else {
    for (final ch in castling.split('')) {
      indices.add(_charIndex[ch]!);
    }
    for (var i = castling.length; i < 4; i++) {
      indices.add(_dot);
    }
  }
  if (enPassant == '-') {
    indices.addAll([_dot, _dot]);
  } else {
    for (final ch in enPassant.split('')) {
      indices.add(_charIndex[ch]!);
    }
  }
  halfmoves = halfmoves + '.' * (3 - halfmoves.length);
  for (final ch in halfmoves.split('')) {
    indices.add(_charIndex[ch]!);
  }
  fullmoves = fullmoves + '.' * (3 - fullmoves.length);
  for (final ch in fullmoves.split('')) {
    indices.add(_charIndex[ch]!);
  }
  if (indices.length != seqLength) {
    throw StateError('tokenized length ${indices.length} != $seqLength: $fen');
  }
  return Int32List.fromList(indices);
}

// ---------------------------------------------------------------------------
// Move -> action index (mirrors utils._compute_all_possible_actions)
// ---------------------------------------------------------------------------

String _squareName(int sq) {
  final file = sq & 7;
  final rank = sq >> 3;
  return '${String.fromCharCode(0x61 + file)}${rank + 1}';
}

List<int> _queenAttacks(int sq) {
  final file = sq & 7, rank = sq >> 3;
  final out = <int>[];
  const dirs = [
    [1, 0], [-1, 0], [0, 1], [0, -1], //
    [1, 1], [1, -1], [-1, 1], [-1, -1], //
  ];
  for (final d in dirs) {
    var f = file + d[0], r = rank + d[1];
    while (f >= 0 && f < 8 && r >= 0 && r < 8) {
      out.add(r * 8 + f);
      f += d[0];
      r += d[1];
    }
  }
  out.sort();
  return out;
}

List<int> _knightAttacks(int sq) {
  final file = sq & 7, rank = sq >> 3;
  final out = <int>[];
  const offs = [
    [1, 2], [2, 1], [2, -1], [1, -2], //
    [-1, -2], [-2, -1], [-2, 1], [-1, 2], //
  ];
  for (final o in offs) {
    final f = file + o[0], r = rank + o[1];
    if (f >= 0 && f < 8 && r >= 0 && r < 8) out.add(r * 8 + f);
  }
  out.sort();
  return out;
}

Map<String, int> _buildMoveToAction() {
  final allMoves = <String>[];
  for (var sq = 0; sq < 64; sq++) {
    final from = _squareName(sq);
    for (final to in _queenAttacks(sq)) {
      allMoves.add(from + _squareName(to));
    }
    for (final to in _knightAttacks(sq)) {
      allMoves.add(from + _squareName(to));
    }
  }
  const files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
  const pieces = ['q', 'r', 'b', 'n'];
  for (final ranks in [
    ['2', '1'],
    ['7', '8']
  ]) {
    final rank = ranks[0], nextRank = ranks[1];
    for (var i = 0; i < files.length; i++) {
      final file = files[i];
      for (final p in pieces) {
        allMoves.add('$file$rank$file$nextRank$p');
      }
      if (file.compareTo('a') > 0) {
        final nf = files[i - 1];
        for (final p in pieces) {
          allMoves.add('$file$rank$nf$nextRank$p');
        }
      }
      if (file.compareTo('h') < 0) {
        final nf = files[i + 1];
        for (final p in pieces) {
          allMoves.add('$file$rank$nf$nextRank$p');
        }
      }
    }
  }
  final map = <String, int>{};
  for (var i = 0; i < allMoves.length; i++) {
    map[allMoves[i]] = i;
  }
  return map;
}

// ---------------------------------------------------------------------------
// Transformer weights + forward
// ---------------------------------------------------------------------------

/// SIMD `out[T,N] += X[T,K] @ W[K,N]` (W row-major, out pre-zeroed).
void _matmulAcc(Float32List x, int t, int k, Float32List w, int n,
    Float32List out) {
  final wv = Float32x4List.view(w.buffer, w.offsetInBytes, w.length >> 2);
  final ov = Float32x4List.view(out.buffer, out.offsetInBytes, out.length >> 2);
  final n4 = n >> 2;
  for (var ti = 0; ti < t; ti++) {
    final xoff = ti * k;
    final oRow = ti * n4;
    for (var ki = 0; ki < k; ki++) {
      final xtk = x[xoff + ki];
      if (xtk == 0.0) continue;
      final xs = Float32x4.splat(xtk);
      final wRow = ki * n4;
      for (var j = 0; j < n4; j++) {
        ov[oRow + j] += xs * wv[wRow + j];
      }
    }
  }
}

class SearchlessNet {
  static const int buckets = 128;
  static const int maxSeq = 79;
  static const int vocab = 1968;
  static const double _eps = 1e-5;

  final int dim;
  final int numLayers;
  final int numHeads;
  final int headDim;
  final int ffn;
  final double _embScale;

  final Float32List tokEmb; // [vocab, dim]
  final Float32List posEmb; // [maxSeq, dim]
  final List<Float32List> lnOffAttn, lnScaleAttn; // per layer
  final List<Float32List> lnOffMlp, lnScaleMlp; // per layer
  final Float32List lnOffFinal, lnScaleFinal;
  final List<Float32List> wq, wk, wv, wo; // [dim,dim] each
  final List<Float32List> w1, w2, w3; // [dim,ffn],[dim,ffn],[ffn,dim]
  final Float32List wLogits; // [dim,buckets]
  final Float32List bLogits; // [buckets]

  SearchlessNet._({
    required this.dim,
    required this.numLayers,
    required this.numHeads,
    required this.headDim,
    required this.ffn,
    required this.tokEmb,
    required this.posEmb,
    required this.lnOffAttn,
    required this.lnScaleAttn,
    required this.lnOffMlp,
    required this.lnScaleMlp,
    required this.lnOffFinal,
    required this.lnScaleFinal,
    required this.wq,
    required this.wk,
    required this.wv,
    required this.wo,
    required this.w1,
    required this.w2,
    required this.w3,
    required this.wLogits,
    required this.bLogits,
  }) : _embScale = math.sqrt(dim.toDouble());

  /// Loads from a `params.npz`, resolving the jax sorted-key flatten order.
  /// Model dimensions are auto-detected: `dim` from the token embedding and
  /// `numLayers` from the array count (all released nets use 8 heads).
  factory SearchlessNet.load(String npzPath, {int numHeads = 8}) {
    final arr = _loadNpz(npzPath);
    final dim = arr[0].cols; // arr_0 = token embedding [vocab, dim]
    final numLayers = (arr.length - 6) ~/ 11;
    final ffn = dim * 4; // widening_factor = 4
    final headDim = dim ~/ numHeads;

    // Rebuild the (module_key, param_name) -> arr_index mapping by replicating
    // jax.tree_util's deterministic sort of the Haiku params dict: lexicographic
    // by module key string, then by parameter name string.
    final entries = <List<String>>[]; // [moduleKey, paramName]
    void add(String key, String param) => entries.add([key, param]);
    add('embed', 'embeddings'); // token embedding
    add('embed_1', 'embeddings'); // learned positional embedding
    for (var i = 0; i < 2 * numLayers + 1; i++) {
      final key = i == 0 ? 'layer_norm' : 'layer_norm_$i';
      add(key, 'offset');
      add(key, 'scale');
    }
    final finalLin = 3 * numLayers; // final logits linear index
    for (var i = 0; i <= finalLin; i++) {
      final key = i == 0 ? 'linear' : 'linear_$i';
      if (i == finalLin) add(key, 'b'); // final logits linear has bias
      add(key, 'w');
    }
    for (var m = 0; m < numLayers; m++) {
      final mod = m == 0
          ? 'multi_head_dot_product_attention'
          : 'multi_head_dot_product_attention_$m';
      for (var s = 0; s < 4; s++) {
        add('$mod/linear${s == 0 ? '' : '_$s'}', 'w');
      }
    }
    entries.sort((a, b) {
      final c = a[0].compareTo(b[0]);
      return c != 0 ? c : a[1].compareTo(b[1]);
    });
    final index = <String, int>{};
    for (var i = 0; i < entries.length; i++) {
      index['${entries[i][0]}/${entries[i][1]}'] = i;
    }
    Float32List w(String key, String param) => arr[index['$key/$param']!].data;

    List<Float32List> lnOffA = [], lnScaleA = [], lnOffM = [], lnScaleM = [];
    for (var layer = 0; layer < numLayers; layer++) {
      final attnLn = layer * 2; // call order: attn-preLN then mlp-preLN
      final mlpLn = layer * 2 + 1;
      lnOffA.add(w(_lnKey(attnLn), 'offset'));
      lnScaleA.add(w(_lnKey(attnLn), 'scale'));
      lnOffM.add(w(_lnKey(mlpLn), 'offset'));
      lnScaleM.add(w(_lnKey(mlpLn), 'scale'));
    }

    final wqs = <Float32List>[], wks = <Float32List>[];
    final wvs = <Float32List>[], wos = <Float32List>[];
    for (var m = 0; m < numLayers; m++) {
      final mod = m == 0
          ? 'multi_head_dot_product_attention'
          : 'multi_head_dot_product_attention_$m';
      wqs.add(w('$mod/linear', 'w'));
      wks.add(w('$mod/linear_1', 'w'));
      wvs.add(w('$mod/linear_2', 'w'));
      wos.add(w('$mod/linear_3', 'w'));
    }

    final w1s = <Float32List>[], w2s = <Float32List>[], w3s = <Float32List>[];
    for (var layer = 0; layer < numLayers; layer++) {
      w1s.add(w(_linKey(layer * 3), 'w'));
      w2s.add(w(_linKey(layer * 3 + 1), 'w'));
      w3s.add(w(_linKey(layer * 3 + 2), 'w'));
    }

    return SearchlessNet._(
      dim: dim,
      numLayers: numLayers,
      numHeads: numHeads,
      headDim: headDim,
      ffn: ffn,
      tokEmb: w('embed', 'embeddings'),
      posEmb: w('embed_1', 'embeddings'),
      lnOffAttn: lnOffA,
      lnScaleAttn: lnScaleA,
      lnOffMlp: lnOffM,
      lnScaleMlp: lnScaleM,
      lnOffFinal: w(_lnKey(2 * numLayers), 'offset'),
      lnScaleFinal: w(_lnKey(2 * numLayers), 'scale'),
      wq: wqs,
      wk: wks,
      wv: wvs,
      wo: wos,
      w1: w1s,
      w2: w2s,
      w3: w3s,
      wLogits: w(_linKey(finalLin), 'w'),
      bLogits: w(_linKey(finalLin), 'b'),
    );
  }

  static String _lnKey(int i) => i == 0 ? 'layer_norm' : 'layer_norm_$i';
  static String _linKey(int i) => i == 0 ? 'linear' : 'linear_$i';

  void _layerNorm(Float32List h, int t, Float32List scale, Float32List offset) {
    for (var ti = 0; ti < t; ti++) {
      final off = ti * dim;
      var mean = 0.0;
      for (var i = 0; i < dim; i++) {
        mean += h[off + i];
      }
      mean /= dim;
      var varc = 0.0;
      for (var i = 0; i < dim; i++) {
        final d = h[off + i] - mean;
        varc += d * d;
      }
      varc /= dim;
      final inv = 1.0 / math.sqrt(varc + _eps);
      for (var i = 0; i < dim; i++) {
        h[off + i] = (h[off + i] - mean) * inv * scale[i] + offset[i];
      }
    }
  }

  void _attention(Float32List x, int t, int layer, Float32List out) {
    final q = Float32List(t * dim);
    final k = Float32List(t * dim);
    final v = Float32List(t * dim);
    _matmulAcc(x, t, dim, wq[layer], dim, q);
    _matmulAcc(x, t, dim, wk[layer], dim, k);
    _matmulAcc(x, t, dim, wv[layer], dim, v);
    final scale = 1.0 / math.sqrt(headDim.toDouble());
    final ctx = Float32List(t * dim); // concat heads
    final scores = Float32List(t);
    for (var head = 0; head < numHeads; head++) {
      final hoff = head * headDim;
      for (var i = 0; i < t; i++) {
        final qoff = i * dim + hoff;
        var maxS = double.negativeInfinity;
        for (var j = 0; j < t; j++) {
          final koff = j * dim + hoff;
          var dot = 0.0;
          for (var d = 0; d < headDim; d++) {
            dot += q[qoff + d] * k[koff + d];
          }
          dot *= scale;
          scores[j] = dot;
          if (dot > maxS) maxS = dot;
        }
        var sum = 0.0;
        for (var j = 0; j < t; j++) {
          final e = math.exp(scores[j] - maxS);
          scores[j] = e;
          sum += e;
        }
        final inv = 1.0 / sum;
        final coff = i * dim + hoff;
        for (var j = 0; j < t; j++) {
          final p = scores[j] * inv;
          final voff = j * dim + hoff;
          for (var d = 0; d < headDim; d++) {
            ctx[coff + d] += p * v[voff + d];
          }
        }
      }
    }
    _matmulAcc(ctx, t, dim, wo[layer], dim, out);
  }

  void _mlp(Float32List x, int t, int layer, Float32List out) {
    final s1 = Float32List(t * ffn);
    final s2 = Float32List(t * ffn);
    _matmulAcc(x, t, dim, w1[layer], ffn, s1);
    _matmulAcc(x, t, dim, w2[layer], ffn, s2);
    // SwiGLU: silu(s1) * s2
    for (var i = 0; i < s1.length; i++) {
      final a = s1[i];
      s1[i] = (a / (1.0 + math.exp(-a))) * s2[i];
    }
    _matmulAcc(s1, t, ffn, w3[layer], dim, out);
  }

  /// Runs the model on a token sequence (length [maxSeq]) and returns the 128
  /// bucket logits at the final position. [tokens] already includes the BOS
  /// shift (i.e. `[0, fen.., action]`).
  Float32List forwardLast(Int32List tokens) {
    final t = tokens.length;
    final h = Float32List(t * dim);
    for (var i = 0; i < t; i++) {
      final e = tokens[i] * dim;
      final p = i * dim;
      final off = i * dim;
      for (var d = 0; d < dim; d++) {
        h[off + d] = tokEmb[e + d] * _embScale + posEmb[p + d];
      }
    }
    final tmp = Float32List(t * dim);
    for (var layer = 0; layer < numLayers; layer++) {
      // Pre-LN attention.
      final normed = Float32List.fromList(h);
      _layerNorm(normed, t, lnScaleAttn[layer], lnOffAttn[layer]);
      for (var i = 0; i < tmp.length; i++) {
        tmp[i] = 0.0;
      }
      _attention(normed, t, layer, tmp);
      for (var i = 0; i < h.length; i++) {
        h[i] += tmp[i];
      }
      // Pre-LN MLP.
      final normed2 = Float32List.fromList(h);
      _layerNorm(normed2, t, lnScaleMlp[layer], lnOffMlp[layer]);
      for (var i = 0; i < tmp.length; i++) {
        tmp[i] = 0.0;
      }
      _mlp(normed2, t, layer, tmp);
      for (var i = 0; i < h.length; i++) {
        h[i] += tmp[i];
      }
    }
    _layerNorm(h, t, lnScaleFinal, lnOffFinal);
    // Final logits at the last position only.
    final last = Float32List(dim);
    final loff = (t - 1) * dim;
    for (var d = 0; d < dim; d++) {
      last[d] = h[loff + d];
    }
    final logits = Float32List(buckets);
    for (var b = 0; b < buckets; b++) {
      logits[b] = bLogits[b];
    }
    _matmulAcc(last, 1, dim, wLogits, buckets, logits);
    return logits;
  }
}

// ---------------------------------------------------------------------------
// Action-value engine
// ---------------------------------------------------------------------------

class SearchlessEngine {
  final SearchlessNet net;
  final Map<String, int> moveToAction;
  final Float32List bucketValues; // uniform bucket centers (i+0.5)/128

  SearchlessEngine._(this.net, this.moveToAction, this.bucketValues);

  factory SearchlessEngine.load(String npzPath) {
    final values = Float32List(SearchlessNet.buckets);
    for (var i = 0; i < values.length; i++) {
      values[i] = (i + 0.5) / SearchlessNet.buckets;
    }
    return SearchlessEngine._(
        SearchlessNet.load(npzPath), _buildMoveToAction(), values);
  }

  String _uci(ch.Move m) =>
      m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');

  double _scoreMove(Int32List fenTokens, int action) {
    // Shifted input: [BOS=0, fen(77), action]  (length 79).
    final seq = Int32List(SearchlessNet.maxSeq);
    seq[0] = 0;
    for (var i = 0; i < seqLength; i++) {
      seq[i + 1] = fenTokens[i];
    }
    seq[seqLength + 1] = action;
    final logits = net.forwardLast(seq);
    // Softmax over buckets, then expected bucket value = win probability.
    var maxL = double.negativeInfinity;
    for (final l in logits) {
      if (l > maxL) maxL = l;
    }
    var sum = 0.0;
    for (var b = 0; b < logits.length; b++) {
      final e = math.exp(logits[b] - maxL);
      logits[b] = e;
      sum += e;
    }
    var win = 0.0;
    for (var b = 0; b < logits.length; b++) {
      win += (logits[b] / sum) * bucketValues[b];
    }
    return win;
  }

  /// Returns the win probability (side-to-move) for each legal move, by UCI.
  Map<String, double> evaluate(String fen) =>
      evaluateRange(fen, 0, 1 << 30);

  /// Scores only legal moves with index in `[lo, hi)` (stable move order), for
  /// splitting work across isolates.
  Map<String, double> evaluateRange(String fen, int lo, int hi) {
    final board = ch.Chess.fromFEN(fen);
    final moves =
        (board.moves({'asObjects': true}) as List).cast<ch.Move>();
    final fenTokens = tokenizeFen(fen);
    final result = <String, double>{};
    final end = hi < moves.length ? hi : moves.length;
    for (var i = lo; i < end; i++) {
      final uci = _uci(moves[i]);
      final action = moveToAction[uci];
      if (action == null) continue;
      result[uci] = _scoreMove(fenTokens, action);
    }
    return result;
  }

  /// Returns the best move (highest win probability) in UCI, or null if none.
  String? bestMove(String fen) {
    final scores = evaluate(fen);
    String? best;
    var bestWin = double.negativeInfinity;
    scores.forEach((uci, win) {
      if (win > bestWin) {
        bestWin = win;
        best = uci;
      }
    });
    return best;
  }
}
