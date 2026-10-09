import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';

import 'move_index.dart';

/// Network evaluation of a single position.
class Evaluation {
  /// Win/Draw/Loss probabilities from the side-to-move's perspective.
  final List<double> wdl;

  /// Legal move (absolute UCI) -> policy prior probability. Sums to ~1.
  final Map<String, double> priors;

  Evaluation(this.wdl, this.priors);

  /// Scalar value in [-1, 1] from the side-to-move's perspective.
  double get value => wdl[0] - wdl[2];
}

/// Wraps an lc0 network as an MCTS leaf evaluator: returns a value (WDL) and a
/// policy prior over legal moves. Dispatches between classical conv-policy nets
/// ([Lc0Net]) and attention-body nets ([Lc0AttnNet], 1858 policy) by format.
class Lc0Evaluator {
  final Lc0Net? _net; // classical conv/SE-ResNet (conv policy)
  final Lc0AttnNet? _attn; // attention body (1858 policy)

  Lc0Evaluator._(this._net) : _attn = null;
  Lc0Evaluator._attn(this._attn) : _net = null;

  factory Lc0Evaluator.load(String weightsPath, {Device device = Device.CPU}) {
    if (Lc0AttnWeights.isAttentionNet(weightsPath)) {
      return Lc0Evaluator._attn(Lc0AttnNet(Lc0AttnReader.readFile(weightsPath)));
    }
    return Lc0Evaluator._(Lc0Net(Lc0Reader.readFile(weightsPath), device: device));
  }

  /// Evaluates [fen] (with [sideToMove] 0 = White, 1 = Black) and softmaxes the
  /// policy logits over the supplied [legalUci] moves into priors.
  Evaluation evaluate(String fen, int sideToMove, List<String> legalUci) {
    return _attn != null
        ? _evaluateAttn(fen, sideToMove, legalUci)
        : _evaluateConv(fen, sideToMove, legalUci);
  }

  Evaluation _evaluateAttn(String fen, int stm, List<String> legalUci) {
    final out = _attn!.forward(Lc0Input.fromFen(fen).toFloat32List());
    final logits = <String, double>{};
    var maxL = double.negativeInfinity;
    for (final uci in legalUci) {
      final idx = moveIndexForUci(uci, stm);
      final l = idx != null ? out.policy[idx] : -30.0;
      logits[uci] = l;
      if (l > maxL) maxL = l;
    }
    return _softmax(out.wdl, logits, maxL);
  }

  Evaluation _evaluateConv(String fen, int stm, List<String> legalUci) {
    final out = _net!(Lc0Input.fromFen(fen));
    final wdl = out.value.toList().cast<double>();
    final flat = out.policyLogits.toList().cast<double>(); // plane*64 + square
    final logits = <String, double>{};
    var maxL = double.negativeInfinity;
    for (final uci in legalUci) {
      final ci = convIndexForUci(uci, stm);
      // Unmapped moves (e.g. knight underpromotions) get a floor so they stay
      // legal/selectable but carry negligible prior.
      final l = (ci != null && ci < flat.length) ? flat[ci] : -30.0;
      logits[uci] = l;
      if (l > maxL) maxL = l;
    }
    return _softmax(wdl, logits, maxL);
  }

  Evaluation _softmax(
      List<double> wdl, Map<String, double> logits, double maxL) {
    var sum = 0.0;
    final priors = <String, double>{};
    logits.forEach((uci, l) {
      final e = math.exp(l - maxL);
      priors[uci] = e;
      sum += e;
    });
    if (sum > 0) priors.updateAll((_, p) => p / sum);
    return Evaluation(wdl, priors);
  }
}
