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

/// Wraps lc0's real network ([Lc0Net]) as an MCTS leaf evaluator: it returns a
/// value (WDL) and a policy prior over the legal moves of a position.
class Lc0Evaluator {
  final Lc0Net _net;

  Lc0Evaluator._(this._net);

  factory Lc0Evaluator.load(String weightsPath, {Device device = Device.CPU}) {
    final w = Lc0Reader.readFile(weightsPath);
    return Lc0Evaluator._(Lc0Net(w, device: device));
  }

  /// Evaluates [fen] (with [sideToMove] 0 = White, 1 = Black) and softmaxes the
  /// conv-policy logits over the supplied [legalUci] moves into priors.
  Evaluation evaluate(String fen, int sideToMove, List<String> legalUci) {
    final out = _net(Lc0Input.fromFen(fen));
    final wdl = out.value.toList().cast<double>();
    final flat = out.policyLogits.toList().cast<double>(); // plane*64 + square

    final logits = <String, double>{};
    var maxL = double.negativeInfinity;
    for (final uci in legalUci) {
      final ci = convIndexForUci(uci, sideToMove);
      // Unmapped moves (e.g. knight underpromotions) get a floor so they stay
      // legal/selectable but carry negligible prior.
      final l = (ci != null && ci < flat.length) ? flat[ci] : -30.0;
      logits[uci] = l;
      if (l > maxL) maxL = l;
    }

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
