// Smoke test: confirm dart_pytorch's CPU path (tensor ops + GPT forward/
// backward + Adam step) runs on this machine before building the full pipeline.
import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  // 1) Plain CPU matmul.
  final a = Tensor.fromList([2, 3], [1, 2, 3, 4, 5, 6]);
  final b = Tensor.fromList([3, 2], [7, 8, 9, 10, 11, 12]);
  print('matmul: ${a.matmul(b).toList()}'); // expect [58, 64, 139, 154]

  // 2) Tiny GPT forward + loss + one optimizer step, all on CPU.
  const vocab = 16;
  const ctx = 8;
  final gpt = GPT(GPTConfig(
    vocabSize: vocab,
    maxCtx: ctx,
    embedDim: 16,
    numLayers: 1,
    numHeads: 2,
    dropoutP: 0.0,
    tieWeights: true,
    seed: 1,
  ));

  final x = Tensor.fromList([1, 4], [1, 2, 3, 4]);
  final y = Tensor.fromList([1, 4], [2, 3, 4, 5]);

  final opt = Adam(gpt.parameters(), lr: 1e-3);
  opt.zeroGrad();
  final loss = gpt(x).crossEntropy(y).mean();
  loss.backward();
  clipGradNorm(gpt.parameters(), 1.0);
  opt.step();

  print('gpt loss: ${loss.toList()[0].toStringAsFixed(4)}');
  print('SMOKE OK');
}
