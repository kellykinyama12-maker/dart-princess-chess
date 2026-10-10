// Measures conv-net (Lc0Net) throughput: single forward vs batched forward,
// to decide whether minibatching leaves in the MCTS eval pool pays off.
import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

void main() {
  const path = 'models/lc0/744706.pb.gz';
  if (!File(path).existsSync()) {
    stderr.writeln('missing $path');
    exit(64);
  }
  final w = Lc0Reader.readFile(path);
  final net = Lc0Net(w, device: Device.CPU);

  const fens = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2',
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
  ];
  Tensor mk(int i) => Lc0Input.fromFen(fens[i % fens.length]);

  // Warmup (JIT).
  for (var i = 0; i < 4; i++) {
    net(mk(i));
  }
  net.callBatch([for (var i = 0; i < 4; i++) mk(i)]);

  for (final n in [1, 4, 8, 16, 32]) {
    // single: n separate forwards
    final sw1 = Stopwatch()..start();
    for (var i = 0; i < n; i++) {
      net(mk(i));
    }
    sw1.stop();
    // batched: one callBatch of n
    final batch = [for (var i = 0; i < n; i++) mk(i)];
    final sw2 = Stopwatch()..start();
    net.callBatch(batch);
    sw2.stop();
    final ps = sw1.elapsedMicroseconds / n / 1000;
    final pb = sw2.elapsedMicroseconds / n / 1000;
    print('N=${n.toString().padLeft(2)}  '
        'single ${ps.toStringAsFixed(1)} ms/pos   '
        'batch ${pb.toStringAsFixed(1)} ms/pos   '
        'speedup ${(ps / pb).toStringAsFixed(2)}x');
  }
}
