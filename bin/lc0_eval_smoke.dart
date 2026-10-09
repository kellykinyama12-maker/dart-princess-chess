// Smoke test: can lc0's own network (dart_pytorch Lc0Net) run on the CPU on
// this machine? If yes, we can reuse lc0's real weights as the MCTS evaluator;
// if the conv path (im2colNhwc) is GPU-only, we fall back to training our own
// small policy/value net.
import 'dart:io';

import 'package:dart_pytorch/dart_pytorch.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';
const _weightsUrl =
    'https://storage.lczero.org/files/networks-contrib/744706.pb.gz';

Future<void> main() async {
  await _ensureWeights();

  final w = Lc0Reader.readFile(_weightsPath);
  stdout.writeln('net: filters=${w.filters} blocks=${w.numBlocks} '
      'policyPlanes=${w.policyOutputPlanes} valueFC=${w.valueFCUnits} wdl=${w.wdl}');

  final net = Lc0Net(w, device: Device.CPU);
  final input = Lc0Input.fromFen(startFen);

  final sw = Stopwatch()..start();
  final out = net(input);
  sw.stop();

  final v = out.value.toList();
  stdout.writeln('value WDL (STM): W=${v[0].toStringAsFixed(3)} '
      'D=${v[1].toStringAsFixed(3)} L=${v[2].toStringAsFixed(3)}');
  stdout.writeln('policy shape   : ${out.policyLogits.shape}');
  stdout.writeln('forward time   : ${sw.elapsedMilliseconds} ms (CPU)');
  stdout.writeln('CPU-LC0-OK');
}

Future<void> _ensureWeights() async {
  final file = File(_weightsPath);
  if (await file.exists()) return;
  stderr.writeln('downloading lc0 weights (~6 MB) ...');
  await file.parent.create(recursive: true);
  final client = HttpClient()..userAgent = 'chess_moves/lc0_eval_smoke';
  try {
    final req = await client.getUrl(Uri.parse(_weightsUrl));
    final resp = await req.close();
    if (resp.statusCode != 200) {
      throw HttpException('HTTP ${resp.statusCode} for $_weightsUrl');
    }
    final sink = file.openWrite();
    await resp.pipe(sink);
  } finally {
    client.close(force: true);
  }
}
