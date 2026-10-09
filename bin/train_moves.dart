// Trains a GPT-style autoregressive transformer (dart_pytorch) over UCI move
// sequences produced by bin/parse_lc0.dart, then checkpoints the model, the
// move vocabulary, and the model config so bin/predict_move.dart can reload it.
//
// Usage:
//   dart run bin/train_moves.dart [--games data/games.txt] [--steps N]
//       [--batch B] [--ctx S] [--embed D] [--layers L] [--heads H]
//       [--max-lr X] [--out-dir data] [--save-every-min M]
//
// The model is checkpointed every --save-every-min minutes (default 60) and at
// the end, so long CPU runs never lose more than that much progress.
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:chess_moves/move_vocab.dart';

void main(List<String> args) {
  var gamesPath = 'data/games.txt';
  var steps = 800;
  var batch = 8;
  var ctx = 48;
  var embed = 128;
  var layers = 3;
  var heads = 4;
  var maxLr = 3e-3;
  var outDir = 'data';
  var saveEveryMin = 60;

  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--games':
        gamesPath = args[++i];
      case '--steps':
        steps = int.parse(args[++i]);
      case '--batch':
        batch = int.parse(args[++i]);
      case '--ctx':
        ctx = int.parse(args[++i]);
      case '--embed':
        embed = int.parse(args[++i]);
      case '--layers':
        layers = int.parse(args[++i]);
      case '--heads':
        heads = int.parse(args[++i]);
      case '--max-lr':
        maxLr = double.parse(args[++i]);
      case '--out-dir':
        outDir = args[++i];
      case '--save-every-min':
        saveEveryMin = int.parse(args[++i]);
    }
  }

  final file = File(gamesPath);
  if (!file.existsSync()) {
    stderr.writeln('games file not found: $gamesPath (run bin/parse_lc0.dart)');
    exit(64);
  }

  // ---- Load games + build vocabulary ----
  final games = <List<String>>[];
  for (final line in file.readAsLinesSync()) {
    if (line.trim().isEmpty) continue;
    games.add(line.split(' '));
  }
  final vocab = MoveVocab.build(games);
  print('games             : ${games.length}');
  print('vocab size        : ${vocab.size} '
      '(${vocab.size - 4} distinct moves + 4 specials)');

  // ---- Encode into one token stream with <bos>/<eos> game boundaries ----
  final stream = <int>[];
  for (final g in games) {
    stream.add(vocab.bosId);
    for (final m in g) {
      stream.add(vocab.encodeToken(m));
    }
    stream.add(vocab.eosId);
  }
  final trainable = stream.length - ctx - 1;
  if (trainable <= 0) {
    stderr.writeln('corpus too short for ctx=$ctx');
    exit(1);
  }
  print('tokens            : ${stream.length}');

  // ---- Model ----
  final gpt = GPT(GPTConfig(
    vocabSize: vocab.size,
    maxCtx: ctx,
    embedDim: embed,
    numLayers: layers,
    numHeads: heads,
    dropoutP: 0.0,
    tieWeights: true,
    seed: 1,
  ));
  final scalars = gpt.parameters().fold<int>(0, (a, p) => a + p.length);
  print('model             : d=$embed, L=$layers, H=$heads, ctx=$ctx '
      '($scalars scalars)');

  // ---- Optimizer + schedule ----
  final warmup = (steps * 0.05).ceil().clamp(1, steps);
  final opt = Adam(gpt.parameters(), lr: 0.0);
  final sched = LinearWarmupCosineDecay(
    opt,
    warmupSteps: warmup,
    totalSteps: steps,
    maxLr: maxLr,
    minLr: maxLr / 10,
  );
  print('optimizer         : Adam + warmup=$warmup/$steps, '
      'maxLr=${maxLr.toStringAsExponential(1)}');
  print('batch             : $batch');
  print('checkpoint        : every $saveEveryMin min + at end\n');

  // ---- Checkpoint sink (called periodically and at the end) ----
  Directory(outDir).createSync(recursive: true);
  final ckptPath = '$outDir/model.dptc';
  final vocabPath = '$outDir/vocab.json';
  final configPath = '$outDir/config.json';
  void saveCheckpoint(String tag, int step) {
    Checkpoint.saveFile(gpt, ckptPath);
    vocab.saveFile(vocabPath);
    File(configPath).writeAsStringSync(jsonEncode({
      'vocabSize': vocab.size,
      'maxCtx': ctx,
      'embedDim': embed,
      'numLayers': layers,
      'numHeads': heads,
    }));
    print('  [checkpoint:$tag] saved $ckptPath '
        '(${File(ckptPath).lengthSync()} bytes) at step $step');
  }

  final saveEvery = Duration(minutes: saveEveryMin);
  final saveTimer = Stopwatch()..start();

  // ---- Training loop (real [B, S] batched forward) ----
  final rng = math.Random(0);
  final xBuf = List<double>.filled(batch * ctx, 0.0);
  final yBuf = List<double>.filled(batch * ctx, 0.0);
  final sw = Stopwatch()..start();
  print('training ...');
  for (var step = 1; step <= steps; step++) {
    opt.zeroGrad();
    for (var b = 0; b < batch; b++) {
      final start = rng.nextInt(trainable);
      for (var t = 0; t < ctx; t++) {
        xBuf[b * ctx + t] = stream[start + t].toDouble();
        yBuf[b * ctx + t] = stream[start + t + 1].toDouble();
      }
    }
    final x = Tensor.fromList([batch, ctx], List<double>.from(xBuf));
    final y = Tensor.fromList([batch, ctx], List<double>.from(yBuf));

    final loss = gpt(x).crossEntropy(y).mean();
    loss.backward();
    final avg = loss.toList()[0];
    clipGradNorm(gpt.parameters(), 1.0);
    opt.step();
    sched.step();

    if (step == 1 || step % 5 == 0 || step == steps) {
      final msPerStep = sw.elapsedMilliseconds / step;
      print('  step ${step.toString().padLeft(4)}  '
          'loss=${avg.toStringAsFixed(4)}  '
          'lr=${opt.lr.toStringAsExponential(2)}  '
          '(${msPerStep.toStringAsFixed(0)} ms/step)');
    }

    if (saveEveryMin > 0 && saveTimer.elapsed >= saveEvery) {
      saveCheckpoint('periodic', step);
      saveTimer.reset();
    }
  }
  sw.stop();
  print('training done in ${sw.elapsedMilliseconds} ms\n');

  saveCheckpoint('final', steps);
  print('saved vocab       : $vocabPath');
  print('saved config      : $configPath');
}
