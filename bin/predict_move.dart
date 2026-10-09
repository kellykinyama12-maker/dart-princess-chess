// Predicts the next chess move(s) with the trained autoregressive transformer.
//
// Give it the moves played so far (UCI); it prints the greedy next move plus a
// sampled distribution of likely continuations.
//
// Usage:
//   dart run bin/predict_move.dart e2e4 e7e5 g1f3
//   dart run bin/predict_move.dart            # predicts the opening move
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:dart_pytorch/dart_pytorch.dart';
import 'package:chess_moves/move_vocab.dart';

void main(List<String> args) {
  const dir = 'data';
  final configFile = File('$dir/config.json');
  if (!configFile.existsSync()) {
    stderr.writeln('No trained model in $dir/ (run bin/train_moves.dart first).');
    exit(64);
  }

  final cfg = jsonDecode(configFile.readAsStringSync()) as Map<String, dynamic>;
  final maxCtx = cfg['maxCtx'] as int;
  final vocab = MoveVocab.loadFile('$dir/vocab.json');

  final gpt = GPT(GPTConfig(
    vocabSize: cfg['vocabSize'] as int,
    maxCtx: maxCtx,
    embedDim: cfg['embedDim'] as int,
    numLayers: cfg['numLayers'] as int,
    numHeads: cfg['numHeads'] as int,
    dropoutP: 0.0,
    tieWeights: true,
    seed: 1,
  ));
  Checkpoint.loadIntoFile(gpt, '$dir/model.dptc');
  gpt.eval();

  // Build the prompt: <bos> + moves played so far, truncated to the context.
  final prefix = args.where((a) => a.trim().isNotEmpty).toList();
  final ids = <int>[vocab.bosId, ...prefix.map(vocab.encodeToken)];
  for (final m in prefix) {
    if (vocab.encodeToken(m) == vocab.unkId) {
      stderr.writeln('warning: "$m" is not in the training vocabulary.');
    }
  }
  final maxPrompt = maxCtx - 1;
  final prompt =
      (ids.length > maxPrompt ? ids.sublist(ids.length - maxPrompt) : ids)
          .map((i) => i.toDouble())
          .toList();

  print('moves so far      : ${prefix.isEmpty ? "(start)" : prefix.join(" ")}');

  // ---- Greedy next move ----
  final greedy = gpt.generate(prompt, maxNewTokens: 1, temperature: 0.0, useCache: true);
  final greedyId = greedy[prompt.length].toInt();
  print('greedy next move  : ${_label(vocab, greedyId)}');

  // ---- Sampled distribution of the next move ----
  const samples = 60;
  final tally = <int, int>{};
  for (var s = 0; s < samples; s++) {
    final out = gpt.generate(
      prompt,
      maxNewTokens: 1,
      temperature: 0.9,
      topK: 12,
      rng: math.Random(s),
      useCache: true,
    );
    final id = out[prompt.length].toInt();
    tally[id] = (tally[id] ?? 0) + 1;
  }
  final ranked = tally.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  print('\nlikely next moves (T=0.9, top-k=12, $samples samples):');
  for (final e in ranked.take(6)) {
    final pct = (100 * e.value / samples).toStringAsFixed(0).padLeft(3);
    print('  ${_label(vocab, e.key).padRight(8)} $pct%');
  }
}

String _label(MoveVocab v, int id) {
  final tok = v.decodeId(id);
  if (id == v.eosId) return '<end of game>';
  if (v.isSpecial(id)) return '<$tok>';
  return tok;
}
