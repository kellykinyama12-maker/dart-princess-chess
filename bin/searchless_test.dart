import 'dart:io';

import 'package:chess_moves/searchless.dart';

void main(List<String> args) {
  final npz = args.isNotEmpty ? args[0] : 'models/searchless/params.npz';
  stdout.writeln('Loading $npz ...');
  var sw = Stopwatch()..start();
  final engine = SearchlessEngine.load(npz);
  stdout.writeln('Loaded in ${sw.elapsedMilliseconds} ms');

  const positions = {
    'startpos': 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'after 1.e4': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
    'mate-in-1 (Qh5-f7 style)':
        'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1',
  };

  positions.forEach((name, fen) {
    sw = Stopwatch()..start();
    final scores = engine.evaluate(fen);
    final ms = sw.elapsedMilliseconds;
    final ranked = scores.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    stdout.writeln('\n$name  (${scores.length} moves, ${ms} ms)');
    for (final e in ranked.take(5)) {
      stdout.writeln('  ${e.key}  win=${(e.value * 100).toStringAsFixed(1)}%');
    }
  });
}
