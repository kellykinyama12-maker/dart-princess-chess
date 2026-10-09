// Validates the full V6 record decoder (lib/lc0_record.dart): prints the policy,
// value (WDL), moves-left, and played move for the first few positions of one
// game — the targets a future lc0-parity policy/value net will train on.
//
// Usage:
//   dart run bin/inspect_record.dart [path\to.tar] [--positions N]
import 'dart:io';

import 'package:chess_moves/lc0_record.dart';
import 'package:chess_moves/lc0_tar.dart';

void main(List<String> args) {
  var positions = 4;
  String? tar;
  for (var i = 0; i < args.length; i++) {
    if (args[i] == '--positions') {
      positions = int.parse(args[++i]);
    } else {
      tar = args[i];
    }
  }
  tar ??= _firstTar();
  if (tar == null) {
    stderr.writeln('No .tar found. Pass a path or populate training_data/.');
    exit(64);
  }

  final chunks = readGameChunks(tar, maxGames: 1);
  if (chunks.isEmpty) {
    stderr.writeln('No game chunks in $tar');
    exit(1);
  }
  final recs = recordsOf(chunks.first).toList();
  print('tar               : ${_short(tar)}');
  print('positions in game : ${recs.length}');
  print('planes/record     : 104 (8-ply history x 13), policy slots: 1858\n');

  final show = positions < recs.length ? positions : recs.length;
  for (var i = 0; i < show; i++) {
    final r = recs[i];
    final wdl = r.resultWdl();
    final best = r.bestWdl();
    final played = r.playedUci();
    final top = r.topPolicy(5);
    final occupied = _occupiedSquares(r);

    print('position $i  (${r.sideToMove == 0 ? "White" : "Black"} to move, '
        'input_format=${r.inputFormat})');
    print('  played move     : $played');
    print('  result WDL(STM) : W=${_p(wdl[0])} D=${_p(wdl[1])} L=${_p(wdl[2])}');
    print('  search WDL(STM) : W=${_p(best[0])} D=${_p(best[1])} L=${_p(best[2])}');
    print('  plies left      : ${r.pliesLeft.toStringAsFixed(0)}');
    print('  occupied squares: $occupied (plane 0..11 = pieces, current pos)');
    print('  top policy      : '
        '${top.map((e) => "${e.key} ${(e.value * 100).toStringAsFixed(0)}%").join("  ")}');
    print('  played in top-5 : ${top.any((e) => e.key == played)}');
    print('');
  }
}

// Count squares occupied across the 12 piece planes of the current position
// (planes 0..11); a legal start position should show 32.
int _occupiedSquares(Lc0Record r) {
  final seen = <int>{};
  for (var p = 0; p < 12; p++) {
    final plane = r.plane(p);
    for (var s = 0; s < 64; s++) {
      if (plane[s] > 0) seen.add(s);
    }
  }
  return seen.length;
}

String _p(double v) => v.toStringAsFixed(2);

String? _firstTar() {
  final dir = Directory('training_data');
  if (!dir.existsSync()) return null;
  final tars = dir
      .listSync()
      .whereType<File>()
      .where((f) => f.path.toLowerCase().endsWith('.tar'))
      .map((f) => f.path)
      .toList()
    ..sort();
  return tars.isEmpty ? null : tars.first;
}

String _short(String p) => p.replaceAll('\\', '/').split('/').last;
