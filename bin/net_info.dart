// Prints an lc0 net's architecture + its top policy priors at the start
// position (noise-free), to check conv-policy + WDL evaluator compatibility.
//   dart run bin/net_info.dart models/lc0/744706.pb.gz
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart';

import 'package:chess_moves/lc0_evaluator.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

void main(List<String> args) {
  final path = args.isNotEmpty ? args[0] : 'models/lc0/744706.pb.gz';
  if (!File(path).existsSync()) {
    stderr.writeln('missing $path');
    exit(64);
  }
  final w = Lc0Reader.readFile(path);
  stdout.writeln('net: $path');
  stdout.writeln('filters=${w.filters} blocks=${w.numBlocks} '
      'policyPlanes=${w.policyOutputPlanes} valueFC=${w.valueFCUnits} '
      'wdl=${w.wdl}');
  final seBlocks = w.residual.where((r) => r.se != null).length;
  stdout.writeln('residuals=${w.residual.length} seBlocks=$seBlocks '
      'movesLeft=${w.movesLeftFilters != null}');

  final evaluator = Lc0Evaluator.load(path);
  final board = ch.Chess.fromFEN(_startFen);
  final legal = <String>[];
  for (final mv in board.moves({'verbose': true}) as List) {
    final m = mv as Map;
    legal.add((m['from'] as String) + (m['to'] as String));
  }
  final e = evaluator.evaluate(_startFen, 0, legal);
  stdout.writeln('value WDL=${e.wdl.map((v) => v.toStringAsFixed(3)).toList()}');
  final top = e.priors.entries.toList()
    ..sort((a, b) => b.value.compareTo(a.value));
  stdout.writeln('top priors: ' +
      top
          .take(6)
          .map((x) => '${x.key} ${(x.value * 100).toStringAsFixed(1)}%')
          .join('  '));
}

