// UCI engine wrapping the DeepMind searchless-chess action-value net.
// Pure Dart, no Python. Evaluates legal moves across an isolate pool.
//   dart run bin/searchless_uci.dart
import 'dart:convert';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/searchless_pool.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main(List<String> args) async {
  var npzPath = 'models/searchless/params.npz';
  var workers = (Platform.numberOfProcessors - 1).clamp(1, 15);
  SearchlessPool? pool;
  var board = ch.Chess.fromFEN(_startFen);

  Future<void> ensure() async {
    pool ??= await SearchlessPool.start(npzPath, workers);
  }

  void applyMoves(List<String> ucis) {
    for (final uci in ucis) {
      final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
      for (final m in moves) {
        final mu = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
        if (mu == uci) {
          board.make_move(m);
          break;
        }
      }
    }
  }

  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    final s = line.trim();
    if (s == 'uci') {
      stdout.writeln('id name Searchless9M');
      stdout.writeln('id author chess_moves');
      stdout.writeln('option name WeightsFile type string default $npzPath');
      stdout.writeln('option name Workers type spin default $workers min 1 max 64');
      stdout.writeln('uciok');
    } else if (s == 'isready') {
      await ensure();
      stdout.writeln('readyok');
    } else if (s.startsWith('setoption')) {
      final nameMatch = RegExp(r'name\s+(.+?)\s+value\s+(.+)$').firstMatch(s);
      if (nameMatch != null) {
        final name = nameMatch.group(1)!.trim().toLowerCase();
        final value = nameMatch.group(2)!.trim();
        if (name == 'weightsfile') {
          npzPath = value;
          pool = null;
        } else if (name == 'workers') {
          workers = int.tryParse(value) ?? workers;
          pool = null;
        }
      }
    } else if (s == 'ucinewgame') {
      board = ch.Chess.fromFEN(_startFen);
    } else if (s.startsWith('position')) {
      final movesIdx = s.indexOf(' moves ');
      final head = movesIdx >= 0 ? s.substring(0, movesIdx) : s;
      if (head.contains('startpos')) {
        board = ch.Chess.fromFEN(_startFen);
      } else {
        final fenIdx = head.indexOf('fen ');
        if (fenIdx >= 0) board = ch.Chess.fromFEN(head.substring(fenIdx + 4).trim());
      }
      if (movesIdx >= 0) {
        applyMoves(s.substring(movesIdx + 7).trim().split(RegExp(r'\s+')));
      }
    } else if (s.startsWith('go')) {
      await ensure();
      final best = await pool!.bestMove(board.generate_fen());
      stdout.writeln('bestmove ${best ?? '0000'}');
    } else if (s == 'quit') {
      break;
    }
  }
}
