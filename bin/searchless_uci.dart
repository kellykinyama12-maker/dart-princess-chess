// UCI engine wrapping the DeepMind searchless-chess 9M action-value net.
// Pure Dart, no Python. Evaluates legal moves across an isolate pool.
//   dart run bin/searchless_uci.dart
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/searchless.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

/// A pool of worker isolates, each holding its own copy of the net, that score
/// disjoint slices of a position's legal moves in parallel.
class SearchlessPool {
  final List<SendPort> _workers;
  SearchlessPool._(this._workers);

  static Future<SearchlessPool> start(String npz, int workers) async {
    final ports = <SendPort>[];
    for (var i = 0; i < workers; i++) {
      final ready = ReceivePort();
      await Isolate.spawn(_worker, [ready.sendPort, npz]);
      ports.add(await ready.first as SendPort);
    }
    return SearchlessPool._(ports);
  }

  Future<String?> bestMove(String fen) async {
    final board = ch.Chess.fromFEN(fen);
    final count = (board.moves({'asObjects': true}) as List).length;
    if (count == 0) return null;
    final n = _workers.length;
    final chunk = (count + n - 1) ~/ n;
    final futures = <Future<Map<String, double>>>[];
    for (var i = 0; i < n; i++) {
      final lo = i * chunk;
      if (lo >= count) break;
      final hi = (lo + chunk) < count ? (lo + chunk) : count;
      final reply = ReceivePort();
      _workers[i].send([reply.sendPort, fen, lo, hi]);
      futures.add(reply.first.then((v) => (v as Map).cast<String, double>()));
    }
    final parts = await Future.wait(futures);
    String? best;
    var bestWin = double.negativeInfinity;
    for (final p in parts) {
      p.forEach((uci, win) {
        if (win > bestWin) {
          bestWin = win;
          best = uci;
        }
      });
    }
    return best;
  }

  static void _worker(List init) {
    final SendPort ready = init[0];
    final String npz = init[1];
    final engine = SearchlessEngine.load(npz);
    final rp = ReceivePort();
    ready.send(rp.sendPort);
    rp.listen((msg) {
      final SendPort reply = msg[0];
      final String fen = msg[1];
      final int lo = msg[2];
      final int hi = msg[3];
      reply.send(engine.evaluateRange(fen, lo, hi));
    });
  }
}

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
