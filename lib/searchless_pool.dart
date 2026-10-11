/// A pool of worker isolates, each holding its own copy of the searchless net,
/// that score disjoint slices of a position's legal moves in parallel.
library;

import 'dart:async';
import 'dart:isolate';

import 'package:chess/chess.dart' as ch;

import 'searchless.dart';

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

  int get workers => _workers.length;

  /// Win probability (side-to-move) for every legal move, by UCI.
  Future<Map<String, double>> evaluate(String fen) async {
    final board = ch.Chess.fromFEN(fen);
    final count = (board.moves({'asObjects': true}) as List).length;
    final merged = <String, double>{};
    if (count == 0) return merged;
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
    for (final part in await Future.wait(futures)) {
      merged.addAll(part);
    }
    return merged;
  }

  /// Highest win-probability move, or null if none.
  Future<String?> bestMove(String fen) async {
    final q = await evaluate(fen);
    String? best;
    var bestWin = double.negativeInfinity;
    q.forEach((uci, win) {
      if (win > bestWin) {
        bestWin = win;
        best = uci;
      }
    });
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
