import 'dart:async';
import 'dart:isolate';

import 'package:dart_pytorch/dart_pytorch.dart' show Device;

import 'board_expand.dart';
import 'lc0_evaluator.dart';

/// Full leaf expansion computed on a worker isolate: both the board work
/// (legal moves + child FENs, or terminal detection) and the network eval.
class ExpandResult {
  final bool terminal;
  final bool checkmate;
  final List<double> wdl; // STM POV; empty when terminal
  final List<String> legal; // absolute UCI
  final List<String> childFens; // aligned with [legal]
  final List<double> priors; // aligned with [legal]

  ExpandResult(this.terminal, this.checkmate, this.wdl, this.legal,
      this.childFens, this.priors);

  double get value => wdl[0] - wdl[2];
}

/// A pool of worker isolates, each holding its own [Lc0Evaluator]. Each worker
/// does the full leaf expansion (move generation + network eval), so both the
/// chess work and the eval run in parallel, off the main isolate. This turns
/// the virtual-loss leaf batches into a real speedup.
class EvalPool {
  final List<Isolate> _isolates;
  final List<SendPort> _sendPorts;
  final ReceivePort _recv;
  final Map<int, Completer<ExpandResult>> _pending = {};
  // lc0-style NN cache: reuse a position's eval instead of re-running the net.
  final Map<String, ExpandResult> _cache = {};
  final Map<String, Future<ExpandResult>> _inflight = {};
  static const int _maxCache = 200000;
  int _hits = 0, _misses = 0;
  int _nextId = 0;
  int _rr = 0;

  EvalPool._(this._isolates, this._sendPorts, this._recv) {
    _recv.listen((msg) {
      final list = msg as List;
      final id = list[0] as int;
      final terminal = list[1] as bool;
      final ExpandResult result;
      if (terminal) {
        result = ExpandResult(
            true, list[2] as bool, const [], const [], const [], const []);
      } else {
        result = ExpandResult(
          false,
          false,
          (list[2] as List).cast<double>(),
          (list[3] as List).cast<String>(),
          (list[4] as List).cast<String>(),
          (list[5] as List).cast<double>(),
        );
      }
      _pending.remove(id)?.complete(result);
    });
  }

  int get workers => _sendPorts.length;

  /// Spawns [workers] isolates, each loading the net from [weightsPath].
  static Future<EvalPool> start(String weightsPath, int workers) async {
    final isolates = <Isolate>[];
    final sendPorts = <SendPort>[];
    for (var i = 0; i < workers; i++) {
      final ready = ReceivePort();
      final iso = await Isolate.spawn(_worker, [weightsPath, ready.sendPort]);
      sendPorts.add(await ready.first as SendPort);
      isolates.add(iso);
      ready.close();
    }
    return EvalPool._(isolates, sendPorts, ReceivePort());
  }

  /// Expands one position (move-gen + eval). Served from the NN cache on a
  /// repeat, shared with an in-flight request for the same position, else
  /// dispatched round-robin to a worker isolate.
  Future<ExpandResult> expand(String fen, int stm) {
    final cached = _cache[fen];
    if (cached != null) {
      _hits++;
      return Future.value(cached);
    }
    final pending = _inflight[fen];
    if (pending != null) {
      _hits++;
      return pending;
    }
    _misses++;
    final id = _nextId++;
    final completer = Completer<ExpandResult>();
    _pending[id] = completer;
    final fut = completer.future.then((r) {
      _inflight.remove(fen);
      _put(fen, r);
      return r;
    });
    _inflight[fen] = fut;
    final sp = _sendPorts[_rr++ % _sendPorts.length];
    sp.send([id, fen, stm, _recv.sendPort]);
    return fut;
  }

  void _put(String fen, ExpandResult r) {
    if (_cache.length >= _maxCache) {
      for (final k in _cache.keys.take(_cache.length ~/ 4).toList()) {
        _cache.remove(k);
      }
    }
    _cache[fen] = r;
  }

  /// NN-cache hit rate (diagnostics / UCI info).
  String cacheStats() {
    final total = _hits + _misses;
    final pct = total == 0 ? 0 : (100 * _hits / total).round();
    return 'nncache hits=$_hits misses=$_misses size=${_cache.length} hit=$pct%';
  }

  /// Drops cached evals (call on a new game).
  void clearCache() {
    _cache.clear();
    _inflight.clear();
    _hits = 0;
    _misses = 0;
  }

  void dispose() {
    for (final sp in _sendPorts) {
      sp.send(null);
    }
    for (final iso in _isolates) {
      iso.kill(priority: Isolate.immediate);
    }
    _recv.close();
  }
}

void _worker(List<Object> args) {
  final weightsPath = args[0] as String;
  final ready = args[1] as SendPort;
  final evaluator = Lc0Evaluator.load(weightsPath, device: Device.CPU);
  final port = ReceivePort();
  ready.send(port.sendPort);
  port.listen((msg) {
    if (msg == null) {
      port.close();
      return;
    }
    final list = msg as List;
    final id = list[0] as int;
    final fen = list[1] as String;
    final stm = list[2] as int;
    final reply = list[3] as SendPort;

    final exp = expandBoard(fen);
    if (exp.gameOver) {
      reply.send([id, true, exp.checkmate]);
      return;
    }
    final e = evaluator.evaluate(fen, stm, exp.legal);
    final priors = [for (final uci in exp.legal) e.priors[uci] ?? 0.0];
    reply.send([id, false, e.wdl, exp.legal, exp.childFens, priors]);
  });
}
