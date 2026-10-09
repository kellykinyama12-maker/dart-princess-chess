import 'dart:async';
import 'dart:isolate';

import 'package:dart_pytorch/dart_pytorch.dart' show Device;

import 'lc0_evaluator.dart';

/// A pool of worker isolates, each holding its own [Lc0Evaluator], so position
/// evaluations run truly in parallel across CPU cores. This turns the
/// virtual-loss leaf batches into a real speedup: a batch of distinct leaves is
/// evaluated concurrently instead of one-at-a-time on the main isolate.
class EvalPool {
  final List<Isolate> _isolates;
  final List<SendPort> _sendPorts;
  final ReceivePort _recv;
  final Map<int, Completer<Evaluation>> _pending = {};
  int _nextId = 0;
  int _rr = 0;

  EvalPool._(this._isolates, this._sendPorts, this._recv) {
    _recv.listen((msg) {
      final list = msg as List;
      final id = list[0] as int;
      final wdl = (list[1] as List).cast<double>();
      final priors = (list[2] as Map).cast<String, double>();
      _pending.remove(id)?.complete(Evaluation(wdl, priors));
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

  /// Evaluates one position; dispatched round-robin to a worker isolate.
  Future<Evaluation> evaluate(String fen, int stm, List<String> legal) {
    final id = _nextId++;
    final completer = Completer<Evaluation>();
    _pending[id] = completer;
    final sp = _sendPorts[_rr++ % _sendPorts.length];
    sp.send([id, fen, stm, legal, _recv.sendPort]);
    return completer.future;
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
    final legal = (list[3] as List).cast<String>();
    final reply = list[4] as SendPort;
    final e = evaluator.evaluate(fen, stm, legal);
    reply.send([id, e.wdl, e.priors]);
  });
}
