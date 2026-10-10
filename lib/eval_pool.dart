import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;

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
  final Map<int, ExpandResult> _cache = {};
  final Map<int, Future<ExpandResult>> _inflight = {};
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
    final key = _zobrist.ofFen(fen);
    final cached = _cache[key];
    if (cached != null) {
      _hits++;
      return Future.value(cached);
    }
    final pending = _inflight[key];
    if (pending != null) {
      _hits++;
      return pending;
    }
    _misses++;
    final id = _nextId++;
    final completer = Completer<ExpandResult>();
    _pending[id] = completer;
    final fut = completer.future.then((r) {
      _inflight.remove(key);
      _put(key, r);
      return r;
    });
    _inflight[key] = fut;
    final sp = _sendPorts[_rr++ % _sendPorts.length];
    sp.send([id, fen, stm, _recv.sendPort]);
    return fut;
  }

  void _put(int key, ExpandResult r) {
    if (_cache.length >= _maxCache) {
      for (final k in _cache.keys.take(_cache.length ~/ 4).toList()) {
        _cache.remove(k);
      }
    }
    _cache[key] = r;
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

/// Shared Zobrist hasher for NN-cache keys (int keys beat FEN strings for map
/// lookups and memory). Keyed on the position identity the net eval depends on:
/// placement, side-to-move, castling, en-passant file and the rule-50 counter.
final _Zobrist _zobrist = _Zobrist();

class _Zobrist {
  final List<int> _piece = List<int>.filled(12 * 64, 0);
  final List<int> _castle = List<int>.filled(4, 0);
  final List<int> _ep = List<int>.filled(8, 0);
  final List<int> _rule50 = List<int>.filled(101, 0);
  late final int _black;

  _Zobrist() {
    final r = math.Random(0x1a2b3c4d);
    int rnd() => (r.nextInt(1 << 32) << 32) ^ r.nextInt(1 << 32);
    for (var i = 0; i < _piece.length; i++) {
      _piece[i] = rnd();
    }
    for (var i = 0; i < 4; i++) {
      _castle[i] = rnd();
    }
    for (var i = 0; i < 8; i++) {
      _ep[i] = rnd();
    }
    for (var i = 0; i < 101; i++) {
      _rule50[i] = rnd();
    }
    _black = rnd();
  }

  int ofFen(String fen) {
    final parts = fen.split(' ');
    final placement = parts[0];
    var h = 0;
    var sq = 0;
    for (var i = 0; i < placement.length; i++) {
      final c = placement.codeUnitAt(i);
      if (c == 0x2F) continue; // '/'
      if (c >= 0x31 && c <= 0x38) {
        sq += c - 0x30; // '1'..'8' empty squares
        continue;
      }
      final idx = _pieceIndex(c);
      if (idx >= 0) {
        h ^= _piece[idx * 64 + sq];
        sq++;
      }
    }
    if (parts.length > 1 && parts[1] == 'b') h ^= _black;
    if (parts.length > 2) {
      final cs = parts[2];
      if (cs.contains('K')) h ^= _castle[0];
      if (cs.contains('Q')) h ^= _castle[1];
      if (cs.contains('k')) h ^= _castle[2];
      if (cs.contains('q')) h ^= _castle[3];
    }
    if (parts.length > 3 && parts[3] != '-' && parts[3].isNotEmpty) {
      final f = parts[3].codeUnitAt(0) - 0x61; // 'a'..'h'
      if (f >= 0 && f < 8) h ^= _ep[f];
    }
    if (parts.length > 4) {
      final r50 = int.tryParse(parts[4]) ?? 0;
      h ^= _rule50[r50.clamp(0, 100)];
    }
    return h;
  }

  int _pieceIndex(int c) {
    switch (c) {
      case 0x50:
        return 0; // P
      case 0x4E:
        return 1; // N
      case 0x42:
        return 2; // B
      case 0x52:
        return 3; // R
      case 0x51:
        return 4; // Q
      case 0x4B:
        return 5; // K
      case 0x70:
        return 6; // p
      case 0x6E:
        return 7; // n
      case 0x62:
        return 8; // b
      case 0x72:
        return 9; // r
      case 0x71:
        return 10; // q
      case 0x6B:
        return 11; // k
      default:
        return -1;
    }
  }
}
