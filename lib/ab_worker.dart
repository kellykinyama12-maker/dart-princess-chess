import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;

import 'mcts.dart' show MoveStat;
import 'alphabeta.dart';

/// AB value (position side-to-move POV) and weight to transfer into the MCTS
/// transposition table, or null to leave a quiet position to the net.
class AbInject {
  final double value; // in [-1, 1]
  final int weight; // like this many MCTS visits
  AbInject(this.value, this.weight);
}

/// Maps an [AbInfo] to a stat-transfer contribution: mates dominate; decisive
/// scores get a modest weight; quiet positions are skipped (net decides).
AbInject? abInjection(AbInfo info) {
  if (info.mate) {
    return AbInject(info.scoreCp > 0 ? 1.0 : -1.0, 1 << 16);
  }
  if (info.scoreCp.abs() < 150) return null;
  final x = info.scoreCp / 400.0;
  final v = x > 20
      ? 1.0
      : x < -20
          ? -1.0
          : (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1); // tanh
  final w = (info.depth * 2).clamp(4, 40);
  return AbInject(v, w);
}

/// Alpha-beta result for one position, as streamed back from the worker.
class AbInfo {
  final String fen;
  final String? bestUci;
  final int scoreCp; // side-to-move POV
  final int depth;
  final bool mate;
  AbInfo(this.fen, this.bestUci, this.scoreCp, this.depth, this.mate);
}

/// One isolate running a persistent [AlphaBeta] (its transposition table grows
/// across calls, so knowledge about positions the MCTS keeps revisiting
/// accumulates and is reused). The main isolate feeds it focus positions — the
/// root plus the MCTS's current best-move child nodes — and reads back scores.
class AbWorker {
  final Isolate _iso;
  final SendPort _send;
  final ReceivePort _recv;
  final Map<int, Completer<List<AbInfo>>> _pending = {};
  int _nextId = 0;

  AbWorker._(this._iso, this._send, this._recv) {
    _recv.listen((msg) {
      final list = msg as List;
      final id = list[0] as int;
      final rows = (list[1] as List).cast<List>();
      final infos = [
        for (final r in rows)
          AbInfo(r[0] as String, r[1] as String?, r[2] as int, r[3] as int,
              r[4] as bool)
      ];
      _pending.remove(id)?.complete(infos);
    });
  }

  static Future<AbWorker> start({bool complexity = false}) async {
    final ready = ReceivePort();
    final iso = await Isolate.spawn(_abMain, [ready.sendPort, complexity]);
    final send = await ready.first as SendPort;
    ready.close();
    return AbWorker._(iso, send, ReceivePort());
  }

  /// Searches each of [fens] (root first) sharing a total of [budgetMs], and
  /// returns the per-position results. [priors] is aligned with [fens] and
  /// gives each position's net policy (move UCI -> prior) for root move
  /// ordering, or null to fall back to MVV-LVA. Successive calls reuse the TT.
  Future<List<AbInfo>> focus(
      List<String> fens, List<Map<String, double>?> priors, int budgetMs) {
    final id = _nextId++;
    final c = Completer<List<AbInfo>>();
    _pending[id] = c;
    _send.send([id, fens, priors, budgetMs, _recv.sendPort]);
    return c.future;
  }

  void dispose() {
    _send.send(null);
    _iso.kill(priority: Isolate.immediate);
    _recv.close();
  }
}

void _abMain(List args) {
  final ready = args[0] as SendPort;
  final complexity = args[1] as bool;
  final ab = AlphaBeta(complexity: complexity); // persistent transposition table
  final port = ReceivePort();
  ready.send(port.sendPort);
  port.listen((msg) {
    if (msg == null) {
      port.close();
      return;
    }
    final list = msg as List;
    final id = list[0] as int;
    final fens = (list[1] as List).cast<String>();
    final priors = (list[2] as List).cast<Map<String, double>?>();
    final budgetMs = list[3] as int;
    final reply = list[4] as SendPort;

    // Root gets a double share; the rest split the remainder.
    final rows = <List>[];
    final per = fens.isEmpty ? 0 : (budgetMs ~/ (fens.length + 1));
    for (var i = 0; i < fens.length; i++) {
      final slice = (i == 0 ? per * 2 : per).clamp(5, budgetMs);
      final r = ab.search(fens[i],
          time: Duration(milliseconds: slice), rootPriors: priors[i]);
      rows.add([fens[i], r.bestUci, r.scoreCp, r.depth, r.isMate]);
    }
    reply.send([id, rows]);
  });
}

/// Combines the MCTS visit ranking with alpha-beta scores to pick the root
/// move. The alpha-beta search of the root itself is the authority (it is what
/// finds the tactic); it only overrides the MCTS choice when it sees a concrete
/// gap — a forced mate, or a line clearly better by [overrideMarginCp]
/// centipawns — so the net's positional judgment is kept in quiet positions.
String combineRoot({
  required String rootFen,
  required List<MoveStat> mctsStats,
  required Map<String, String> childFenByUci,
  required Map<String, AbInfo> abByFen,
  int overrideMarginCp = 150,
}) {
  if (mctsStats.isEmpty) return '';
  final mctsBest = mctsStats.first.uci;
  final legal = {for (final s in mctsStats) s.uci};

  final rootAb = abByFen[rootFen];
  if (rootAb == null ||
      rootAb.bestUci == null ||
      !legal.contains(rootAb.bestUci) ||
      rootAb.bestUci == mctsBest) {
    return mctsBest;
  }

  // Forced mate for the side to move: always trust it.
  if (rootAb.mate && rootAb.scoreCp > 0) return rootAb.bestUci!;

  // Otherwise override only if AB's root line is clearly better than MCTS's
  // move by AB's own reckoning (negated score of the MCTS move's child), and
  // that comparison was searched deep enough to be trustworthy.
  final mctsChild = childFenByUci[mctsBest];
  final mctsInfo = mctsChild == null ? null : abByFen[mctsChild];
  if (mctsInfo == null || mctsInfo.depth < 5) return mctsBest;
  final mctsVal = -mctsInfo.scoreCp; // root POV value of the MCTS move
  if (rootAb.scoreCp - mctsVal >= overrideMarginCp) return rootAb.bestUci!;
  return mctsBest;
}
