// Minimal UCI engine wrapper around the lc0-net PUCT search, so the engine can
// play in any chess GUI or against other engines.
//
//   dart run bin/uci.dart
//   (then speak UCI: "uci", "isready", "position startpos moves e2e4", "go movetime 2000")
//
// Options: Threads (worker isolates), Nodes (default go budget), Cpuct.
import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

import 'package:chess_moves/ab_worker.dart';
import 'package:chess_moves/eval_pool.dart';
import 'package:chess_moves/lc0_evaluator.dart';
import 'package:chess_moves/mcts.dart';

const _defaultWeightsPath = 'models/lc0/744706.pb.gz';
const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// Network can be overridden via the LC0_WEIGHTS env var or `setoption WeightsFile`.
String _resolveWeights() =>
    Platform.environment['LC0_WEIGHTS'] ?? _defaultWeightsPath;

Future<void> main() async {
  final engine = _Engine();
  final lines = stdin.transform(utf8.decoder).transform(const LineSplitter());
  await for (final line in lines) {
    final done = await engine.handle(line.trim());
    if (done) break;
  }
  engine.stopSearch(); // EOF behaves like quit: end any unbounded search
  await engine.awaitSearch();
  engine.dispose();
}

class _Engine {
  int threads = 1;
  int defaultNodes = 800;
  double cpuct = 1.5;
  bool ponderEnabled = false;
  bool useAb = true; // run the alpha-beta tactical worker alongside MCTS
  String weightsPath = _resolveWeights();

  EvalPool? _pool;
  AbWorker? _ab;
  Mcts? _mcts;
  ch.Chess _board = ch.Chess.fromFEN(_startFen);

  String _baseFen = _startFen;
  List<String> _moves = [];
  bool _searching = false;
  bool _pondering = false;
  Timer? _timer; // fires _mcts.stop() when the time budget elapses
  int? _pendingBudgetMs; // budget to apply on ponderhit
  Stopwatch _sw = Stopwatch();
  Future<void>? _searchFuture;

  Future<bool> handle(String line) async {
    if (line.isEmpty) return false;
    final tok = line.split(RegExp(r'\s+'));
    switch (tok[0]) {
      case 'uci':
        _out('id name ChessMoves-lc0');
        _out('id author chess_moves');
        _out('option name Threads type spin default 1 min 1 max 64');
        _out('option name Nodes type spin default 800 min 1 max 100000000');
        _out('option name Cpuct type string default 1.5');
        _out('option name Ponder type check default false');
        _out('option name UseAB type check default true');
        _out('option name WeightsFile type string default $weightsPath');
        _out('uciok');
      case 'isready':
        await _ensureEngine();
        _out('readyok');
      case 'setoption':
        _setOption(tok);
      case 'ucinewgame':
        _mcts?.reset();
      case 'position':
        _position(tok);
      case 'go':
        await _go(tok);
      case 'ponderhit':
        _ponderHit();
      case 'stop':
        _timer?.cancel();
        _timer = null;
        _mcts?.stop();
      case 'quit':
        _mcts?.stop();
        return true;
    }
    return false;
  }

  void _setOption(List<String> tok) {
    final ni = tok.indexOf('name');
    final vi = tok.indexOf('value');
    if (ni < 0 || vi < 0) return;
    final name = tok.sublist(ni + 1, vi).join(' ').toLowerCase();
    final value = tok.sublist(vi + 1).join(' ');
    switch (name) {
      case 'threads':
        final t = int.tryParse(value) ?? threads;
        if (t != threads) {
          threads = t;
          _disposeEngine(); // rebuilt lazily on next isready/go
        }
      case 'nodes':
        defaultNodes = int.tryParse(value) ?? defaultNodes;
      case 'cpuct':
        final c = double.tryParse(value);
        if (c != null && c != cpuct) {
          cpuct = c;
          _disposeEngine();
        }
      case 'ponder':
        ponderEnabled = value.toLowerCase() == 'true';
      case 'useab':
        final v = value.toLowerCase() == 'true';
        if (v != useAb) {
          useAb = v;
          _disposeEngine();
        }
      case 'weightsfile':
        if (value.isNotEmpty && value != weightsPath) {
          weightsPath = value;
          _disposeEngine();
        }
    }
  }

  Future<void> _ensureEngine() async {
    if (_mcts != null) return;
    if (!File(weightsPath).existsSync()) {
      _out('info string ERROR missing $weightsPath (run bin/lc0_eval_smoke.dart)');
      return;
    }
    if (threads > 1) {
      _pool = await EvalPool.start(weightsPath, threads);
      _mcts = Mcts.pooled(_pool!, cPuct: cpuct, batchSize: threads * 2);
      if (useAb) _ab = await AbWorker.start();
    } else {
      final evaluator = Lc0Evaluator.load(weightsPath);
      _mcts = Mcts(evaluator, cPuct: cpuct);
    }
  }

  void _disposeEngine() {
    _pool?.dispose();
    _pool = null;
    _ab?.dispose();
    _ab = null;
    _mcts = null;
  }

  void _position(List<String> tok) {
    var i = 1;
    String base;
    if (i < tok.length && tok[i] == 'startpos') {
      base = _startFen;
      i++;
    } else if (i < tok.length && tok[i] == 'fen') {
      base = tok.sublist(i + 1, math.min(i + 7, tok.length)).join(' ');
      i += 7;
    } else {
      return;
    }
    final moves = <String>[];
    final mi = tok.indexOf('moves', i);
    if (mi >= 0) moves.addAll(tok.sublist(mi + 1));

    // Tree reuse: if this position just extends the previous one, advance the
    // search tree over the appended moves instead of discarding it.
    final extend = base == _baseFen &&
        moves.length >= _moves.length &&
        _listStartsWith(moves, _moves);
    if (extend && _mcts != null) {
      for (var k = _moves.length; k < moves.length; k++) {
        _mcts!.advance(moves[k]);
      }
    } else {
      _mcts?.reset();
    }

    _board = ch.Chess.fromFEN(base);
    for (final uci in moves) {
      _applyUci(_board, uci);
    }
    _baseFen = base;
    _moves = moves;
  }

  Future<void> _go(List<String> tok) async {
    if (_searching) return;
    await _ensureEngine();
    if (_mcts == null) {
      _out('bestmove 0000');
      return;
    }
    int? nodes;
    int? movetime, wtime, btime, winc, binc, movestogo;
    var ponder = false;
    final infinite = tok.contains('infinite');
    for (var i = 1; i < tok.length; i++) {
      switch (tok[i]) {
        case 'movetime':
          movetime = int.parse(tok[++i]);
        case 'nodes':
          nodes = int.parse(tok[++i]);
        case 'wtime':
          wtime = int.parse(tok[++i]);
        case 'btime':
          btime = int.parse(tok[++i]);
        case 'winc':
          winc = int.parse(tok[++i]);
        case 'binc':
          binc = int.parse(tok[++i]);
        case 'movestogo':
          movestogo = int.parse(tok[++i]);
        case 'ponder':
          ponder = true;
        case 'depth': // no true depth notion
          ++i;
      }
    }
    final budgetMs = _computeBudgetMs(
        movetime: movetime,
        wtime: wtime,
        btime: btime,
        winc: winc,
        binc: binc,
        movestogo: movestogo);

    final fen = _board.fen;
    _searching = true;
    _pondering = ponder;
    _pendingBudgetMs = budgetMs;
    _sw = Stopwatch()..start();
    _timer?.cancel();
    _timer = null;

    if (_pool != null) {
      // Async + interruptible: a Timer fires stop() at the deadline; `ponder`
      // and `infinite` run until ponderhit/stop.
      if (!ponder && !infinite && budgetMs != null) {
        _timer = Timer(Duration(milliseconds: budgetMs), () => _mcts?.stop());
      }
      _searchFuture = _runAsyncSearch(fen, nodes);
    } else {
      // Sync blocking search can't be interrupted; bound it by nodes or a
      // captured time budget (ponder/infinite fall back to a node budget).
      Duration? time;
      if (!ponder && !infinite && budgetMs != null) {
        time = Duration(milliseconds: budgetMs);
      }
      if (time == null && nodes == null) nodes = defaultNodes;
      _finishSearch(_mcts!.searchBudget(fen, nodes: nodes, time: time));
    }
  }

  Future<void> _runAsyncSearch(String fen, int? nodes) async {
    final abByFen = <String, AbInfo>{};
    try {
      var done = false;
      final searchFut = _mcts!
          .searchBudgetAsync(fen, nodes: nodes)
          .whenComplete(() => done = true);
      // Feed the root + MCTS best-move nodes (with their net priors) to the
      // alpha-beta worker in parallel until the MCTS search finishes.
      Future<void> feed() async {
        while (!done && _ab != null) {
          final focus = <String>[fen, ..._mcts!.topRootChildFens(2)];
          final pri = _mcts!.focusPriors(2);
          final priors = [for (final f in focus) pri[f]];
          final infos = await _ab!.focus(focus, priors, 150);
          for (final info in infos) {
            abByFen[info.fen] = info;
          }
        }
      }

      final feedFut = _ab != null ? feed() : Future<void>.value();
      final results = await searchFut;
      await feedFut;
      _finishSearch(results, abByFen);
    } catch (_) {
      // engine disposed mid-search (e.g. quit) — ignore
    }
  }

  void _ponderHit() {
    if (!_pondering) return;
    _pondering = false;
    final b = _pendingBudgetMs;
    if (b != null) {
      _timer?.cancel();
      _timer = Timer(Duration(milliseconds: b), () => _mcts?.stop());
    }
  }

  void _finishSearch(List<MoveStat> stats,
      [Map<String, AbInfo> abByFen = const {}]) {
    _timer?.cancel();
    _timer = null;
    _sw.stop();
    _searching = false;
    _pondering = false;
    if (stats.isEmpty) {
      _out('bestmove 0000');
      return;
    }
    final best = stats.first;
    final ms = _sw.elapsedMilliseconds;
    final nps = ms > 0 ? (_mcts!.rootVisits * 1000 ~/ ms) : 0;
    final pv = _mcts!.principalVariation();
    final cp = _toCentipawns(-best.q); // side-to-move POV
    _out('info depth ${pv.length} nodes ${_mcts!.rootVisits} nps $nps '
        'score cp $cp time $ms pv ${pv.join(' ')}');

    // Tactical override: let alpha-beta correct the MCTS move when it sees a
    // concrete gap (mate / material), else keep the net's choice.
    var chosen = best.uci;
    if (_ab != null && abByFen.isNotEmpty && _mcts!.currentRootFen != null) {
      chosen = combineRoot(
        rootFen: _mcts!.currentRootFen!,
        mctsStats: stats,
        childFenByUci: _mcts!.rootChildFens(),
        abByFen: abByFen,
      );
      if (chosen != best.uci) {
        _out('info string ab override ${best.uci} -> $chosen');
      }
    }
    final ponderMove = (ponderEnabled && chosen == best.uci && pv.length >= 2)
        ? ' ponder ${pv[1]}'
        : '';
    _out('bestmove $chosen$ponderMove');
  }

  /// Time to spend this move in ms, or null for no time limit.
  int? _computeBudgetMs(
      {int? movetime,
      int? wtime,
      int? btime,
      int? winc,
      int? binc,
      int? movestogo}) {
    if (movetime != null) return movetime;
    final white = _board.turn == ch.Chess.WHITE;
    final my = white ? wtime : btime;
    if (my == null) return null; // no clock info (nodes-only / infinite)
    final inc = (white ? winc : binc) ?? 0;
    final mtg = (movestogo != null && movestogo > 0) ? movestogo : 30;
    const overhead = 30; // communication/safety margin
    final alloc = my ~/ (mtg + 1) + (inc * 3 ~/ 4);
    final cap = ((my - overhead) * 4) ~/ 5; // never spend >80% of the clock
    if (cap < 10) return math.max(1, my - overhead); // in time trouble
    return alloc.clamp(10, cap);
  }

  int _toCentipawns(double q) {
    final c = q.clamp(-0.999, 0.999);
    return (111.714640912 * math.tan(1.5620688421 * c)).round();
  }

  void _applyUci(ch.Chess board, String uci) {
    final move = <String, String>{
      'from': uci.substring(0, 2),
      'to': uci.substring(2, 4),
    };
    if (uci.length > 4) move['promotion'] = uci.substring(4);
    board.move(move);
  }

  bool _listStartsWith(List<String> list, List<String> prefix) {
    if (prefix.length > list.length) return false;
    for (var i = 0; i < prefix.length; i++) {
      if (list[i] != prefix[i]) return false;
    }
    return true;
  }

  void dispose() => _disposeEngine();

  void stopSearch() {
    _timer?.cancel();
    _timer = null;
    _mcts?.stop();
  }

  /// Waits for any in-flight detached search to finish (emitting its bestmove).
  Future<void> awaitSearch() async {
    try {
      await _searchFuture;
    } catch (_) {}
  }

  void _out(String s) {
    stdout.writeln(s);
  }
}
