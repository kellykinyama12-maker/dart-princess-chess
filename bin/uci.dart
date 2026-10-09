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

import 'package:chess_moves/eval_pool.dart';
import 'package:chess_moves/lc0_evaluator.dart';
import 'package:chess_moves/mcts.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';
const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main() async {
  final engine = _Engine();
  final lines = stdin.transform(utf8.decoder).transform(const LineSplitter());
  await for (final line in lines) {
    final done = await engine.handle(line.trim());
    if (done) break;
  }
  engine.dispose();
}

class _Engine {
  int threads = 1;
  int defaultNodes = 800;
  double cpuct = 1.5;

  EvalPool? _pool;
  Mcts? _mcts;
  ch.Chess _board = ch.Chess.fromFEN(_startFen);

  String _baseFen = _startFen;
  List<String> _moves = [];
  bool _searching = false;

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
      case 'stop':
        _mcts?.stop();
      case 'quit':
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
    }
  }

  Future<void> _ensureEngine() async {
    if (_mcts != null) return;
    if (!File(_weightsPath).existsSync()) {
      _out('info string ERROR missing $_weightsPath (run bin/lc0_eval_smoke.dart)');
      return;
    }
    if (threads > 1) {
      _pool = await EvalPool.start(_weightsPath, threads);
      _mcts = Mcts.pooled(_pool!, cPuct: cpuct, batchSize: threads * 2);
    } else {
      final evaluator = Lc0Evaluator.load(_weightsPath);
      _mcts = Mcts(evaluator, cPuct: cpuct);
    }
  }

  void _disposeEngine() {
    _pool?.dispose();
    _pool = null;
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
    Duration? time;
    for (var i = 1; i < tok.length; i++) {
      switch (tok[i]) {
        case 'movetime':
          time = Duration(milliseconds: int.parse(tok[++i]));
        case 'nodes':
          nodes = int.parse(tok[++i]);
        case 'infinite':
          nodes = null;
          time = null;
        case 'depth': // no true depth notion; map to the default node budget
          ++i;
        case 'wtime':
          if (_board.turn == ch.Chess.WHITE) time = _alloc(tok, i);
          ++i;
        case 'btime':
          if (_board.turn == ch.Chess.BLACK) time = _alloc(tok, i);
          ++i;
      }
    }
    final hasInfinite = tok.contains('infinite');
    if (nodes == null && time == null && !hasInfinite) nodes = defaultNodes;

    _searching = true;
    final sw = Stopwatch()..start();
    final fen = _board.fen;
    final stats = _pool != null
        ? await _mcts!.searchBudgetAsync(fen, nodes: nodes, time: time)
        : _mcts!.searchBudget(fen, nodes: nodes, time: time);
    sw.stop();
    _searching = false;

    if (stats.isEmpty) {
      _out('bestmove 0000');
      return;
    }
    final best = stats.first;
    final nps = sw.elapsedMilliseconds > 0
        ? (_mcts!.rootVisits * 1000 ~/ sw.elapsedMilliseconds)
        : 0;
    final pv = _mcts!.principalVariation();
    final cp = _toCentipawns(-best.q); // side-to-move POV
    _out('info depth ${pv.length} nodes ${_mcts!.rootVisits} nps $nps '
        'score cp $cp time ${sw.elapsedMilliseconds} pv ${pv.join(' ')}');
    _out('bestmove ${best.uci}');
  }

  Duration _alloc(List<String> tok, int i) {
    final ms = int.parse(tok[i + 1]);
    return Duration(milliseconds: math.max(50, ms ~/ 30));
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

  void _out(String s) {
    stdout.writeln(s);
  }
}
