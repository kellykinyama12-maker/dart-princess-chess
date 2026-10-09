// Stage 2 demo: run lc0's real network inside a PUCT MCTS and print the search
// result for a position. Reuses lc0's weights (Lc0Net) as the leaf evaluator,
// with Dirichlet root noise and tree reuse across moves.
//
// Usage:
//   dart run bin/play_search.dart --sims 800
//   dart run bin/play_search.dart --sims 800 --play 10   # self-play, reuses tree
//   dart run bin/play_search.dart --sims 800 --batch 8    # virtual-loss batching
//   dart run bin/play_search.dart --sims 800 --workers 4  # parallel evals (isolates)
//   dart run bin/play_search.dart --play 20 --temp 1 --temp-moves 10  # temp decay
//   dart run bin/play_search.dart --sims 400 "<FEN>"
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:dart_pytorch/dart_pytorch.dart' show Device;

import 'package:chess_moves/eval_pool.dart';
import 'package:chess_moves/lc0_evaluator.dart';
import 'package:chess_moves/mcts.dart';

const _weightsPath = 'models/lc0/744706.pb.gz';
const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

Future<void> main(List<String> args) async {
  var sims = 800;
  var play = 0;
  var batch = 1;
  var workers = 1;
  var temp = 0.0;
  var tempMoves = 0;
  var fpu = 0.2;
  int? seed;
  String? net;
  final rest = <String>[];
  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--sims':
        sims = int.parse(args[++i]);
      case '--play':
        play = int.parse(args[++i]);
      case '--batch':
        batch = int.parse(args[++i]);
      case '--workers':
        workers = int.parse(args[++i]);
      case '--temp':
        temp = double.parse(args[++i]);
      case '--temp-moves':
        tempMoves = int.parse(args[++i]);
      case '--fpu':
        fpu = double.parse(args[++i]);
      case '--seed':
        seed = int.parse(args[++i]);
      case '--net':
        net = args[++i];
      default:
        rest.add(args[i]);
    }
  }
  final fen = rest.isNotEmpty ? rest.join(' ') : _startFen;
  final weightsPath = net ?? Platform.environment['LC0_WEIGHTS'] ?? _weightsPath;

  if (!File(weightsPath).existsSync()) {
    stderr.writeln('Missing $weightsPath. Run bin/lc0_eval_smoke.dart first to '
        'download the lc0 weights.');
    exit(64);
  }

  // Parallel evals need enough leaves in flight to keep every worker busy.
  if (workers > 1 && batch <= 1) batch = workers * 2;

  EvalPool? pool;
  Mcts mcts;
  if (workers > 1) {
    stdout.writeln('cpus: ${Platform.numberOfProcessors}  '
        'starting $workers eval isolates (CPU) ...');
    pool = await EvalPool.start(weightsPath, workers);
    mcts = Mcts.pooled(pool, batchSize: batch, fpuReduction: fpu, seed: seed);
  } else {
    stdout.writeln('loading lc0 net (CPU) ...');
    final evaluator = Lc0Evaluator.load(weightsPath, device: Device.CPU);
    mcts = Mcts(evaluator, batchSize: batch, fpuReduction: fpu, seed: seed);
  }

  stdout.writeln('position : $fen');
  stdout.writeln('sims     : $sims  batch: $batch  workers: $workers  '
      'temp: $temp (${tempMoves}p)  fpu: $fpu\n');

  final pooled = pool != null;
  if (play <= 0) {
    await _analyze(mcts, fen, sims, temp, pooled);
  } else {
    await _selfPlay(mcts, fen, sims, play, temp, tempMoves, pooled);
  }
  pool?.dispose();
}

Future<List<MoveStat>> _run(Mcts mcts, String fen, int sims, bool pooled) =>
    pooled ? mcts.searchAsync(fen, sims) : Future.value(mcts.search(fen, sims));

Future<void> _analyze(
    Mcts mcts, String fen, int sims, double temp, bool pooled) async {
  final sw = Stopwatch()..start();
  final stats = await _run(mcts, fen, sims, pooled);
  sw.stop();
  _printStats(stats);
  if (stats.isNotEmpty) {
    final pick = mcts.selectMove(stats, temperature: temp);
    stdout.writeln('\nbest move: ${pick.uci}  '
        '(${sw.elapsedMilliseconds} ms, '
        '${(sw.elapsedMilliseconds / sims).toStringAsFixed(0)} ms/sim)');
  }
}

Future<void> _selfPlay(Mcts mcts, String startFen, int sims, int plies,
    double temp, int tempMoves, bool pooled) async {
  final board = ch.Chess.fromFEN(startFen);
  final line = <String>[];
  for (var ply = 0; ply < plies && !board.game_over; ply++) {
    final reused = mcts.rootVisits; // carried over from the previous move
    final stats = await _run(mcts, board.fen, sims, pooled);
    if (stats.isEmpty) break;
    final t = ply < tempMoves ? temp : 0.0; // temperature decay
    final pick = mcts.selectMove(stats, temperature: t);
    stdout.writeln('move ${ply + 1}: ${pick.uci}  '
        'N=${pick.visits}  Q=${pick.q.toStringAsFixed(3)}  '
        't=${t.toStringAsFixed(1)}  (reused $reused visits)');
    _applyUci(board, pick.uci);
    line.add(pick.uci);
    mcts.advance(pick.uci); // reuse this subtree as the next root
  }
  stdout.writeln('\nline: ${line.join(' ')}');
  stdout.writeln('final: ${board.fen}');
  if (board.game_over) {
    final result = board.in_checkmate
        ? 'checkmate'
        : (board.in_stalemate ? 'stalemate' : 'draw');
    stdout.writeln('result: $result');
  }
}

void _applyUci(ch.Chess board, String uci) {
  final move = <String, String>{
    'from': uci.substring(0, 2),
    'to': uci.substring(2, 4),
  };
  if (uci.length > 4) move['promotion'] = uci.substring(4);
  board.move(move);
}

void _printStats(List<MoveStat> stats) {
  stdout.writeln('top moves (visits / Q / prior):');
  for (final s in stats.take(8)) {
    stdout.writeln('  ${s.uci.padRight(6)} '
        'N=${s.visits.toString().padLeft(4)}  '
        'Q=${s.q.toStringAsFixed(3).padLeft(7)}  '
        'P=${(s.prior * 100).toStringAsFixed(1).padLeft(5)}%');
  }
}
