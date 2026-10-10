// A minimal cutechess-cli-style match manager for UCI engines, written in Dart.
// Drives two UCI engines, enforces time control, detects game end via
// package:chess, alternates colors, and reports results (optionally PGN).
//
//   # self-play of our engine, 4 games, 1s/move
//   dart run bin/match.dart --e1 "dart run bin/uci.dart" --opt1 Threads=8 \
//       --opt1 UseAB=true --games 4 --movetime 1000
//
//   # our engine vs lc0, 10 games, 10s + 0.1s increment
//   dart run bin/match.dart \
//       --e1 "dart run bin/uci.dart" --name1 AB --opt1 Threads=8 \
//       --e2 "C:\projects\lc0ref\lc0.exe" --name2 lc0 \
//           --opt2 WeightsFile=models/lc0/t1-256x10-distilled.pb.gz \
//       --games 10 --tc 10+0.1 --pgn match.pgn
import 'dart:async';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/uci_engine.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

class _Spec {
  String? cmd;
  String? name;
  final Map<String, String> opts = {};
}

Future<void> main(List<String> args) async {
  final e1 = _Spec();
  final e2 = _Spec();
  var games = 2;
  int? movetimeMs;
  int? baseMs, incMs; // time control
  String? openingsFile;
  String? pgnFile;
  var maxMoves = 200; // plies before draw adjudication

  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--e1':
        e1.cmd = args[++i];
      case '--e2':
        e2.cmd = args[++i];
      case '--name1':
        e1.name = args[++i];
      case '--name2':
        e2.name = args[++i];
      case '--opt1':
        _addOpt(e1.opts, args[++i]);
      case '--opt2':
        _addOpt(e2.opts, args[++i]);
      case '--games':
        games = int.parse(args[++i]);
      case '--movetime':
        movetimeMs = int.parse(args[++i]);
      case '--tc':
        final tc = _parseTc(args[++i]);
        baseMs = tc[0];
        incMs = tc[1];
      case '--openings':
        openingsFile = args[++i];
      case '--pgn':
        pgnFile = args[++i];
      case '--maxmoves':
        maxMoves = int.parse(args[++i]);
      default:
        stderr.writeln('unknown arg: ${args[i]}');
        exit(64);
    }
  }
  if (e1.cmd == null) {
    stderr.writeln('need --e1 "<engine command>"');
    exit(64);
  }
  if (e2.cmd == null) {
    // Default to self-play.
    e2.cmd = e1.cmd;
    e2.opts.addAll(e1.opts);
  }
  e1.name ??= _deriveName(e1.cmd!, 1);
  e2.name ??= _deriveName(e2.cmd!, 2);
  if (e1.name == e2.name) e2.name = '${e2.name}#2';
  if (movetimeMs == null && baseMs == null) movetimeMs = 1000;

  final openings = openingsFile != null
      ? _loadOpenings(openingsFile)
      : <String>[_startFen];

  stdout.writeln('Match: ${e1.name} vs ${e2.name}  |  '
      '${movetimeMs != null ? 'movetime ${movetimeMs}ms' : 'tc ${baseMs! ~/ 1000}+${(incMs! / 1000)}'}'
      '  |  $games games');

  final eng1 = await UciEngine.start(e1.cmd!, name: e1.name!, options: e1.opts);
  final eng2 = await UciEngine.start(e2.cmd!, name: e2.name!, options: e2.opts);

  IOSink? pgn;
  if (pgnFile != null) pgn = File(pgnFile).openWrite(mode: FileMode.append);

  var w1 = 0, d = 0, l1 = 0; // from engine 1's perspective
  for (var g = 0; g < games; g++) {
    final opening = openings[(g ~/ 2) % openings.length];
    final e1White = g.isEven; // alternate colors each game
    final white = e1White ? eng1 : eng2;
    final black = e1White ? eng2 : eng1;

    final res = await _playGame(white, black, opening,
        movetimeMs: movetimeMs,
        baseMs: baseMs,
        incMs: incMs,
        maxMoves: maxMoves);

    // Map white-POV result to engine 1.
    final s1 = e1White ? res.whiteScore : (1.0 - res.whiteScore);
    if (s1 == 1.0) {
      w1++;
    } else if (s1 == 0.0) {
      l1++;
    } else {
      d++;
    }

    final tag = res.whiteScore == 1.0
        ? '1-0'
        : res.whiteScore == 0.0
            ? '0-1'
            : '1/2-1/2';
    stdout.writeln('Game ${g + 1}: ${white.name}(W) vs ${black.name}(B)  '
        '$tag  {${res.reason}}  ${res.sans.length} plies');
    if (pgn != null) {
      pgn.write(_pgn(white.name, black.name, tag, opening, res.sans, g + 1));
    }
    stdout.writeln('  score ${e1.name} ${w1}-${l1}-$d '
        '(+$w1 =$d -$l1), ${_pct(w1, d, l1)}%');
  }

  await pgn?.flush();
  await pgn?.close();
  await eng1.quit();
  await eng2.quit();

  stdout.writeln('\nFinal: ${e1.name} vs ${e2.name}  '
      '+$w1 =$d -$l1  (${_pct(w1, d, l1)}%)');
}

class _GameResult {
  final double whiteScore; // 1.0 white win, 0.0 black win, 0.5 draw
  final String reason;
  final List<String> sans;
  _GameResult(this.whiteScore, this.reason, this.sans);
}

Future<_GameResult> _playGame(UciEngine white, UciEngine black, String startFen,
    {int? movetimeMs, int? baseMs, int? incMs, required int maxMoves}) async {
  white.newGame();
  black.newGame();
  await white.isReady();
  await black.isReady();

  final board = ch.Chess.fromFEN(startFen);
  final startpos = startFen == _startFen;
  final moves = <String>[]; // uci moves played
  final sans = <String>[];
  var wClock = baseMs ?? 0, bClock = baseMs ?? 0;

  while (!board.game_over) {
    if (moves.length >= maxMoves) {
      return _GameResult(0.5, 'adjudicated: max moves', sans);
    }
    final whiteToMove = board.turn == ch.Chess.WHITE;
    final engine = whiteToMove ? white : black;
    final clock = whiteToMove ? wClock : bClock;

    final position = startpos
        ? 'position startpos${moves.isEmpty ? '' : ' moves ${moves.join(' ')}'}'
        : 'position fen $startFen${moves.isEmpty ? '' : ' moves ${moves.join(' ')}'}';
    final go = movetimeMs != null
        ? 'go movetime $movetimeMs'
        : 'go wtime $wClock btime $bClock winc ${incMs ?? 0} binc ${incMs ?? 0}';
    final hardLimit = Duration(
        milliseconds: (movetimeMs != null ? movetimeMs * 2 : clock) + 5000);

    final sw = Stopwatch()..start();
    String best;
    try {
      best = await engine.bestMove(position, go, hardLimit);
    } on TimeoutException {
      return _GameResult(whiteToMove ? 0.0 : 1.0,
          '${engine.name} forfeits: no move (timeout/crash)', sans);
    }
    sw.stop();

    // Time-control forfeit (real clock mode only).
    if (movetimeMs == null) {
      final remaining = clock - sw.elapsedMilliseconds;
      if (remaining < -200) {
        return _GameResult(whiteToMove ? 0.0 : 1.0,
            '${engine.name} forfeits on time', sans);
      }
      final updated = remaining + (incMs ?? 0);
      if (whiteToMove) {
        wClock = updated;
      } else {
        bClock = updated;
      }
    }

    final san = _applyUci(board, best);
    if (san == null) {
      return _GameResult(whiteToMove ? 0.0 : 1.0,
          '${engine.name} played illegal move "$best"', sans);
    }
    moves.add(best);
    sans.add(san);
  }

  if (board.in_checkmate) {
    final whiteMated = board.turn == ch.Chess.WHITE;
    return _GameResult(whiteMated ? 0.0 : 1.0, 'checkmate', sans);
  }
  final reason = board.in_stalemate
      ? 'stalemate'
      : board.insufficient_material
          ? 'insufficient material'
          : board.in_threefold_repetition
              ? 'threefold repetition'
              : 'fifty-move / draw';
  return _GameResult(0.5, reason, sans);
}

/// Applies [uci] to [board], returning the move's SAN or null if illegal.
String? _applyUci(ch.Chess board, String uci) {
  if (uci.length < 4) return null;
  final from = uci.substring(0, 2);
  final to = uci.substring(2, 4);
  final promo = uci.length > 4 ? uci[4].toLowerCase() : null;
  // chess 0.8.1's verbose map omits promotion; match on the Move object.
  for (final mv in board.moves({'asObjects': true}) as List) {
    final m = mv as ch.Move;
    if (m.fromAlgebraic == from &&
        m.toAlgebraic == to &&
        m.promotion?.name == promo) {
      final san = board.move_to_san(m);
      board.move(m);
      return san;
    }
  }
  return null;
}

void _addOpt(Map<String, String> opts, String kv) {
  final i = kv.indexOf('=');
  if (i < 0) {
    stderr.writeln('bad option (want K=V): $kv');
    exit(64);
  }
  opts[kv.substring(0, i)] = kv.substring(i + 1);
}

List<int> _parseTc(String tc) {
  // "base+inc" in seconds, e.g. "10+0.1" or "60".
  final plus = tc.split('+');
  final base = (double.parse(plus[0]) * 1000).round();
  final inc = plus.length > 1 ? (double.parse(plus[1]) * 1000).round() : 0;
  return [base, inc];
}

List<String> _loadOpenings(String file) {
  final out = <String>[];
  for (final line in File(file).readAsLinesSync()) {
    final t = line.trim();
    if (t.isEmpty || t.startsWith('#')) continue;
    final parts = t.split(RegExp(r'\s+'));
    if (parts.length < 4) continue;
    var fen = parts.sublist(0, 4).join(' ');
    if (!RegExp(r'\s\d+\s\d+$').hasMatch(fen)) fen = '$fen 0 1';
    out.add(fen);
  }
  return out.isEmpty ? [_startFen] : out;
}

String _deriveName(String cmd, int n) {
  final first =
      cmd.split(RegExp(r'\s+')).last.replaceAll(RegExp(r'\.[a-z]+$'), '');
  final base = first.split(RegExp(r'[\\/]')).last;
  return base.isEmpty ? 'engine$n' : base;
}

String _pct(int w, int d, int l) {
  final n = w + d + l;
  if (n == 0) return '0.0';
  return (100 * (w + d / 2) / n).toStringAsFixed(1);
}

String _pgn(String white, String black, String result, String fen,
    List<String> sans, int round) {
  final sb = StringBuffer();
  sb.writeln('[Event "match"]');
  sb.writeln('[Site "local"]');
  sb.writeln('[Round "$round"]');
  sb.writeln('[White "$white"]');
  sb.writeln('[Black "$black"]');
  sb.writeln('[Result "$result"]');
  if (fen != _startFen) {
    sb.writeln('[SetUp "1"]');
    sb.writeln('[FEN "$fen"]');
  }
  sb.writeln();
  final line = StringBuffer();
  for (var i = 0; i < sans.length; i++) {
    if (i.isEven) line.write('${i ~/ 2 + 1}. ');
    line.write('${sans[i]} ');
  }
  line.write(result);
  sb.writeln(line.toString());
  sb.writeln();
  return sb.toString();
}
