// Hybrid UCI engine: the DeepMind searchless-chess net chooses strategically,
// while a shallow NNUE alpha-beta search acts as a tactical veto — overriding
// the net only when its preferred move loses material/gets mated versus the
// alpha-beta best. Pure Dart, no Python.
//   dart run bin/searchless_hybrid_uci.dart
import 'dart:convert';
import 'dart:io';

import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/alphabeta.dart';
import 'package:chess_moves/nnue.dart';
import 'package:chess_moves/searchless_pool.dart';

const _startFen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

String? _childFen(ch.Chess board, String uci) {
  final moves = (board.moves({'asObjects': true}) as List).cast<ch.Move>();
  for (final m in moves) {
    final mu = m.fromAlgebraic + m.toAlgebraic + (m.promotion?.name ?? '');
    if (mu == uci) {
      board.make_move(m);
      final fen = board.generate_fen();
      board.undo_move();
      return fen;
    }
  }
  return null;
}

Future<void> main(List<String> args) async {
  var npzPath = 'models/searchless/params.npz';
  var nnuePath = 'models/nnue/nn-62ef826d1a6d.nnue';
  var workers = (Platform.numberOfProcessors - 1).clamp(1, 15);
  var abMs = 400; // tactical-veto search budget per probe
  var marginCp = 200; // override only if the net move loses >= this vs AB best
  var vetoFloorCp = -120; // ...and only if the net move leaves us this bad (abs)
  SearchlessPool? pool;
  AlphaBeta? ab;
  var board = ch.Chess.fromFEN(_startFen);

  Future<void> ensure() async {
    pool ??= await SearchlessPool.start(npzPath, workers);
    ab ??= AlphaBeta(
        nnue: File(nnuePath).existsSync() ? Nnue.load(nnuePath) : null);
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

  Future<String> pickMove() async {
    final fen = board.generate_fen();
    final q = await pool!.evaluate(fen);
    if (q.isEmpty) return '0000';
    // Searchless strategic choice.
    String slBest = q.keys.first;
    var slWin = double.negativeInfinity;
    q.forEach((uci, win) {
      if (win > slWin) {
        slWin = win;
        slBest = uci;
      }
    });
    // Alpha-beta tactical reference.
    final abRes = ab!.search(fen, time: Duration(milliseconds: abMs));
    final abBest = abRes.bestUci;
    if (abBest == null || abBest == slBest) {
      stdout.writeln('info string pick=$slBest net=${(slWin * 100).toStringAsFixed(1)}% (ab agrees)');
      return slBest;
    }
    // Tactically score the net's move: value = -(score for the opponent after it).
    final child = _childFen(ch.Chess.fromFEN(fen), slBest);
    var slCp = 0;
    if (child != null) {
      final childRes = ab!.search(child, time: Duration(milliseconds: abMs));
      slCp = -childRes.scoreCp;
    }
    final gap = abRes.scoreCp - slCp;
    // Veto only genuine blunders: the net move must both leave us clearly worse
    // in absolute terms AND be clearly beaten by the AB's alternative.
    if (slCp <= vetoFloorCp && gap >= marginCp) {
      stdout.writeln('info string pick=$abBest (veto net=$slBest; abBest=${abRes.scoreCp}cp netMove=${slCp}cp gap=${gap}cp)');
      return abBest;
    }
    stdout.writeln('info string pick=$slBest net=${(slWin * 100).toStringAsFixed(1)}% (ab=$abBest netMove=${slCp}cp gap=${gap}cp kept)');
    return slBest;
  }

  await for (final line
      in stdin.transform(utf8.decoder).transform(const LineSplitter())) {
    final s = line.trim();
    if (s == 'uci') {
      stdout.writeln('id name SearchlessHybrid');
      stdout.writeln('id author chess_moves');
      stdout.writeln('option name WeightsFile type string default $npzPath');
      stdout.writeln('option name AbNnue type string default $nnuePath');
      stdout.writeln('option name Workers type spin default $workers min 1 max 64');
      stdout.writeln('option name AbMs type spin default $abMs min 10 max 5000');
      stdout.writeln('option name MarginCp type spin default $marginCp min 0 max 2000');
      stdout.writeln('option name VetoFloorCp type spin default $vetoFloorCp min -2000 max 0');
      stdout.writeln('uciok');
    } else if (s == 'isready') {
      await ensure();
      stdout.writeln('readyok');
    } else if (s.startsWith('setoption')) {
      final m = RegExp(r'name\s+(.+?)\s+value\s+(.+)$').firstMatch(s);
      if (m != null) {
        final name = m.group(1)!.trim().toLowerCase();
        final value = m.group(2)!.trim();
        switch (name) {
          case 'weightsfile':
            npzPath = value;
            pool = null;
            break;
          case 'abnnue':
            nnuePath = value;
            ab = null;
            break;
          case 'workers':
            workers = int.tryParse(value) ?? workers;
            pool = null;
            break;
          case 'abms':
            abMs = int.tryParse(value) ?? abMs;
            break;
          case 'margincp':
            marginCp = int.tryParse(value) ?? marginCp;
            break;
          case 'vetofloorcp':
            vetoFloorCp = int.tryParse(value) ?? vetoFloorCp;
            break;
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
      stdout.writeln('bestmove ${await pickMove()}');
    } else if (s == 'quit') {
      break;
    }
  }
}
