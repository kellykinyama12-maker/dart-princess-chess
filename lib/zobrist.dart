import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

/// Zobrist position hashing for the engine. Supports a full hash from a FEN
/// (used at the root) and an incremental update across a single move
/// (make/unmake style) so child node keys are derived from the parent's key
/// with a handful of XORs instead of re-scanning the board.
///
/// The key identity matches what the (history-less) net eval depends on:
/// placement, side-to-move, castling, en-passant file and the rule-50 counter.
class Zobrist {
  final List<int> _piece = List<int>.filled(12 * 64, 0);
  final List<int> _castle = List<int>.filled(4, 0); // WK, WQ, BK, BQ
  final List<int> _ep = List<int>.filled(8, 0); // by file
  final List<int> _rule50 = List<int>.filled(101, 0);
  late final int _black;

  Zobrist({int seed = 0x1a2b3c4d}) {
    final r = math.Random(seed);
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

  static int _sq64(int sq0x88) => (sq0x88 >> 4) * 8 + (sq0x88 & 7);

  int _pieceKey(ch.Color color, ch.PieceType type, int sq64) =>
      _piece[((color == ch.Chess.WHITE ? 0 : 6) + type.shift) * 64 + sq64];

  /// Full hash from a FEN string.
  int ofFen(String fen) {
    final parts = fen.split(' ');
    final placement = parts[0];
    var h = 0;
    var sq = 0;
    for (var i = 0; i < placement.length; i++) {
      final c = placement.codeUnitAt(i);
      if (c == 0x2F) continue; // '/'
      if (c >= 0x31 && c <= 0x38) {
        sq += c - 0x30;
        continue;
      }
      final idx = _pieceIndexFromChar(c);
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

  /// Incrementally updates [key] for a move: [before] is the board state prior
  /// to the move, [m] the move, [after] the board state once it has been made.
  int applyMove(int key, ch.Chess before, ch.Move m, ch.Chess after) {
    var h = key;
    final us = m.color;
    final them = us == ch.Chess.WHITE ? ch.Chess.BLACK : ch.Chess.WHITE;
    final fromSq = _sq64(m.from);
    final toSq = _sq64(m.to);

    // Lift the moving piece off `from`; drop it (or its promotion) on `to`.
    h ^= _pieceKey(us, m.piece, fromSq);
    h ^= _pieceKey(us, m.promotion ?? m.piece, toSq);

    // Captured piece.
    final captured = m.captured;
    if (captured != null) {
      int capSq;
      if ((m.flags & ch.Chess.BITS_EP_CAPTURE) != 0) {
        final capRaw = us == ch.Chess.WHITE ? m.to + 16 : m.to - 16;
        capSq = _sq64(capRaw);
      } else {
        capSq = toSq;
      }
      h ^= _pieceKey(them, captured, capSq);
    }

    // Castling rook relocation.
    if ((m.flags & ch.Chess.BITS_KSIDE_CASTLE) != 0) {
      h ^= _pieceKey(us, ch.Chess.ROOK, _sq64(m.to + 1));
      h ^= _pieceKey(us, ch.Chess.ROOK, _sq64(m.to - 1));
    } else if ((m.flags & ch.Chess.BITS_QSIDE_CASTLE) != 0) {
      h ^= _pieceKey(us, ch.Chess.ROOK, _sq64(m.to - 2));
      h ^= _pieceKey(us, ch.Chess.ROOK, _sq64(m.to + 1));
    }

    h ^= _black; // side to move flips every ply.
    h ^= _castleDelta(before.castling, after.castling);
    h ^= _epDelta(before.ep_square, after.ep_square);
    h ^= _rule50[before.half_moves.clamp(0, 100)];
    h ^= _rule50[after.half_moves.clamp(0, 100)];
    return h;
  }

  int _castleDelta(ch.ColorMap<int> before, ch.ColorMap<int> after) {
    var h = 0;
    final bw = before[ch.Chess.WHITE], aw = after[ch.Chess.WHITE];
    final bb = before[ch.Chess.BLACK], ab = after[ch.Chess.BLACK];
    if ((bw & ch.Chess.BITS_KSIDE_CASTLE) != (aw & ch.Chess.BITS_KSIDE_CASTLE)) {
      h ^= _castle[0];
    }
    if ((bw & ch.Chess.BITS_QSIDE_CASTLE) != (aw & ch.Chess.BITS_QSIDE_CASTLE)) {
      h ^= _castle[1];
    }
    if ((bb & ch.Chess.BITS_KSIDE_CASTLE) != (ab & ch.Chess.BITS_KSIDE_CASTLE)) {
      h ^= _castle[2];
    }
    if ((bb & ch.Chess.BITS_QSIDE_CASTLE) != (ab & ch.Chess.BITS_QSIDE_CASTLE)) {
      h ^= _castle[3];
    }
    return h;
  }

  int _epDelta(int? before, int? after) {
    final bf = (before != null && before >= 0) ? (before & 7) : -1;
    final af = (after != null && after >= 0) ? (after & 7) : -1;
    if (bf == af) return 0;
    var h = 0;
    if (bf >= 0) h ^= _ep[bf];
    if (af >= 0) h ^= _ep[af];
    return h;
  }

  static int _pieceIndexFromChar(int c) {
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
