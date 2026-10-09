import 'dart:typed_data';

import 'policy_index.dart';

/// Size of one lc0 V6 training record, in bytes.
/// Layout (little-endian), from LeelaChessZero/lczero-training chunkparser.py:
///   version            uint32  @0
///   input_format       uint32  @4
///   probabilities[1858] float  @8      (7432 bytes)
///   planes[104]        uint64  @7440   (832 bytes)
///   castling us_ooo/us_oo/them_ooo/them_oo  uint8 @8272..8275
///   side_to_move_or_enpassant  uint8   @8276
///   rule50_count       uint8   @8277
///   invariance_info    uint8   @8278   (bit 7 = side to move for input type >=3)
///   dep_result         int8    @8279
///   15 floats (root_q..orig_m)         @8280..8340
///   visits             uint32  @8340
///   played_idx         uint16  @8344   (index into the 1858-move policy array)
///   best_idx           uint16  @8346
///   reserved           8 bytes @8348
const int kV6RecordSize = 8356;

const int _offInputFormat = 4;
const int _offStm = 8276;
const int _offInvariance = 8278;
const int _offPlayedIdx = 8344;

/// Decodes one decompressed chunk (the concatenated V6 records of a single
/// self-play game) into the sequence of moves actually played, as
/// absolute-board UCI strings (e.g. "e2e4", "e7e8q").
List<String> movesFromChunk(Uint8List bytes) {
  final moves = <String>[];
  final count = bytes.length ~/ kV6RecordSize;
  final bd = ByteData.sublistView(bytes);
  for (int r = 0; r < count; r++) {
    final base = r * kV6RecordSize;
    if (bd.getUint32(base, Endian.little) != 6) continue; // only V6
    final inputFormat = bd.getUint32(base + _offInputFormat, Endian.little);
    final playedIdx = bd.getUint16(base + _offPlayedIdx, Endian.little);
    if (playedIdx >= kPolicyIndex.length) continue;
    // Side to move: 0 = White, 1 = Black.
    final stm = inputFormat == 1
        ? bd.getUint8(base + _offStm)
        : (bd.getUint8(base + _offInvariance) >> 7) & 1;
    moves.add(toAbsoluteUci(kPolicyIndex[playedIdx], stm));
  }
  return moves;
}

/// lc0 stores every move from the side-to-move's point of view, with Black's
/// board flipped vertically. Convert back to absolute coordinates by mirroring
/// ranks (file and any promotion suffix are unchanged) when Black is to move.
String toAbsoluteUci(String moverPovMove, int sideToMove) {
  if (sideToMove == 0) return moverPovMove;
  final out = StringBuffer();
  for (int i = 0; i < moverPovMove.length; i++) {
    final c = moverPovMove.codeUnitAt(i);
    if (c >= 0x31 && c <= 0x38) {
      out.writeCharCode(0x39 - (c - 0x30)); // rank d -> 9 - d
    } else {
      out.writeCharCode(c);
    }
  }
  return out.toString();
}
