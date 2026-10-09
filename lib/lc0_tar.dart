import 'dart:io';
import 'dart:typed_data';

/// Reads up to [maxGames] decompressed game chunks (each a stream of V6 records)
/// from an lc0 training `.tar`. Shared by the move-sequence parser and the
/// position-based (lc0-parity) tooling.
List<Uint8List> readGameChunks(String tarPath,
    {int maxGames = 1, int minBytes = 8356}) {
  final chunks = <Uint8List>[];
  final raf = File(tarPath).openSync();
  try {
    final header = Uint8List(512);
    while (chunks.length < maxGames) {
      if (!_readFull(raf, header, 512)) break;
      if (_isZero(header)) break;
      final name = _cstr(header, 0, 100);
      final size = _octal(header, 124, 12);
      final type = header[156];
      final dataLen = ((size + 511) ~/ 512) * 512;
      final isRegular = type == 0x30 || type == 0;
      if (isRegular && size >= minBytes && name.endsWith('.gz')) {
        final padded = Uint8List(dataLen);
        if (!_readFull(raf, padded, dataLen)) break;
        try {
          final raw = gzip.decode(Uint8List.sublistView(padded, 0, size));
          chunks.add(raw is Uint8List ? raw : Uint8List.fromList(raw));
        } catch (_) {
          // skip corrupt entry
        }
      } else if (dataLen > 0) {
        raf.setPositionSync(raf.positionSync() + dataLen);
      }
    }
  } finally {
    raf.closeSync();
  }
  return chunks;
}

bool _readFull(RandomAccessFile raf, Uint8List buf, int len) {
  var got = 0;
  while (got < len) {
    final n = raf.readIntoSync(buf, got, len);
    if (n <= 0) return false;
    got += n;
  }
  return true;
}

bool _isZero(Uint8List b) {
  for (final x in b) {
    if (x != 0) return false;
  }
  return true;
}

String _cstr(Uint8List h, int off, int len) {
  var end = off;
  while (end < off + len && h[end] != 0) {
    end++;
  }
  return String.fromCharCodes(h.sublist(off, end));
}

int _octal(Uint8List h, int off, int len) {
  final sb = StringBuffer();
  for (var i = off; i < off + len; i++) {
    final c = h[i];
    if (c == 0 || c == 0x20) continue;
    sb.writeCharCode(c);
  }
  final s = sb.toString();
  return s.isEmpty ? 0 : int.parse(s, radix: 8);
}
