// Extracts UCI move sequences from downloaded lc0 V6 training tarballs.
//
// Each .tar bundles many gzip-compressed per-game chunks; each chunk is a
// stream of 8356-byte V6 records (one per position). We read the played move
// of every record and emit one game per line (space-separated UCI) to a text
// file the trainer can consume.
//
// Usage:
//   dart run bin/parse_lc0.dart [--max-games N] [--min-moves M]
//                               [--out data/games.txt] [tar files...]
// With no tar files given, every *.tar under training_data/ is processed.
import 'dart:io';
import 'dart:typed_data';

import 'package:chess_moves/lc0_v6.dart';

void main(List<String> args) async {
  var maxGames = 30000;
  var minMoves = 6;
  var outPath = 'data/games.txt';
  final tarArgs = <String>[];

  for (var i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--max-games':
        maxGames = int.parse(args[++i]);
      case '--min-moves':
        minMoves = int.parse(args[++i]);
      case '--out':
        outPath = args[++i];
      default:
        tarArgs.add(args[i]);
    }
  }

  final tarFiles = tarArgs.isNotEmpty
      ? tarArgs
      : (Directory('training_data').existsSync()
          ? (Directory('training_data')
                  .listSync()
                  .whereType<File>()
                  .where((f) => f.path.toLowerCase().endsWith('.tar'))
                  .map((f) => f.path)
                  .toList()
                ..sort())
          : <String>[]);

  if (tarFiles.isEmpty) {
    stderr.writeln('No .tar files found. Pass paths or populate training_data/.');
    exit(64);
  }

  Directory(File(outPath).parent.path).createSync(recursive: true);
  final out = File(outPath).openWrite();

  var games = 0;
  var positions = 0;
  final sw = Stopwatch()..start();

  outer:
  for (final tarPath in tarFiles) {
    stdout.writeln('Parsing ${_short(tarPath)} ...');
    final raf = File(tarPath).openSync();
    try {
      final header = Uint8List(512);
      while (true) {
        if (!_readFull(raf, header, 512)) break; // short read / EOF
        if (_isZeroBlock(header)) break; // end-of-archive marker
        final name = _cstr(header, 0, 100);
        final size = _octal(header, 124, 12);
        final typeFlag = header[156];
        final dataLen = ((size + 511) ~/ 512) * 512; // padded to 512

        final isRegular = typeFlag == 0x30 || typeFlag == 0; // '0' or NUL
        if (isRegular && size > 0 && name.endsWith('.gz')) {
          final padded = Uint8List(dataLen);
          if (!_readFull(raf, padded, dataLen)) break;
          final gz = Uint8List.sublistView(padded, 0, size);
          List<int> raw;
          try {
            raw = gzip.decode(gz);
          } catch (_) {
            continue; // corrupt entry: skip
          }
          final moves = movesFromChunk(
              raw is Uint8List ? raw : Uint8List.fromList(raw));
          if (moves.length >= minMoves) {
            out.writeln(moves.join(' '));
            games++;
            positions += moves.length;
            if (games % 1000 == 0) {
              stdout.writeln('  $games games, $positions positions '
                  '(${sw.elapsedMilliseconds} ms)');
            }
            if (games >= maxGames) break outer;
          }
        } else if (dataLen > 0) {
          raf.setPositionSync(raf.positionSync() + dataLen); // skip entry
        }
      }
    } finally {
      raf.closeSync();
    }
  }

  await out.flush();
  await out.close();
  sw.stop();
  stdout.writeln('\nWrote $games games ($positions positions) to $outPath '
      'in ${sw.elapsedMilliseconds} ms');
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

bool _isZeroBlock(Uint8List b) {
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
    if (c == 0 || c == 0x20) continue; // NUL / space padding
    sb.writeCharCode(c);
  }
  final s = sb.toString();
  return s.isEmpty ? 0 : int.parse(s, radix: 8);
}

String _short(String p) => p.replaceAll('\\', '/').split('/').last;
