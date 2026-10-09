// Generic lc0 weights.pb.gz proto-structure dumper (no protobuf runtime).
// Prints the field-number tree with sizes so we can see exactly which
// architecture fields a net populates (classical conv vs attention body).
//
//   dart run bin/lc0_dump.dart models/lc0/<net>.pb.gz
import 'dart:io';
import 'dart:typed_data';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('usage: dart run bin/lc0_dump.dart <net.pb.gz>');
    exit(64);
  }
  final bytes = Uint8List.fromList(gzip.decode(File(args[0]).readAsBytesSync()));
  stdout.writeln('net: ${args[0]}  (${bytes.length} bytes decoded)');
  _dump(bytes, 0, bytes.length, 0);
}

// Walk a protobuf message [start,end) and print its fields. Length-delimited
// fields are recursed into when they parse cleanly as a sub-message; otherwise
// they're shown as raw bytes (e.g. quantized Layer params).
void _dump(Uint8List d, int start, int end, int depth) {
  if (depth > 7) return;
  final indent = '  ' * depth;
  // Group repeated fields (same number) to keep output compact.
  final seen = <int, List<_FieldInfo>>{};
  var i = start;
  while (i < end) {
    final f = _readField(d, i);
    if (f == null) {
      stdout.writeln('$indent<unparseable at $i>');
      return;
    }
    seen.putIfAbsent(f.number, () => []).add(f);
    i = f.contentEnd;
  }
  final nums = seen.keys.toList()..sort();
  for (final n in nums) {
    final fs = seen[n]!;
    final wt = fs.first.wireType;
    if (wt == 2) {
      final totalLen =
          fs.fold<int>(0, (a, f) => a + (f.contentEnd - f.contentStart));
      final msgLike = _looksLikeMessage(d, fs.first.contentStart, fs.first.contentEnd);
      final kind = msgLike ? 'msg' : 'bytes';
      stdout.writeln('$indent#$n x${fs.length} [$kind] len=$totalLen');
      if (msgLike) {
        _dump(d, fs.first.contentStart, fs.first.contentEnd, depth + 1);
      }
    } else if (wt == 0) {
      final vals = fs.map((f) => _varint(d, f.contentStart)?.value).toList();
      stdout.writeln('$indent#$n x${fs.length} [varint] = $vals');
    } else if (wt == 5) {
      final vals = fs.map((f) {
        final bd = ByteData.sublistView(d, f.contentStart, f.contentStart + 4);
        return bd.getFloat32(0, Endian.little).toStringAsFixed(4);
      }).toList();
      stdout.writeln('$indent#$n x${fs.length} [f32] = $vals');
    } else {
      stdout.writeln('$indent#$n x${fs.length} [wt$wt]');
    }
  }
}

bool _looksLikeMessage(Uint8List d, int start, int end) {
  if (end - start < 2) return false;
  var i = start;
  var fields = 0;
  while (i < end) {
    final f = _readField(d, i);
    if (f == null) return false;
    if (f.contentEnd > end) return false;
    i = f.contentEnd;
    if (++fields > 64) break;
  }
  return i == end && fields > 0;
}

class _FieldInfo {
  final int number;
  final int wireType;
  final int contentStart;
  final int contentEnd;
  _FieldInfo(this.number, this.wireType, this.contentStart, this.contentEnd);
}

_FieldInfo? _readField(Uint8List d, int i) {
  final tag = _varint(d, i);
  if (tag == null) return null;
  final number = tag.value >> 3;
  final wireType = tag.value & 7;
  var p = tag.next;
  switch (wireType) {
    case 0:
      final v = _varint(d, p);
      if (v == null) return null;
      return _FieldInfo(number, wireType, p, v.next);
    case 1:
      if (p + 8 > d.length) return null;
      return _FieldInfo(number, wireType, p, p + 8);
    case 2:
      final len = _varint(d, p);
      if (len == null) return null;
      final cs = len.next;
      final ce = cs + len.value;
      if (ce > d.length) return null;
      return _FieldInfo(number, wireType, cs, ce);
    case 5:
      if (p + 4 > d.length) return null;
      return _FieldInfo(number, wireType, p, p + 4);
    default:
      return null;
  }
}

class _Varint {
  final int value;
  final int next;
  _Varint(this.value, this.next);
}

_Varint? _varint(Uint8List d, int i) {
  var shift = 0;
  var result = 0;
  while (i < d.length) {
    final b = d[i++];
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) == 0) return _Varint(result, i);
    shift += 7;
    if (shift > 63) return null;
  }
  return null;
}
