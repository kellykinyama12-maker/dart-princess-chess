// Minimal ONNX graph dumper: prints the node op_type + outputs in order, so we
// can compare lc0's exported attention graph against our Dart forward.
//   dart run bin/onnx_dump.dart C:\projects\lc0ref\net.onnx [maxNodes]
import 'dart:io';
import 'dart:typed_data';

void main(List<String> args) {
  final path = args.isNotEmpty ? args[0] : r'C:\projects\lc0ref\net.onnx';
  final max = args.length > 1 ? int.parse(args[1]) : 80;
  final d = File(path).readAsBytesSync();
  // ModelProto.graph = field 7 (message).
  final graph = _field(d, _R(0, d.length), 7);
  if (graph == null) {
    stderr.writeln('no graph (field 7)');
    exit(1);
  }
  // GraphProto.node = field 1 (repeated NodeProto).
  final nodes = _fields(d, graph, 1);
  stdout.writeln('nodes: ${nodes.length}');
  var i = 0;
  for (final n in nodes) {
    if (i >= max) break;
    final opType = _str(d, n, 4);
    final name = _str(d, n, 3);
    final outs = _strs(d, n, 2);
    final ins = _strs(d, n, 1);
    final inShort = ins.map(_short).take(3).join(',');
    stdout.writeln('${i.toString().padLeft(3)}  ${opType.padRight(14)} '
        '-> ${outs.map(_short).join(',')}   [in: $inShort]'
        '${name.isNotEmpty ? '  {$name}' : ''}');
    i++;
  }
}

String _short(String s) {
  // keep the meaningful tail of a tensor name
  final parts = s.split('/');
  if (parts.length <= 2) return s;
  return '.../' + parts.sublist(parts.length - 2).join('/');
}

// ---- proto helpers ----
class _R {
  final int start, end;
  const _R(this.start, this.end);
}

_R? _field(Uint8List d, _R m, int field) {
  var i = m.start;
  while (i < m.end) {
    final f = _read(d, i);
    if (f == null) return null;
    if (f.number == field) return _R(f.cs, f.ce);
    i = f.ce;
  }
  return null;
}

List<_R> _fields(Uint8List d, _R m, int field) {
  final out = <_R>[];
  var i = m.start;
  while (i < m.end) {
    final f = _read(d, i);
    if (f == null) break;
    if (f.number == field) out.add(_R(f.cs, f.ce));
    i = f.ce;
  }
  return out;
}

String _str(Uint8List d, _R m, int field) {
  final r = _field(d, m, field);
  if (r == null) return '';
  return String.fromCharCodes(d.sublist(r.start, r.end));
}

List<String> _strs(Uint8List d, _R m, int field) =>
    [for (final r in _fields(d, m, field)) String.fromCharCodes(d.sublist(r.start, r.end))];

class _F {
  final int number, wire, cs, ce;
  _F(this.number, this.wire, this.cs, this.ce);
}

_F? _read(Uint8List d, int i) {
  final tag = _varint(d, i);
  if (tag == null) return null;
  final number = tag.value >> 3;
  final wire = tag.value & 7;
  final p = tag.next;
  switch (wire) {
    case 0:
      final v = _varint(d, p);
      return v == null ? null : _F(number, wire, p, v.next);
    case 1:
      return _F(number, wire, p, p + 8);
    case 2:
      final len = _varint(d, p);
      if (len == null) return null;
      return _F(number, wire, len.next, len.next + len.value);
    case 5:
      return _F(number, wire, p, p + 4);
    default:
      return null;
  }
}

class _V {
  final int value, next;
  _V(this.value, this.next);
}

_V? _varint(Uint8List d, int i) {
  var shift = 0, result = 0;
  while (i < d.length) {
    final b = d[i++];
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) == 0) return _V(result, i);
    shift += 7;
    if (shift > 63) return null;
  }
  return null;
}
