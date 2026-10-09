// Milestone 1 of attention-body support: parse an lc0 ATTENTIONBODY net's
// Weights into named tensors and resolve/validate every shape. No forward pass
// yet — this proves the proto field map and dimensions before porting the
// parser + forward into dart_pytorch.
//
//   dart run bin/lc0_attn_probe.dart models/lc0/t1-256x10-distilled.pb.gz
import 'dart:io';
import 'dart:typed_data';

void main(List<String> args) {
  if (args.isEmpty) {
    stderr.writeln('usage: dart run bin/lc0_attn_probe.dart <net.pb.gz>');
    exit(64);
  }
  final d = Uint8List.fromList(gzip.decode(File(args[0]).readAsBytesSync()));

  // Net.weights = field 10.
  final w = _child(d, 0, d.length, 10)!;
  final W = w; // Weights message range

  int params(int field, [int sub = -1]) {
    final f = sub < 0 ? _child2(d, W, field) : _childOf(d, _child2(d, W, field)!, sub);
    if (f == null) return 0;
    return _layerParams(d, f);
  }

  final heads = _varintField(d, W, 28) ?? 0;
  final embB = params(26); // ip_emb_b -> embedding dim
  final embDim = embB;
  final embW = params(25); // ip_emb_w
  final inEmb = embDim == 0 ? 0 : embW ~/ embDim;

  // Encoder layers (#27, repeated).
  final encoders = _children(d, W, 27);
  final numLayers = encoders.length;

  // Inspect first encoder layer's structure.
  String encInfo = '(none)';
  if (encoders.isNotEmpty) {
    final e = encoders.first;
    final mha = _childOf(d, e, 1)!;
    final qw = _layerParams(d, _childOf(d, mha, 1)!);
    final denseW = _layerParams(d, _childOf(d, mha, 7)!);
    final smolgen = _childOf(d, mha, 9);
    final ffn = _childOf(d, e, 4)!;
    final ffn1 = _layerParams(d, _childOf(d, ffn, 1)!);
    final ffn2 = _layerParams(d, _childOf(d, ffn, 3)!);
    final dff = embDim == 0 ? 0 : ffn1 ~/ embDim;
    final smParts = smolgen == null
        ? 'no-smolgen'
        : 'smolgen{compress=${_layerParams(d, _childOf(d, smolgen, 1)!)}, '
            'dense1=${_layerParams(d, _childOf(d, smolgen, 2)!)}, '
            'dense2=${_layerParams(d, _childOf(d, smolgen, 6)!)}}';
    encInfo = 'q_w=$qw dense_w=$denseW ffn1=$ffn1 ffn2=$ffn2 dff=$dff  $smParts';
  }

  final smolgenW = params(35);
  final smolgenB = params(36);

  print('=== ${args[0]} ===');
  print('embedding: ip_emb_w=$embW  -> in=$inEmb  embDim=$embDim');
  print('gating: ip_mult_gate=${params(33)}  ip_add_gate=${params(34)}');
  print('encoder: layers=$numLayers heads=$heads depth=${heads == 0 ? 0 : embDim ~/ heads}');
  print('  layer0: $encInfo');
  print('global smolgen: smolgen_w=$smolgenW  smolgen_b=$smolgenB'
      '  (gen=${smolgenW == 0 ? 0 : smolgenW ~/ 4096})');
  print('policy(attn): ip_pol_w=${params(4)} ip2_pol_w(wq)=${params(17)} '
      'ip3_pol_w(wk)=${params(19)} ip4_pol_w(ppo)=${params(22)}');
  print('value(attn): ip_val_w=${params(29)}(emb) ip1_val_w=${params(7)} '
      'ip2_val_w=${params(9)} wdl=${params(10)}');
  print('mlh(attn): ip_mov_w=${params(31)}(emb) ip1_mov_w=${params(13)} '
      'ip2_mov_w=${params(15)}');

  // Consistency checks.
  final depth = heads == 0 ? 0 : embDim ~/ heads;
  final ok = embDim == 256 &&
      heads == 8 &&
      depth * heads == embDim &&
      inEmb == 176 &&
      numLayers == 10;
  print(ok ? '\nSHAPES-OK' : '\nSHAPES-MISMATCH');
}

// ---- minimal proto helpers ----

int? _varintField(Uint8List d, _Range m, int field) {
  final f = _child2(d, m, field);
  if (f == null || f.wire != 0) return null;
  return _varint(d, f.start).value;
}

int _layerParams(Uint8List d, _Range layer) {
  // Layer.params = field 3 (bytes); LINEAR16/FLOAT16 => 2 bytes each.
  final p = _childOf(d, layer, 3);
  if (p == null) return 0;
  return (p.end - p.start) ~/ 2;
}

_Range? _child(Uint8List d, int start, int end, int field) =>
    _childOf(d, _Range(start, end, 2), field);

_Range? _child2(Uint8List d, _Range m, int field) => _childOf(d, m, field);

_Range? _childOf(Uint8List d, _Range m, int field) {
  var i = m.start;
  while (i < m.end) {
    final f = _readField(d, i);
    if (f == null) return null;
    if (f.number == field) return _Range(f.cs, f.ce, f.wire);
    i = f.ce;
  }
  return null;
}

List<_Range> _children(Uint8List d, _Range m, int field) {
  final out = <_Range>[];
  var i = m.start;
  while (i < m.end) {
    final f = _readField(d, i);
    if (f == null) break;
    if (f.number == field) out.add(_Range(f.cs, f.ce, f.wire));
    i = f.ce;
  }
  return out;
}

class _Range {
  final int start, end, wire;
  _Range(this.start, this.end, [this.wire = 2]);
}

class _F {
  final int number, wire, cs, ce;
  _F(this.number, this.wire, this.cs, this.ce);
}

_F? _readField(Uint8List d, int i) {
  final tag = _varint(d, i);
  final number = tag.value >> 3;
  final wire = tag.value & 7;
  var p = tag.next;
  switch (wire) {
    case 0:
      return _F(number, wire, p, _varint(d, p).next);
    case 1:
      return _F(number, wire, p, p + 8);
    case 2:
      final len = _varint(d, p);
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

_V _varint(Uint8List d, int i) {
  var shift = 0, result = 0;
  while (i < d.length) {
    final b = d[i++];
    result |= (b & 0x7f) << shift;
    if ((b & 0x80) == 0) return _V(result, i);
    shift += 7;
  }
  return _V(result, i);
}
