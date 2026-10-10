// Validates the HalfKP NNUE port: parse, startpos sanity, material imbalance,
// and color-mirror symmetry (eval must be invariant under flip+color-swap).
//   dart run bin/nnue_test.dart [path.nnue]
import 'package:chess/chess.dart' as ch;
import 'package:chess_moves/nnue.dart';

const _net = 'models/nnue/nn-62ef826d1a6d.nnue';

int _eval(Nnue n, String fen) => n.evaluate(ch.Chess.fromFEN(fen));

/// Mirrors a FEN by 180° rotation + color swap (the symmetry HalfKP's `s ^ 63`
/// orientation respects): expand to 64 chars, reverse, swap colors, flip stm.
String _mirror(String fen) {
  final parts = fen.split(' ');
  final buf = StringBuffer();
  for (final c in parts[0].replaceAll('/', '').split('')) {
    if (RegExp(r'\d').hasMatch(c)) {
      buf.write('.' * int.parse(c));
    } else {
      buf.write(c);
    }
  }
  final rotated = buf.toString().split('').reversed.map((c) {
    if (RegExp(r'[A-Z]').hasMatch(c)) return c.toLowerCase();
    if (RegExp(r'[a-z]').hasMatch(c)) return c.toUpperCase();
    return c;
  }).toList();
  final ranks = <String>[];
  for (var r = 0; r < 8; r++) {
    final sb = StringBuffer();
    var empty = 0;
    for (var f = 0; f < 8; f++) {
      final c = rotated[r * 8 + f];
      if (c == '.') {
        empty++;
      } else {
        if (empty > 0) {
          sb.write(empty);
          empty = 0;
        }
        sb.write(c);
      }
    }
    if (empty > 0) sb.write(empty);
    ranks.add(sb.toString());
  }
  final stm = parts[1] == 'w' ? 'b' : 'w';
  return '${ranks.join('/')} $stm - - 0 1';
}

void main(List<String> args) {
  final path = args.isNotEmpty ? args[0] : _net;
  final n = Nnue.load(path);
  print('parsed OK: $path');

  const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const startB = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1';
  const wUpQueen = 'rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const wDownRook = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN1 w KQkq - 0 1';

  print('startpos (w):     ${_eval(n, start)} cp');
  print('startpos (b):     ${_eval(n, startB)} cp');
  print('white up a queen: ${_eval(n, wUpQueen)} cp');
  print('white down a rook:${_eval(n, wDownRook)} cp');

  // Symmetry: eval(pos) must equal eval(mirror(pos)).
  var symOk = true;
  for (final fen in [start, wUpQueen, wDownRook,
    'r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 0 1']) {
    final a = _eval(n, fen);
    final b = _eval(n, _mirror(fen));
    final ok = a == b;
    symOk = symOk && ok;
    print('symmetry ${ok ? 'OK ' : 'FAIL'}: $a vs $b  [$fen]');
  }

  print(symOk ? '\nALL SYMMETRY CHECKS PASSED' : '\nSYMMETRY FAILED');
}
