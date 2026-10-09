import 'conv_policy_map.dart';
import 'lc0_v6.dart' show toAbsoluteUci;
import 'policy_index.dart';

// UCI (mover POV) -> classical 1858 move index.
final Map<String, int> _uciToIdx = {
  for (var i = 0; i < kPolicyIndex.length; i++) kPolicyIndex[i]: i
};

// 1858 move index -> conv output index (inverse of kConvPolicyMap).
final List<int> _idxToConv = _buildIdxToConv();

List<int> _buildIdxToConv() {
  final out = List<int>.filled(kPolicyIndex.length, -1);
  for (var i = 0; i < kConvPolicyMap.length; i++) {
    final m = kConvPolicyMap[i];
    if (m >= 0 && m < out.length) out[m] = i;
  }
  return out;
}

/// Conv-policy output index for an absolute-board UCI move given the side to
/// move (0 = White, 1 = Black), or null if the move isn't representable (e.g. a
/// knight underpromotion, which lc0's classical policy doesn't enumerate).
///
/// lc0 encodes moves from the mover's POV; `toAbsoluteUci` is an involution, so
/// applying it to an absolute move with the same side-to-move recovers the
/// mover-POV move used by the index tables.
int? convIndexForUci(String absoluteUci, int sideToMove) {
  final moverPov = toAbsoluteUci(absoluteUci, sideToMove);
  final idx = _uciToIdx[moverPov];
  if (idx == null) return null;
  final conv = _idxToConv[idx];
  return conv >= 0 ? conv : null;
}
