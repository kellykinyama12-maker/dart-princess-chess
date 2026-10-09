import 'dart:convert';
import 'dart:io';

/// Maps UCI move strings to integer token ids and back. Reserves four special
/// tokens at the front so sequences can mark game boundaries and padding.
class MoveVocab {
  final List<String> itos;
  final Map<String, int> stoi;

  MoveVocab(this.itos)
      : stoi = {for (var i = 0; i < itos.length; i++) itos[i]: i};

  static const pad = '<pad>';
  static const bos = '<bos>';
  static const eos = '<eos>';
  static const unk = '<unk>';

  int get size => itos.length;
  int get padId => stoi[pad]!;
  int get bosId => stoi[bos]!;
  int get eosId => stoi[eos]!;
  int get unkId => stoi[unk]!;

  bool isSpecial(int id) => id <= unkId; // first four ids are specials

  int encodeToken(String move) => stoi[move] ?? unkId;
  String decodeId(int id) => (id >= 0 && id < itos.length) ? itos[id] : unk;

  /// Builds a vocabulary from games, sorting moves by descending frequency so
  /// common moves get small, stable ids.
  static MoveVocab build(Iterable<List<String>> games) {
    final counts = <String, int>{};
    for (final game in games) {
      for (final move in game) {
        counts[move] = (counts[move] ?? 0) + 1;
      }
    }
    final moves = counts.keys.toList()
      ..sort((a, b) {
        final c = counts[b]!.compareTo(counts[a]!);
        return c != 0 ? c : a.compareTo(b);
      });
    return MoveVocab(<String>[pad, bos, eos, unk, ...moves]);
  }

  void saveFile(String path) {
    File(path).writeAsStringSync(jsonEncode({'itos': itos}));
  }

  static MoveVocab loadFile(String path) {
    final data = jsonDecode(File(path).readAsStringSync()) as Map<String, dynamic>;
    return MoveVocab((data['itos'] as List).cast<String>());
  }
}
