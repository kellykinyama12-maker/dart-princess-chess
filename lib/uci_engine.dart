import 'dart:async';
import 'dart:convert';
import 'dart:io';

/// A UCI engine running as a child process, driven over stdin/stdout.
/// Minimal cutechess-style controller: handshake, options, per-move search.
class UciEngine {
  final String name;
  final Process _proc;
  final Stream<String> _lines; // broadcast of trimmed stdout lines
  final List<String> _log = []; // recent stderr/stdout for crash diagnostics

  UciEngine._(this.name, this._proc, this._lines);

  /// Launches [command] (e.g. `dart run bin/uci.dart` or `C:\eng.exe`),
  /// performs the UCI handshake and applies [options] (setoption name/value).
  static Future<UciEngine> start(String command,
      {required String name,
      Map<String, String> options = const {},
      String? workingDirectory}) async {
    final parts = _tokenize(command);
    final proc = await Process.start(
      parts.first,
      parts.sublist(1),
      runInShell: true,
      workingDirectory: workingDirectory,
    );
    final lines = proc.stdout
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .map((l) => l.trimRight())
        .asBroadcastStream();
    final engine = UciEngine._(name, proc, lines);
    proc.stderr
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen(engine._remember);
    lines.listen(engine._remember);

    await engine._expect('uci', (l) => l == 'uciok',
        Duration(seconds: 20));
    for (final e in options.entries) {
      engine._send('setoption name ${e.key} value ${e.value}');
    }
    await engine.isReady();
    return engine;
  }

  void _remember(String l) {
    _log.add(l);
    if (_log.length > 40) _log.removeAt(0);
  }

  List<String> get recentLog => List.unmodifiable(_log);

  void _send(String cmd) {
    _proc.stdin.writeln(cmd);
  }

  /// Sends [cmd] and waits for a line satisfying [match] within [timeout].
  Future<String> _expect(
      String cmd, bool Function(String) match, Duration timeout) {
    final fut = _lines.firstWhere(match).timeout(timeout);
    _send(cmd);
    return fut;
  }

  Future<void> isReady() =>
      _expect('isready', (l) => l == 'readyok', Duration(seconds: 20));

  void newGame() => _send('ucinewgame');

  /// Runs one search: sends [position] then [go], returns the bestmove UCI.
  /// Throws [TimeoutException] if no bestmove arrives within [hardLimit].
  Future<String> bestMove(
      String position, String go, Duration hardLimit) async {
    final fut = _lines.firstWhere((l) => l.startsWith('bestmove')).timeout(
        hardLimit,
        onTimeout: () => throw TimeoutException(
            '$name produced no bestmove in ${hardLimit.inMilliseconds}ms'));
    _send(position);
    _send(go);
    final line = await fut;
    final tok = line.split(RegExp(r'\s+'));
    return tok.length >= 2 ? tok[1] : '0000';
  }

  Future<void> quit() async {
    try {
      _send('quit');
      await _proc.exitCode.timeout(Duration(seconds: 3));
    } catch (_) {
      _proc.kill(ProcessSignal.sigkill);
    }
  }

  /// Splits a command line into tokens, honoring double-quoted spans so paths
  /// with spaces survive (e.g. `"C:\Program Files\eng.exe" --foo`).
  static List<String> _tokenize(String s) {
    final out = <String>[];
    final sb = StringBuffer();
    var inQuote = false;
    for (var i = 0; i < s.length; i++) {
      final c = s[i];
      if (c == '"') {
        inQuote = !inQuote;
      } else if (c == ' ' && !inQuote) {
        if (sb.isNotEmpty) {
          out.add(sb.toString());
          sb.clear();
        }
      } else {
        sb.write(c);
      }
    }
    if (sb.isNotEmpty) out.add(sb.toString());
    return out;
  }
}
