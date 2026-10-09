import 'dart:async';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

import 'eval_pool.dart';
import 'lc0_evaluator.dart';

/// A single MCTS tree node. Values are stored from the perspective of the side
/// to move *at this node* (negamax convention).
class _Node {
  final String fen;
  final int stm; // side to move: 0 = White, 1 = Black
  double prior; // mutable so root exploration noise can be mixed in

  int visits = 0;
  double valueSum = 0.0;

  Map<String, _Node>? children; // null until expanded; {} for terminal nodes
  bool terminal = false;
  double terminalValue = 0.0;
  bool noiseApplied = false; // guards one-time Dirichlet noise per search root
  int vloss = 0; // pending virtual losses while a playout is in flight

  _Node(this.fen, this.stm, this.prior);

  double get q => visits == 0 ? 0.0 : valueSum / visits;
}

/// Result row for one root move after a search.
class MoveStat {
  final String uci;
  final int visits;
  final double q;
  final double prior;
  MoveStat(this.uci, this.visits, this.q, this.prior);
}

/// PUCT Monte-Carlo Tree Search driven by an [Lc0Evaluator] leaf evaluator,
/// mirroring lc0's search: selection by Q + c_puct * P * sqrt(N) / (1 + n).
/// Supports Dirichlet root-exploration noise and tree reuse across moves.
class Mcts {
  final Lc0Evaluator? eval;
  final EvalPool? pool;
  final double cPuct;
  final bool addRootNoise;
  final double dirichletAlpha;
  final double dirichletEpsilon;
  final double fpuReduction; // penalty applied to unvisited children (FPU)
  final int batchSize; // >1 collects a batch of distinct leaves via virtual loss
  final math.Random _rng;

  _Node? _root; // persisted across searches to enable tree reuse

  Mcts(
    Lc0Evaluator this.eval, {
    this.cPuct = 1.5,
    this.addRootNoise = true,
    this.dirichletAlpha = 0.3,
    this.dirichletEpsilon = 0.25,
    this.fpuReduction = 0.2,
    this.batchSize = 1,
    int? seed,
  })  : pool = null,
        _rng = math.Random(seed);

  /// Pool-backed engine: leaf evaluations run in parallel on worker isolates.
  /// Use [searchAsync] with this constructor.
  Mcts.pooled(
    EvalPool this.pool, {
    this.cPuct = 1.5,
    this.addRootNoise = true,
    this.dirichletAlpha = 0.3,
    this.dirichletEpsilon = 0.25,
    this.fpuReduction = 0.2,
    this.batchSize = 8,
    int? seed,
  })  : eval = null,
        _rng = math.Random(seed);

  /// Visit count accumulated at the current persistent root (0 if none).
  int get rootVisits => _root?.visits ?? 0;

  /// Drops the persistent tree (call when starting an unrelated position).
  void reset() => _root = null;

  /// Advances the persistent tree to the child reached by [uci] so the next
  /// [search] reuses its subtree. Returns false (and clears the tree) if no
  /// cached child exists for [uci].
  bool advance(String uci) {
    final child = _root?.children?[uci];
    if (child == null) {
      _root = null;
      return false;
    }
    _root = child;
    return true;
  }

  /// Runs simulations until the root reaches [sims] total visits (reused-tree
  /// visits count toward that budget), then returns root moves ranked by visit
  /// count. Reuses the persistent subtree when it matches [rootFen].
  List<MoveStat> search(String rootFen, int sims) {
    final reuse = _root != null && _root!.fen == rootFen;
    final root = reuse ? _root! : _Node(rootFen, _stm(rootFen), 1.0);
    _root = root;

    // Expand the root first so exploration noise can be seeded on its children.
    if (root.children == null) {
      _expand(root);
    }
    _applyRootNoise(root);

    if (!root.terminal) {
      while (root.visits < sims) {
        if (batchSize <= 1) {
          _simulate(root);
        } else {
          _runBatch(root, math.min(batchSize, sims - root.visits));
        }
      }
    }

    final stats = <MoveStat>[];
    root.children?.forEach((uci, child) {
      stats.add(MoveStat(uci, child.visits, child.q, child.prior));
    });
    stats.sort((a, b) => b.visits.compareTo(a.visits));
    return stats;
  }

  /// Picks a move from search [stats]. [temperature] 0 is greedy (most visits);
  /// higher values sample proportionally to visits^(1/temperature).
  MoveStat selectMove(List<MoveStat> stats, {double temperature = 0.0}) {
    if (stats.isEmpty) {
      throw StateError('no moves to select from');
    }
    if (temperature <= 0) return stats.first; // stats are visit-sorted
    final inv = 1.0 / temperature;
    final weights =
        stats.map((s) => math.pow(s.visits.toDouble(), inv).toDouble()).toList();
    final total = weights.fold<double>(0.0, (a, b) => a + b);
    if (total <= 0) return stats.first;
    var r = _rng.nextDouble() * total;
    for (var i = 0; i < stats.length; i++) {
      r -= weights[i];
      if (r <= 0) return stats[i];
    }
    return stats.last;
  }

  /// Pool-backed search: evaluates each virtual-loss leaf batch in parallel on
  /// the isolate pool. Requires [Mcts.pooled]. Same total-visit budget and
  /// tree-reuse semantics as [search].
  Future<List<MoveStat>> searchAsync(String rootFen, int sims) async {
    final reuse = _root != null && _root!.fen == rootFen;
    final root = reuse ? _root! : _Node(rootFen, _stm(rootFen), 1.0);
    _root = root;

    if (root.children == null) {
      final prep = _prepareExpand(root);
      if (!root.terminal) {
        final e = await pool!.evaluate(root.fen, root.stm, prep.legal);
        _applyExpansion(root, prep, e);
      }
    }
    _applyRootNoise(root);

    if (!root.terminal) {
      while (root.visits < sims) {
        await _runBatchAsync(root, math.min(batchSize, sims - root.visits));
      }
    }
    final stats = <MoveStat>[];
    root.children?.forEach((uci, child) {
      stats.add(MoveStat(uci, child.visits, child.q, child.prior));
    });
    stats.sort((a, b) => b.visits.compareTo(a.visits));
    return stats;
  }

  /// Collects [count] leaves (virtual loss), evaluates the unique unexpanded
  /// ones concurrently on the pool, then expands and backs them up.
  Future<void> _runBatchAsync(_Node root, int count) async {
    final paths = <List<_Node>>[];
    for (var b = 0; b < count; b++) {
      paths.add(_collectLeaf(root));
    }
    final values = <_Node, double>{};
    final preps = <_Node, _Prep>{};
    final leaves = <_Node>[];
    for (final path in paths) {
      final leaf = path.last;
      if (values.containsKey(leaf) || preps.containsKey(leaf)) continue;
      if (leaf.terminal) {
        values[leaf] = leaf.terminalValue;
        continue;
      }
      final prep = _prepareExpand(leaf);
      if (leaf.terminal) {
        values[leaf] = leaf.terminalValue;
        continue;
      }
      preps[leaf] = prep;
      leaves.add(leaf);
    }
    final results = await Future.wait([
      for (final leaf in leaves)
        pool!.evaluate(leaf.fen, leaf.stm, preps[leaf]!.legal),
    ]);
    for (var i = 0; i < leaves.length; i++) {
      final leaf = leaves[i];
      values[leaf] = _applyExpansion(leaf, preps[leaf]!, results[i]);
    }
    for (final path in paths) {
      _backup(path, values[path.last]!);
    }
  }

  /// Mixes Dirichlet(alpha) noise into the root priors once per root:
  /// P <- (1 - eps) * P + eps * noise, as in AlphaZero/lc0.
  void _applyRootNoise(_Node root) {
    final children = root.children;
    if (!addRootNoise ||
        root.noiseApplied ||
        children == null ||
        children.isEmpty) {
      return;
    }
    root.noiseApplied = true;
    final moves = children.keys.toList();
    final noise = _dirichlet(moves.length, dirichletAlpha);
    for (var i = 0; i < moves.length; i++) {
      final child = children[moves[i]]!;
      child.prior =
          (1 - dirichletEpsilon) * child.prior + dirichletEpsilon * noise[i];
    }
  }

  List<double> _dirichlet(int n, double alpha) {
    final g = List<double>.generate(n, (_) => _gamma(alpha));
    final sum = g.fold<double>(0.0, (a, b) => a + b);
    if (sum <= 0) return List<double>.filled(n, 1.0 / n);
    return g.map((x) => x / sum).toList();
  }

  /// Gamma(alpha, 1) via Marsaglia–Tsang, with the alpha<1 boosting trick.
  double _gamma(double alpha) {
    if (alpha < 1.0) {
      final u = _rng.nextDouble();
      return _gamma(alpha + 1.0) * math.pow(u, 1.0 / alpha).toDouble();
    }
    final d = alpha - 1.0 / 3.0;
    final c = 1.0 / math.sqrt(9.0 * d);
    while (true) {
      double x, v;
      do {
        x = _gaussian();
        v = 1.0 + c * x;
      } while (v <= 0.0);
      v = v * v * v;
      final u = _rng.nextDouble();
      if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
      if (math.log(u) < 0.5 * x * x + d * (1.0 - v + math.log(v))) return d * v;
    }
  }

  double _gaussian() {
    final u1 = _rng.nextDouble();
    final u2 = _rng.nextDouble();
    return math.sqrt(-2.0 * math.log(u1 <= 0 ? 1e-12 : u1)) *
        math.cos(2 * math.pi * u2);
  }

  /// One simulation; returns the value from [node]'s side-to-move perspective.
  double _simulate(_Node node) {
    if (node.terminal) {
      node.visits++;
      node.valueSum += node.terminalValue;
      return node.terminalValue;
    }

    if (node.children == null) {
      final v = _expand(node);
      node.visits++;
      node.valueSum += node.terminal ? node.terminalValue : v;
      return node.terminal ? node.terminalValue : v;
    }

    final parentVisits = node.visits;
    final parentQ = node.q; // FPU baseline for unvisited children
    _Node? best;
    var bestScore = double.negativeInfinity;
    node.children!.forEach((_, child) {
      final exploit =
          child.visits > 0 ? -child.q : (parentQ - fpuReduction);
      final explore =
          cPuct * child.prior * math.sqrt(parentVisits + 1) / (1 + child.visits);
      final score = exploit + explore;
      if (score > bestScore) {
        bestScore = score;
        best = child;
      }
    });

    final childValue = _simulate(best!);
    final v = -childValue;
    node.visits++;
    node.valueSum += v;
    return v;
  }

  /// Collects [count] leaves using virtual loss so the paths diverge, then
  /// evaluates each unique leaf once and backs the values up. This is the leaf
  /// batching that virtual loss enables (network eval stays sequential on CPU;
  /// true parallelism would run the evals on isolates).
  void _runBatch(_Node root, int count) {
    final paths = <List<_Node>>[];
    for (var b = 0; b < count; b++) {
      paths.add(_collectLeaf(root));
    }
    final values = <_Node, double>{};
    for (final path in paths) {
      final leaf = path.last;
      final v = values.putIfAbsent(
          leaf, () => leaf.terminal ? leaf.terminalValue : _expand(leaf));
      _backup(path, v);
    }
  }

  /// Descends from [root] to an unexpanded/terminal leaf, applying a virtual
  /// loss to every node on the path.
  List<_Node> _collectLeaf(_Node root) {
    final path = <_Node>[];
    var node = root;
    node.vloss++;
    path.add(node);
    while (!node.terminal &&
        node.children != null &&
        node.children!.isNotEmpty) {
      final parentVisits = node.visits + node.vloss;
      final parentQ = node.q; // FPU baseline for unvisited children
      _Node? best;
      var bestScore = double.negativeInfinity;
      node.children!.forEach((_, child) {
        final cv = child.visits + child.vloss;
        // child.vloss adds a pending win for the child (= loss for this parent).
        final effQ = cv > 0 ? (child.valueSum + child.vloss) / cv : 0.0;
        final exploit = cv > 0 ? -effQ : (parentQ - fpuReduction);
        final explore =
            cPuct * child.prior * math.sqrt(parentVisits + 1) / (1 + cv);
        final score = exploit + explore;
        if (score > bestScore) {
          bestScore = score;
          best = child;
        }
      });
      node = best!;
      node.vloss++;
      path.add(node);
    }
    return path;
  }

  /// Backs the leaf-POV value [vLeaf] up [path], flipping sign each ply and
  /// removing the virtual loss applied during collection.
  void _backup(List<_Node> path, double vLeaf) {
    final last = path.length - 1;
    for (var i = last; i >= 0; i--) {
      final node = path[i];
      final sign = ((last - i) % 2 == 0) ? 1.0 : -1.0;
      node.visits++;
      node.valueSum += vLeaf * sign;
      node.vloss--;
    }
  }

  /// Expands [node] synchronously using the in-process evaluator.
  double _expand(_Node node) {
    final prep = _prepareExpand(node);
    if (node.terminal) return node.terminalValue;
    final e = eval!.evaluate(node.fen, node.stm, prep.legal);
    return _applyExpansion(node, prep, e);
  }

  /// Generates legal moves + resulting child FENs (or marks [node] terminal)
  /// without running the network, so the eval can be offloaded to a pool.
  _Prep _prepareExpand(_Node node) {
    final board = ch.Chess.fromFEN(node.fen);

    if (board.game_over) {
      node.terminal = true;
      node.children = <String, _Node>{};
      node.terminalValue = board.in_checkmate ? -1.0 : 0.0; // mated STM loses
      return _Prep(const [], const {});
    }

    final verbose = board.moves({'verbose': true}) as List;
    final legal = <String>[];
    final childFen = <String, String>{};
    for (final mv in verbose) {
      final m = mv as Map;
      final from = m['from'] as String;
      final to = m['to'] as String;
      final promo = m['promotion'];
      final uci =
          from + to + (promo != null ? promo.toString().toLowerCase() : '');
      final child = ch.Chess.fromFEN(node.fen);
      final moveMap = <String, dynamic>{'from': from, 'to': to};
      if (promo != null) moveMap['promotion'] = promo;
      child.move(moveMap);
      legal.add(uci);
      childFen[uci] = child.fen;
    }
    return _Prep(legal, childFen);
  }

  double _applyExpansion(_Node node, _Prep prep, Evaluation e) {
    node.children = <String, _Node>{};
    for (final uci in prep.legal) {
      node.children![uci] =
          _Node(prep.childFen[uci]!, 1 - node.stm, e.priors[uci] ?? 0.0);
    }
    return e.value;
  }

  int _stm(String fen) {
    final parts = fen.split(' ');
    return (parts.length > 1 && parts[1] == 'b') ? 1 : 0;
  }
}

class _Prep {
  final List<String> legal;
  final Map<String, String> childFen;
  _Prep(this.legal, this.childFen);
}
