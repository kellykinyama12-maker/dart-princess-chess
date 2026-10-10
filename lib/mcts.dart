import 'dart:async';
import 'dart:math' as math;

import 'package:chess/chess.dart' as ch;

import 'alphabeta.dart';
import 'ab_worker.dart';
import 'board_expand.dart';
import 'eval_pool.dart';
import 'lc0_evaluator.dart';

/// Per-position statistics, shared across all tree nodes that transpose to the
/// same Zobrist key (the transposition table).
class _Stats {
  int visits = 0;
  double valueSum = 0.0;
  int vloss = 0; // pending virtual losses while a playout is in flight
  // Alpha-beta value prior (position STM POV) injected via the TT, weighted
  // like [abWeight] extra visits — stat transfer from the AB worker.
  double abValue = 0.0;
  int abWeight = 0;
  int abDepth = 0; // deepest embedded alpha-beta searched here (for ID)
}

/// A single MCTS tree node. Values are stored from the perspective of the side
/// to move *at this node* (negamax convention). Visit/value stats live in a
/// shared [_Stats] so transposing nodes pool their search knowledge.
class _Node {
  final String fen;
  final int stm; // side to move: 0 = White, 1 = Black
  final int key; // Zobrist hash, maintained incrementally
  final _Stats _s;
  double prior; // mutable so root exploration noise can be mixed in

  Map<String, _Node>? children; // null until expanded; {} for terminal nodes
  bool terminal = false;
  double terminalValue = 0.0;
  bool noiseApplied = false; // guards one-time Dirichlet noise per search root

  _Node(this.fen, this.stm, this.prior, this.key, this._s);

  int get visits => _s.visits;
  set visits(int v) => _s.visits = v;
  double get valueSum => _s.valueSum;
  set valueSum(double v) => _s.valueSum = v;
  int get vloss => _s.vloss;
  set vloss(int v) => _s.vloss = v;
  int get abWeight => _s.abWeight;
  double get abValue => _s.abValue;
  int get abDepth => _s.abDepth;
  set abDepth(int v) => _s.abDepth = v;

  /// Value estimate blending MCTS visits with the AB value prior.
  double get q {
    final n = _s.visits + _s.abWeight;
    if (n == 0) return 0.0;
    return (_s.valueSum + _s.abWeight * _s.abValue) / n;
  }
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
  final bool transpositions; // share stats across nodes with the same Zobrist key
  final double abProb; // per-ply chance a rollout commits to alpha-beta descent
  final math.Random _rng;

  // Embedded alpha-beta searcher (classical eval, persistent TT) used by the
  // unified rollout. Depth per position is iteratively deepened, capped here.
  static const int _abMaxDepth = 10;
  AlphaBeta _abSearcher = AlphaBeta();

  // Off-main-isolate AB searcher for the pooled/async rollout, so alpha-beta
  // runs concurrently with the net evaluations in flight (spawned lazily the
  // first time an AB leaf appears, kept across searches for TT reuse).
  AbWorker? _abWorker;
  Future<AbWorker>? _abWorkerStarting;

  _Node? _root; // persisted across searches to enable tree reuse
  final Map<int, _Stats> _tt = {}; // Zobrist key -> shared stats
  bool _stopFlag = false; // set by stop() to end a search budget early

  Mcts(
    Lc0Evaluator this.eval, {
    this.cPuct = 1.5,
    this.addRootNoise = true,
    this.dirichletAlpha = 0.3,
    this.dirichletEpsilon = 0.25,
    this.fpuReduction = 0.2,
    this.batchSize = 1,
    this.transpositions = true,
    this.abProb = 0.0,
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
    this.transpositions = true,
    this.abProb = 0.0,
    int? seed,
  })  : eval = null,
        _rng = math.Random(seed);

  /// Creates a node whose stats come from the transposition table (shared by
  /// Zobrist key) when [transpositions] is on, else a fresh per-node stat.
  _Node _mkNode(String fen, int stm, double prior, int key) => _Node(fen, stm,
      prior, key, transpositions ? _tt.putIfAbsent(key, _Stats.new) : _Stats());

  /// Visit count accumulated at the current persistent root (0 if none).
  int get rootVisits => _root?.visits ?? 0;

  /// FEN at the current persistent root (null if no search has started).
  String? get currentRootFen => _root?.fen;

  /// FENs of the top-[k] root children by visit count (alpha-beta focus set).
  List<String> topRootChildFens(int k) {
    final children = _root?.children;
    if (children == null) return const [];
    final entries = children.entries.toList()
      ..sort((a, b) => b.value.visits.compareTo(a.value.visits));
    return [for (final e in entries.take(k)) e.value.fen];
  }

  /// Map of each root move (absolute UCI) to its resulting child FEN.
  Map<String, String> rootChildFens() {
    final children = _root?.children;
    if (children == null) return const {};
    return {for (final e in children.entries) e.key: e.value.fen};
  }

  /// Net policy priors `{fen: {moveUci: prior}}` for the root and its top-[k]
  /// children, so the alpha-beta worker can order moves by the net's policy on
  /// exactly the nodes the MCTS has already expanded.
  Map<String, Map<String, double>> focusPriors(int k) {
    final out = <String, Map<String, double>>{};
    final root = _root;
    final children = root?.children;
    if (children == null) return out;
    out[root!.fen] = {
      for (final e in children.entries) e.key: e.value.prior
    };
    final top = children.entries.toList()
      ..sort((a, b) => b.value.visits.compareTo(a.value.visits));
    for (final e in top.take(k)) {
      final c = e.value;
      final gc = c.children;
      if (gc != null && gc.isNotEmpty) {
        out[c.fen] = {for (final g in gc.entries) g.key: g.value.prior};
      }
    }
    return out;
  }

  /// Drops the persistent tree (call when starting an unrelated position).
  void reset() {
    _root = null;
    _tt.clear();
    _abSearcher = AlphaBeta();
  }

  /// Releases the rollout AB worker isolate, if one was spawned.
  void dispose() {
    _abWorker?.dispose();
    _abWorker = null;
    _abWorkerStarting = null;
  }

  /// Stat transfer from alpha-beta: records [value] (position side-to-move POV,
  /// in [-1,1]) for [fen] in the transposition table, weighted like [weight]
  /// visits. Mates use a dominant weight; scores a modest one. No-op when the
  /// transposition table is off.
  void injectAb(String fen, double value, int weight) {
    if (!transpositions) return;
    final s = _tt.putIfAbsent(boardZobrist.ofFen(fen), _Stats.new);
    s.abValue = value;
    s.abWeight = weight;
  }

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
  List<MoveStat> search(String rootFen, int sims) =>
      searchBudget(rootFen, nodes: sims);

  /// Asks a running [searchBudget]/[searchBudgetAsync] to return early.
  void stop() => _stopFlag = true;

  /// Searches until [nodes] total root visits or the [time] deadline is hit
  /// (whichever comes first), or until [stop] is called.
  List<MoveStat> searchBudget(String rootFen, {int? nodes, Duration? time}) {
    _stopFlag = false;
    final root = _prepareRoot(rootFen);
    final target = nodes ?? (1 << 30);
    final deadline = time == null ? null : DateTime.now().add(time);
    if (!root.terminal) {
      while (root.visits < target && !_stopFlag && !_passed(deadline)) {
        if (batchSize <= 1) {
          _simulate(root);
        } else {
          _runBatch(root, math.min(batchSize, target - root.visits));
        }
      }
    }
    return _rootStats(root);
  }

  _Node _prepareRoot(String rootFen) {
    final reuse = _root != null && _root!.fen == rootFen;
    final root = reuse
        ? _root!
        : _mkNode(rootFen, _stm(rootFen), 1.0, boardZobrist.ofFen(rootFen));
    _root = root;
    // Expand the root first so exploration noise can be seeded on its children.
    if (root.children == null) {
      _expand(root);
    }
    _applyRootNoise(root);
    return root;
  }

  bool _passed(DateTime? deadline) =>
      deadline != null && !DateTime.now().isBefore(deadline);

  List<MoveStat> _rootStats(_Node root) {
    final stats = <MoveStat>[];
    root.children?.forEach((uci, child) {
      stats.add(MoveStat(uci, child.visits, child.q, child.prior));
    });
    stats.sort((a, b) => b.visits.compareTo(a.visits));
    return stats;
  }

  /// Principal variation from the current root, following the most-visited
  /// child at each ply, as absolute UCI moves.
  List<String> principalVariation({int maxLen = 16}) {
    final pv = <String>[];
    var node = _root;
    while (node != null && node.children != null && pv.length < maxLen) {
      String? bestUci;
      _Node? bestChild;
      var bestVisits = -1;
      node.children!.forEach((uci, child) {
        if (child.visits > bestVisits) {
          bestVisits = child.visits;
          bestUci = uci;
          bestChild = child;
        }
      });
      if (bestUci == null || bestVisits <= 0) break;
      pv.add(bestUci!);
      node = bestChild;
    }
    return pv;
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

  /// Pool-backed search: each virtual-loss leaf is expanded (move-gen + eval)
  /// in parallel on a worker isolate. Requires [Mcts.pooled]. Same total-visit
  /// budget and tree-reuse semantics as [search].
  Future<List<MoveStat>> searchAsync(String rootFen, int sims) =>
      searchBudgetAsync(rootFen, nodes: sims);

  /// Node/time-budgeted pool-backed search (see [searchBudget]).
  Future<List<MoveStat>> searchBudgetAsync(String rootFen,
      {int? nodes, Duration? time}) async {
    _stopFlag = false;
    final reuse = _root != null && _root!.fen == rootFen;
    final root = reuse
        ? _root!
        : _mkNode(rootFen, _stm(rootFen), 1.0, boardZobrist.ofFen(rootFen));
    _root = root;

    if (root.children == null) {
      _applyExpandResult(root, await pool!.expand(root.fen, root.stm, root.key));
    }
    _applyRootNoise(root);

    final target = nodes ?? (1 << 30);
    final deadline = time == null ? null : DateTime.now().add(time);
    if (!root.terminal) {
      while (root.visits < target && !_stopFlag && !_passed(deadline)) {
        await _runBatchAsync(root, math.min(batchSize, target - root.visits));
      }
    }
    return _rootStats(root);
  }

  /// Collects [count] leaves (virtual loss), expands the unique unexpanded ones
  /// concurrently on the pool, then backs the values up. AB-selected leaves are
  /// searched on a worker isolate in parallel, so alpha-beta compute overlaps
  /// the net evaluations instead of blocking the main isolate.
  Future<void> _runBatchAsync(_Node root, int count) async {
    final rollouts = <(List<_Node>, int)>[];
    for (var b = 0; b < count; b++) {
      rollouts.add(_collectLeafAb(root));
    }
    final values = <_Node, double>{};
    final leaves = <_Node>[]; // unique net leaves to expand on the pool
    final abLeaves = <_Node>[]; // unique AB leaves to search on the worker
    for (final (path, abDepth) in rollouts) {
      final leaf = path.last;
      if (abDepth > 0 && !leaf.terminal) {
        if (!values.containsKey(leaf) && !abLeaves.contains(leaf)) {
          abLeaves.add(leaf);
        }
        continue;
      }
      if (values.containsKey(leaf) || leaf.children != null) {
        if (leaf.terminal) values[leaf] = leaf.terminalValue;
        continue;
      }
      if (!leaves.contains(leaf)) leaves.add(leaf);
    }
    // Launch net and alpha-beta concurrently; await both.
    final nnFuture = Future.wait(
        [for (final leaf in leaves) pool!.expand(leaf.fen, leaf.stm, leaf.key)]);
    Future<List<AbInfo>>? abFuture;
    if (abLeaves.isNotEmpty) {
      final worker = await _ensureAbWorker();
      abFuture = worker.rollout(
          [for (final l in abLeaves) l.fen],
          [for (final l in abLeaves) l.abDepth],
          30);
    }
    final results = await nnFuture;
    for (var i = 0; i < leaves.length; i++) {
      values[leaves[i]] = _applyExpandResult(leaves[i], results[i]);
    }
    if (abFuture != null) {
      final infos = await abFuture;
      for (var i = 0; i < abLeaves.length; i++) {
        values[abLeaves[i]] = _abInfoValue(infos[i]);
      }
    }
    for (final (path, _) in rollouts) {
      _backup(path, values[path.last] ?? path.last.terminalValue);
    }
  }

  /// Returns the rollout AB worker, spawning it once on first use.
  Future<AbWorker> _ensureAbWorker() {
    final w = _abWorker;
    if (w != null) return Future.value(w);
    return _abWorkerStarting ??= AbWorker.start().then((worker) {
      _abWorker = worker;
      return worker;
    });
  }

  /// Attaches children (or marks terminal) from a worker [ExpandResult] and
  /// returns the node-POV value.
  double _applyExpandResult(_Node node, ExpandResult r) {
    if (r.terminal) {
      node.terminal = true;
      node.children = <String, _Node>{};
      node.terminalValue = r.checkmate ? -1.0 : 0.0;
      return node.terminalValue;
    }
    node.children = <String, _Node>{};
    for (var i = 0; i < r.legal.length; i++) {
      node.children![r.legal[i]] =
          _mkNode(r.childFens[i], 1 - node.stm, r.priors[i], r.childKeys[i]);
    }
    return r.value;
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
      final known = child.visits > 0 || child.abWeight > 0;
      final exploit = known ? -child.q : (parentQ - fpuReduction);
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
    final rollouts = <(List<_Node>, int)>[];
    for (var b = 0; b < count; b++) {
      rollouts.add(_collectLeafAb(root));
    }
    final values = <_Node, double>{};
    for (final (path, abDepth) in rollouts) {
      final leaf = path.last;
      final double v;
      if (abDepth > 0 && !leaf.terminal) {
        v = _abEval(leaf, abDepth);
      } else {
        v = values.putIfAbsent(
            leaf, () => leaf.terminal ? leaf.terminalValue : _expand(leaf));
      }
      _backup(path, v);
    }
  }

  /// Descends from [root] to an unexpanded/terminal leaf, applying a virtual
  /// loss to every node on the path.
  /// Descends from [root] to a leaf. Beyond the root, with probability [abProb]
  /// a node is evaluated by an embedded iteratively-deepened alpha-beta search
  /// instead of descending further; returns the path and that AB depth (0 = a
  /// normal leaf). Virtual loss is applied along the path.
  (List<_Node>, int) _collectLeafAb(_Node root) {
    final path = <_Node>[];
    var node = root;
    node.vloss++;
    path.add(node);
    while (!node.terminal &&
        node.children != null &&
        node.children!.isNotEmpty) {
      // Only an MCTS-visited (expanded) node may be handed to alpha-beta; a node
      // that merely has children via a transposition isn't a valid AB leaf yet.
      if (abProb > 0 &&
          !identical(node, root) &&
          node.visits > 0 &&
          _rng.nextDouble() < abProb) {
        final d = (node.abDepth + 1).clamp(1, _abMaxDepth);
        node.abDepth = d;
        return (path, d);
      }
      final parentVisits = node.visits + node.vloss;
      final parentQ = node.q; // FPU baseline for unvisited children
      _Node? best;
      var bestScore = double.negativeInfinity;
      node.children!.forEach((_, child) {
        final cv = child.visits + child.vloss;
        final abW = child.abWeight;
        final denom = cv + abW;
        // child.vloss adds a pending win for the child (= loss for this parent);
        // the AB prior adds abW virtual visits at child.abValue.
        final effQ = denom > 0
            ? (child.valueSum + child.vloss + abW * child.abValue) / denom
            : 0.0;
        final exploit = denom > 0 ? -effQ : (parentQ - fpuReduction);
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
    return (path, 0);
  }

  /// Embedded alpha-beta value of [node] to [depth] (position STM POV, [-1,1]),
  /// run inline (used by the single-threaded path).
  double _abEval(_Node node, int depth) {
    final r = _abSearcher.search(node.fen,
        maxDepth: depth, time: const Duration(milliseconds: 30));
    return _abScoreValue(r.scoreCp, r.isMate);
  }

  /// Converts an alpha-beta result from the rollout worker to a node-POV value.
  double _abInfoValue(AbInfo r) => _abScoreValue(r.scoreCp, r.mate);

  /// Maps an alpha-beta centipawn score (STM POV) to a [-1,1] value.
  double _abScoreValue(int scoreCp, bool mate) {
    if (mate) return scoreCp > 0 ? 1.0 : -1.0;
    final x = scoreCp / 400.0;
    if (x > 20) return 1.0;
    if (x < -20) return -1.0;
    final e = math.exp(2 * x);
    return (e - 1) / (e + 1); // tanh
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
    final exp = expandBoard(node.fen, node.key);
    if (exp.gameOver) {
      node.terminal = true;
      node.children = <String, _Node>{};
      node.terminalValue = exp.checkmate ? -1.0 : 0.0; // mated STM loses
      return _Prep(const [], const {}, const {});
    }
    final childFen = <String, String>{
      for (var i = 0; i < exp.legal.length; i++) exp.legal[i]: exp.childFens[i]
    };
    final childKey = <String, int>{
      for (var i = 0; i < exp.legal.length; i++) exp.legal[i]: exp.childKeys[i]
    };
    return _Prep(exp.legal, childFen, childKey);
  }

  double _applyExpansion(_Node node, _Prep prep, Evaluation e) {
    node.children = <String, _Node>{};
    for (final uci in prep.legal) {
      node.children![uci] = _mkNode(
          prep.childFen[uci]!, 1 - node.stm, e.priors[uci] ?? 0.0,
          prep.childKey[uci]!);
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
  final Map<String, int> childKey;
  _Prep(this.legal, this.childFen, this.childKey);
}
