# lc0 parity roadmap

Goal: evolve the project from a **move-sequence language model** (predicts the
next move from move history) toward lc0's actual design — a **position
evaluator** (policy + value over a board) used inside a **search**. We won't
match lc0's strength (that needs GPU-scale training), but we can reach
*architectural parity*: the same inputs, the same targets, the same inference
shape.

See [README.md](README.md) for the baseline and the lc0 comparison.

## The gaps (baseline → lc0)

| # | Gap | Baseline (move-LM) | lc0 |
|---|---|---|---|
| G1 | Input | move tokens | 112 board planes (8×8) |
| G2 | Targets | next move only | policy(1858) + value(WDL) + moves-left |
| G3 | Heads | one LM head | policy head + value head + MLH |
| G4 | Legality | none | policy masked to legal moves |
| G5 | Inference | single forward pass | MCTS / PUCT search |
| G6 | Scale | ~1–2M params, CPU | millions–100M+, GPU |

## Stages

### Stage 0 — Move-sequence LM baseline  ✅ done
Decoder-only GPT over UCI tokens. Files: `bin/parse_lc0.dart`,
`bin/train_moves.dart`, `bin/predict_move.dart`.

### Stage 1 — Decode the full V6 target set  ✅ done & validated
Recover everything lc0 trains on from each record, not just the played move.
- Files: [lib/lc0_record.dart](../lib/lc0_record.dart), [lib/lc0_tar.dart](../lib/lc0_tar.dart), [bin/inspect_record.dart](../bin/inspect_record.dart).
- Exposes: `plane(0..103)`, `policyAt/ topPolicy`, `resultWdl/ bestWdl`, `pliesLeft`, `playedUci`.
- **Acceptance (met):** start position decodes to 32 occupied squares; policy,
  WDL, and moves-left are all sane (`dart run bin/inspect_record.dart`).

### Stage 2 — Reuse lc0's real net inside MCTS  ✅ done & validated
Addresses **G1, G4, G5** at once by using lc0's own network as the evaluator
(confirmed to run on CPU, ~61 ms/forward for the 128×10 744706 net) instead of
first training a weak CPU net. This is the "reuse lc0 code" path.
- **Evaluator:** [lib/lc0_evaluator.dart](../lib/lc0_evaluator.dart) wraps
  `Lc0Net` (via `Lc0Reader`/`Lc0Input.fromFen`), reads the WDL value head and
  the conv policy head `[1,80,8,8]`, and softmaxes the legal-move logits into
  priors.
- **Move mapping:** [lib/move_index.dart](../lib/move_index.dart) +
  [lib/conv_policy_map.dart](../lib/conv_policy_map.dart) (generated from lc0's
  `src/neural/tables/policy_map.h`, 73×8×8 = 4672 entries). An absolute UCI move
  → mover-POV (reusing `toAbsoluteUci`) → 1858 index (`kPolicyIndex`) →
  conv-output index (inverse `kConvPolicyMap`) → logit. Knight underpromotions
  (absent from the classical 1858) floor to a tiny prior but stay legal.
- **Search:** [lib/mcts.dart](../lib/mcts.dart) — negamax PUCT
  (Q + c_puct·P·√N/(1+n)), `chess` package for legal-move generation and board
  transitions, terminal nodes scored −1 (mated STM) / 0 (draw).
- **Driver:** [bin/play_search.dart](../bin/play_search.dart) —
  `dart run bin/play_search.dart --sims 100 [FEN]`.
- **Acceptance (met):** start position, 40 sims → best move `g1f3` (legal, top
  opening), priors `Nf3 54.6% / e3 16.3% / Nc3 15.0%`, Q ≈ 0 (balanced), ~61
  ms/sim on CPU.

### Stage 2b (optional) — Train our own position net  ⬜
Only needed if we want a from-scratch net rather than reusing lc0's weights.
Addresses **G1, G2, G3**.
- **Input (112 planes):** the 104 stored planes + reconstruct the 8 aux planes
  (castling ×4, side-to-move, rule50, all-zeros, all-ones) from the record
  bytes, exactly as lc0's chunkparser does.
- **Model (dart_pytorch):** treat the 64 squares as tokens, each a 112-dim
  feature → `Linear` embed → `TransformerBlock` ×N (non-causal encoder) →
  pooled/board features → three heads:
  - **policy**: Linear → 1858 logits, trained with cross-entropy/KL against the
    record's policy distribution (legal moves only).
  - **value**: Linear → 3 (WDL), cross-entropy against `resultWdl`.
  - **moves-left**: Linear → 1, regression against `pliesLeft`.
- **Loss:** `policy + c1·value + c2·mlh` (lc0-style weighting).
- **Files (planned):** `lib/lc0_dataset.dart` (plane tensor + targets batcher),
  `lib/board_net.dart` (encoder + heads), `bin/train_eval.dart`.
- **Acceptance:** on held-out positions, policy top-1 agrees with the self-play
  played move > ~40%, and value sign correlates with game result.

### Stage 3 — Legality masking + move selection  ✅ done (via Stage 2)
Addresses **G4**. The `chess` package generates legal moves and
`lib/move_index.dart` maps each to its 1858 index (inverse of `kPolicyIndex`
with orientation via `toAbsoluteUci`); only legal moves receive priors, so
illegal logits are effectively masked. Delivered as part of the Stage 2
evaluator/search.

### Stage 4 — Minimal search (PUCT/MCTS)  ✅ done (via Stage 2)
Addresses **G5**. [lib/mcts.dart](../lib/mcts.dart) is a PUCT tree with priors
from the (lc0) policy head and leaf values from the value head. Remaining polish
(Dirichlet root noise, tree reuse, virtual loss, temperature) is optional.
- **Acceptance:** with search the engine selects sound moves and is strictly
  stronger than the raw argmax policy as sims increase.

### Stage 4c — Parallel search (isolate eval pool)  ✅ done
A pool of worker isolates ([lib/eval_pool.dart](../lib/eval_pool.dart)) each hold
their own `Lc0Net`; virtual-loss leaf batches are expanded (move-gen + eval) in
parallel off the main isolate. `--workers N` in play_search / `Threads` in UCI.
On a 16-core CPU: ~52 ms/sim (1 thread) → ~11 ms/sim (12 threads).

### Stage 4d — UCI interface  ✅ done
[bin/uci.dart](../bin/uci.dart) speaks UCI so the engine plays in any GUI or
against other engines: `position`, `go movetime|nodes|infinite|wtime/btime`,
`stop`, `setoption Threads|Nodes|Cpuct`, with `info … score cp … pv …` and
`bestmove`. Node/time budgets + `stop` + principal variation live in
[lib/mcts.dart](../lib/mcts.dart) (`searchBudget`/`searchBudgetAsync`). Tree
reuse carries across `position` commands that extend the previous line.
- **Acceptance (met):** a full UCI handshake drives legal play; e.g. after
  `e2e4 e7e5`, `go nodes 150` → `bestmove b1c3` with a multi-ply PV.

### Stage 5 — Scale & quality  ⬜
Addresses **G6**. Bigger net + more data; optional SE / attention refinements;
evaluate against puzzles and known positions; optional GPU via WSL (the package
ships CUDA kernels for Linux/WSL2).

- **Tactics harness** ✅ [bin/solve_epd.dart](../bin/solve_epd.dart) runs EPD
  suites (`bm`/`am`, SAN-matched to the engine's move) and reports a solve rate;
  ships a small WAC subset in [test/tactics.epd](../test/tactics.epd).
  `dart run bin/solve_epd.dart --nodes 2000 --workers 8`. Use it to track
  strength as nodes/net scale.
- **Configurable net** ✅ all drivers take `--net <path>` (or `LC0_WEIGHTS` env /
  UCI `WeightsFile`) so a stronger evaluator can be dropped in without code
  changes. [bin/net_info.dart](../bin/net_info.dart) prints a net's architecture
  + noise-free startpos priors to check compatibility.
- **Net-compatibility finding** ⚠️ dart_pytorch 0.1.3's `Lc0Reader` only reliably
  parses 744706-style SE-ResNet nets (conv policy + WDL). Tested larger nets:
  `t1-256x10-distilled` / `t1-256x10-rl-base` are **rejected** ("missing
  classical fields"); the 2019 `256x20-t40` **loads but is mis-parsed** (scrambled
  policy, e.g. h2h3/a2a3 as top priors); attention-body nets (BT*, `*x…h`) are
  unsupported. So bigger-net strength scaling is currently blocked by the reader,
  not by this project's evaluator/search (same code gives correct priors on
  744706). A future dart_pytorch with broader net support unlocks this lever.

## Notes / constraints
- CPU-only here, so Stage 2 nets stay modest; GPU (WSL) unlocks Stage 5.
- test80 data is `input_format == 1` (classical 112-plane) — confirmed by the
  inspector — so the plane reconstruction follows the classical layout.
- Everything stays on a separate track from the move-LM (`data/model.dptc`); the
  position model will save under its own name (e.g. `data/eval.dptc`).
