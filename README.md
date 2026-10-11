# dart-princess-chess

A chess engine written in **Dart** that runs real **Leela Chess Zero (lc0)**
neural networks inside a PUCT **Monte-Carlo Tree Search**, backed by a classical
**alpha-beta tactical co-processor**. It speaks **UCI**, so it plays in any chess
GUI, and ships a small **cutechess-style match runner** for engine-vs-engine
games.

Neural-network inference is pure Dart on top of the
[`dart_pytorch`](https://github.com/KellyKinyama/dart-pytorch) tensor library —
no GPU, no Python, no ONNX runtime.

> This repo began as a move-sequence language model trained on lc0 data (those
> notes live in [`docs/README.md`](docs/README.md)); it has since grown into a
> full search engine around lc0's own networks.

> 💡 **Strongest engine:** the pure-Dart port of DeepMind's **searchless-chess**
> transformer (`bin/searchless_uci.dart`) is the best-performing engine here — the
> only one to draw lc0. See [Searchless-chess engine](#searchless-chess-engine-strongest-so-far).

---

## What's inside

```
          UCI (bin/uci.dart)  ──►  any chess GUI / the match runner
                 │
                 ▼
     PUCT MCTS (lib/mcts.dart)  ◄── Dirichlet noise, tree reuse, virtual-loss batching
         │                     \
         │ leaf eval            \  tactical override
         ▼                       ▼
  EvalPool of isolates     AbWorker isolate (lib/ab_worker.dart)
  (lib/eval_pool.dart)     persistent-TT alpha-beta (lib/alphabeta.dart)
         │                       │
         ▼                       ▼
  lc0 net forward          classical material + PST + quiescence
  (dart_pytorch)           searches the root + MCTS best-move nodes
```

- **PUCT MCTS** — negamax selection (`Q + c·P·√N/(1+n)`), FPU, Dirichlet root
  noise, tree reuse between moves, and virtual-loss **leaf batching** across a
  pool of worker isolates for parallel evaluation.
- **lc0 network inference** — two network families are supported through
  `dart_pytorch`:
  - **Classical** SE-ResNet conv nets (e.g. `744706`, 128×10).
  - **Attention-body transformers** (e.g. `t1-256x10-distilled`) — embedding +
    gating, smolgen, multi-head attention, ReLU² FFN, attention policy map,
    WDL value and moves-left heads. Validated bit-for-bit against lc0 v0.32.1.
  - The attention forward is SIMD-optimized (`Float32x4`, 8-row-blocked matmul):
    **~375 ms → ~136 ms** per forward, output identical to lc0.
- **Alpha-beta tactical co-processor** (`lib/alphabeta.dart` + `lib/ab_worker.dart`)
  — iterative deepening, transposition table, quiescence, and **net-prior move
  ordering** (uses the MCTS policy where available, MVV-LVA otherwise). It runs
  on its own isolate during the MCTS search, focuses on the root and the MCTS
  best-move nodes, and **overrides the net's move only when it sees a concrete
  tactic** (mate or a clear material gap).
- **UCI engine** (`bin/uci.dart`) — options `Threads`, `Nodes`, `Cpuct`,
  `Ponder`, `UseAB`, `WeightsFile`; time management, pondering, `info` output.
- **Match runner** (`bin/match.dart` + `lib/uci_engine.dart`) — drives two UCI
  engines, enforces `movetime` or `base+inc` time control, alternates colors,
  detects game end via the `chess` package, forfeits on illegal/timeout, and
  writes **PGN**.

---

## Results (this build)

**Tactics — WAC subset, 3 s/pos, 8 workers**

| Engine | Solved |
|---|---|
| MCTS alone (net only) | 1 / 7 |
| **MCTS + alpha-beta** | **6 / 7** |
| alpha-beta alone | 6 / 7 (~445k nodes/s) |

**Self-play, 800 ms/move (same conv net)**

- **AB on vs AB off:** **+4 =2 −0 (83.3%)** — the tactical worker won every
  decisive game, never lost. (~+280 Elo at this setting, small sample.)
- **AB on vs lc0 (the reference engine, same net):** **0 – 4.** lc0's mature,
  far faster search (thousands of nodes/move vs our hundreds in pure Dart)
  mates us in 20–55 plies. The net quality is identical; the gap is search
  throughput — which is exactly what the alpha-beta worker and the SIMD forward
  are chipping away at.

---

## Searchless-chess engine (strongest so far)

A pure-Dart loader + UCI engine for DeepMind's **"searchless chess"** transformers
([*Amortized Planning with Large-Scale Transformers*](https://arxiv.org/abs/2402.04494),
NeurIPS 2024). These decoder-only transformers are trained by supervised learning
on Stockfish-annotated positions and predict a **win probability for every legal
move in a single forward pass — no search at all**. The released sizes reach
~2000–2300 (9M) up to ~2700 Lichess blitz (270M).

**No Python, no JAX, no GPU.** The weights load from a plain `params.npz` (a ZIP
of `.npy` arrays) via the `archive` package, and the LLaMa-style decoder (learned
token + positional embeddings, pre-LN blocks, SwiGLU MLP, post-LN, 128-bucket
value head) runs on the CPU with a hand-rolled `Float32x4` SIMD forward. The npz
array order is reconstructed by replicating jax's sorted-key pytree flatten.

> **Result vs lc0** (conv `744706` 128×10, 800 ms/move): **+0 =2 −2 (25%)** — the
> only engine in this repo to hold lc0 to draws. It plays sound, theoretical chess
> (real Sicilians, no positional drift), picking moves purely from the net's
> per-move win probabilities.

### 1. Get the 9M weights (no Python)

Download the npz mirror into `models/searchless/`:

```powershell
$b = 'https://huggingface.co/dbest-isi/searchless-chess-9M-selfplay/resolve/main'
New-Item -ItemType Directory -Force models\searchless | Out-Null
Invoke-WebRequest -UseBasicParsing "$b/params.npz?download=true" -OutFile models\searchless\params.npz
```

### 2. Run it (UCI)

```powershell
dart run bin/searchless_uci.dart
# uci / isready / position startpos / go movetime 800
# options: setoption name WeightsFile value models/searchless/params.npz
#          setoption name Workers value 15
```

On `isready` it spins up an isolate pool (one net copy per worker) and then scores
a position's legal moves in parallel — **~800 ms/move** after a one-time ~3.5 s
load (15 × ~36 MB). `movetime` is accepted but the net is single-pass, so its
strength does not scale with time. Quick self-check (prints top moves + timing):

```powershell
dart run bin/searchless_test.dart
# startpos -> e2e4 53.4%, d2d4 53.0%, g1f3 52.5% ...
```

**Speed — optional OpenBLAS fast path.** The transformer matmuls use OpenBLAS
`sgemm` when a BLAS library is available, otherwise a pure-Dart `Float32x4`
kernel (bit-identical output, slower). Point `DART_PYTORCH_BLAS` at any OpenBLAS
dll/so (numpy/scipy ship one) and keep one BLAS thread per isolate:

```powershell
$env:DART_PYTORCH_BLAS   = 'C:\...\numpy.libs\libscipy_openblas64_-....dll'
$env:OPENBLAS_NUM_THREADS = '1'
dart run bin/searchless_uci.dart
```

Measured **~6× speedup** on the 9M (single-thread **~238 → ~42 ms/move**, same
moves). This is also what makes 270M practical (~7 → ~1.2 s/move single-thread).

### 3. Bigger models (136M / 270M — one-time offline Python)

DeepMind only publishes these in Orbax format, so convert once to npz. This is the
**only** Python and it runs offline — inference stays pure Dart. On Windows,
pin an **older** `orbax-checkpoint` (newer ones fail to unpack under `MAX_PATH`):

```powershell
pip install "jax[cpu]" "orbax-checkpoint==0.5.23" numpy
# download + unzip https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
python scripts/convert_searchless_checkpoint.py --ckpt ./270M --out params_270M.npz
```

The loader auto-detects `dim`/layers from the npz. The 270M weights are ~1 GB
**per isolate**, so lower the worker count to bound RAM, and enable BLAS:

```powershell
$env:DART_PYTORCH_BLAS = '...openblas...dll'; $env:OPENBLAS_NUM_THREADS = '1'
dart run bin/searchless_uci.dart
# setoption name WeightsFile value params_270M.npz
# setoption name Workers value 6     # ~6 GB RAM, ~1–2 s/move with BLAS
```

Verified end-to-end: 270M loads (182 arrays, **dim 1024 × 16 layers**) and plays
correctly (`e4` 54%, mates found). Without BLAS it is ~16× slower.

### 4. Stronger play — 2-ply search

[`bin/searchless_2ply_uci.dart`](bin/searchless_2ply_uci.dart) wraps the net in a
depth-1 minimax: it takes the top-`TopK` moves by the net's win probability,
plays each, and lets the net pick the opponent's best reply — scoring a move by
`1 − max(opponent win prob)`. This is the search the paper shows improves on the
raw amortized policy. Fast with BLAS (~2–3 s/move at `TopK=10`).

> **Result vs lc0** (8 games, opening book [`openings.epd`](openings.epd),
> 800 ms/move): **+0 =6 −2 (37.5%)** — holds lc0 to draws far more often than the
> raw 1-ply net (~25%), fixing several losses into draws.

### 5. Experimental hybrid

[`bin/searchless_hybrid_uci.dart`](bin/searchless_hybrid_uci.dart) runs the net as
the strategic chooser with a shallow NNUE alpha-beta as a *blunder veto*. In
testing this **underperformed the standalone net** (the 9M is stronger than a
sub-second α-β, so overrides add errors), so it is kept only as a tunable
experiment (`AbMs`, `MarginCp`, `VetoFloorCp`).

### Files

| File | Role |
|---|---|
| [`lib/searchless.dart`](lib/searchless.dart) | npz/npy loader, FEN tokenizer, 1968-move action map, SIMD transformer forward, action-value engine |
| [`lib/searchless_pool.dart`](lib/searchless_pool.dart) | Isolate pool: parallel per-move scoring, full win-prob map + best move |
| [`bin/searchless_uci.dart`](bin/searchless_uci.dart) | **UCI engine (standalone net — strongest single-pass)** |
| [`bin/searchless_2ply_uci.dart`](bin/searchless_2ply_uci.dart) | UCI engine: depth-1 minimax using the net as the leaf evaluator (37.5% vs lc0) |
| [`bin/searchless_hybrid_uci.dart`](bin/searchless_hybrid_uci.dart) | Experimental net + alpha-beta blunder-veto hybrid |
| [`bin/searchless_test.dart`](bin/searchless_test.dart) | Smoke test (prints ranked moves + timing) |
| [`scripts/convert_searchless_checkpoint.py`](scripts/convert_searchless_checkpoint.py) | One-time offline Orbax → npz converter for 136M / 270M |
| [`openings.epd`](openings.epd) | Small varied-opening book for the match runner |

---

## Components

| File | Role |
|---|---|
| [`lib/mcts.dart`](lib/mcts.dart) | PUCT MCTS (sync + isolate-pool async), virtual-loss batching, tree reuse |
| [`lib/eval_pool.dart`](lib/eval_pool.dart) | Pool of worker isolates, each holding an `Lc0Evaluator` |
| [`lib/lc0_evaluator.dart`](lib/lc0_evaluator.dart) | Loads a net (conv or attention) and evaluates a position → WDL + priors |
| [`lib/alphabeta.dart`](lib/alphabeta.dart) | Classical negamax α-β: ID, TT, quiescence, PST eval, net-prior ordering |
| [`lib/ab_worker.dart`](lib/ab_worker.dart) | Isolate wrapper + `combineRoot` tactical-override logic |
| [`lib/board_expand.dart`](lib/board_expand.dart) | Legal-move generation + child FENs (via `chess`) |
| [`lib/move_index.dart`](lib/move_index.dart) | UCI ↔ lc0 policy index mapping |
| [`bin/uci.dart`](bin/uci.dart) | UCI engine front-end |
| [`bin/match.dart`](bin/match.dart) + [`lib/uci_engine.dart`](lib/uci_engine.dart) | cutechess-style match manager |
| [`bin/solve_epd.dart`](bin/solve_epd.dart) | EPD tactics benchmark (`--ab` to add the α-β worker) |
| [`bin/ab_solve.dart`](bin/ab_solve.dart) | Standalone alpha-beta tactics check |

Detailed design notes: [`docs/parity-roadmap.md`](docs/parity-roadmap.md) ·
original move-predictor notes: [`docs/README.md`](docs/README.md).

---

## Usage

Place network files under `models/lc0/` (e.g. `744706.pb.gz`,
`t1-256x10-distilled.pb.gz`).

**Play in a GUI (UCI):**
```powershell
dart run bin/uci.dart
# then: uci / isready / position startpos moves e2e4 / go movetime 2000
# options: setoption name Threads value 8 ; setoption name UseAB value true
#          setoption name WeightsFile value models/lc0/t1-256x10-distilled.pb.gz
```

**Tactics benchmark:**
```powershell
dart run bin/solve_epd.dart --movetime 3000 --workers 8 --ab \
  --net models/lc0/t1-256x10-distilled.pb.gz
```

**Engine-vs-engine match (cutechess-style):**
```powershell
# self-play, alpha-beta on vs off
dart run bin/match.dart --e1 "dart run bin/uci.dart" --name1 AB_on  --opt1 Threads=8 --opt1 UseAB=true \
                        --e2 "dart run bin/uci.dart" --name2 AB_off --opt2 Threads=8 --opt2 UseAB=false \
                        --games 6 --movetime 800 --pgn match.pgn

# vs lc0
dart run bin/match.dart --e1 "dart run bin/uci.dart" --name1 AB --opt1 Threads=8 --opt1 UseAB=true \
                        --e2 "C:\path\to\lc0.exe" --name2 lc0 --opt2 WeightsFile=models/lc0/744706.pb.gz \
                        --games 4 --movetime 800 --pgn ab_vs_lc0.pgn
```
Match flags: `--e1/--e2`, `--name1/--name2`, `--opt1/--opt2 K=V` (repeatable),
`--games`, `--movetime` | `--tc base+inc`, `--openings <epd>`, `--pgn`,
`--maxmoves`.

---

## Building

Requires the Dart SDK (`^3.9.3`) and the `dart_pytorch` library. The attention-
body support currently relies on a local checkout of `dart_pytorch` via
`dependency_overrides` in [`pubspec.yaml`](pubspec.yaml) — point that `path:` at
your clone of [KellyKinyama/dart-pytorch](https://github.com/KellyKinyama/dart-pytorch),
then:

```powershell
dart pub get
```

---

## Roadmap

- Close the search-throughput gap to lc0 (GPU backend via WSL; further
  CPU kernel work).
- Deepen MCTS ↔ alpha-beta sharing (feed α-β exact scores into MCTS leaf values).
- Upstream the lc0 attention-body reader into the published `dart_pytorch` and
  drop the local override.
