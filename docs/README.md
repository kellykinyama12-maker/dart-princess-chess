# Chess move predictor — project notes

An autoregressive transformer that predicts chess **moves**, built in Dart on
the [`dart_pytorch`](https://pub.dev/packages/dart_pytorch) tensor library and
trained on **Leela Chess Zero (lc0)** self-play data.

It is a *move-sequence language model*: given the moves played so far (as UCI
tokens), it predicts the next move. This is deliberately different from how lc0
itself works — see [Comparison with the lc0 network](#comparison-with-the-lc0-network).

---

## 1. Pipeline

| Stage | File | What it does |
|---|---|---|
| Download | [`download_lc0_training_data.ps1`](../download_lc0_training_data.ps1) | Pulls recent `test80` V6 training tarballs from `storage.lczero.org`, skipping tiny placeholder files, resumable via BITS. |
| Parse | [`bin/parse_lc0.dart`](../bin/parse_lc0.dart) + [`lib/lc0_v6.dart`](../lib/lc0_v6.dart) | Reads each `.tar` → gzip entries → 8356-byte V6 records → the played move → absolute UCI. Emits one game per line to `data/games.txt`. |
| Tokenize | [`lib/move_vocab.dart`](../lib/move_vocab.dart) | Builds a UCI-move ↔ id vocabulary with `<bos> <eos> <pad> <unk>`. |
| Index table | [`lib/policy_index.dart`](../lib/policy_index.dart) | The authoritative lc0 1858-move index→UCI table (generated from LeelaChessZero's `policy_index.py`). |
| Train | [`bin/train_moves.dart`](../bin/train_moves.dart) | Trains a GPT-style causal transformer with Adam + warmup/cosine LR; saves `model.dptc`, `vocab.json`, `config.json`. |
| Predict | [`bin/predict_move.dart`](../bin/predict_move.dart) | Loads the checkpoint and outputs the greedy next move plus a sampled distribution. |

Artifacts live in `data/` (`games.txt`, `model.dptc`, `vocab.json`,
`config.json`); downloaded tarballs in `training_data/`.

---

## 2. The lc0 V6 data we parse

Each training chunk (one `.gz` per game inside a `.tar`) is a stream of
fixed-size **V6** records, one per position. Layout (little-endian), from
LeelaChessZero's `chunkparser.py`:

| Field | Offset | Type | Notes |
|---|---|---|---|
| `version` | 0 | uint32 | must be `6` |
| `input_format` | 4 | uint32 | `1` = classical; `3+` newer |
| `probabilities[1858]` | 8 | float×1858 | MCTS policy target (we don't use it) |
| `planes[104]` | 7440 | uint64×104 | board + 8-ply history (we don't use it) |
| `side_to_move_or_enpassant` | 8276 | uint8 | side-to-move when `input_format==1` |
| `invariance_info` | 8278 | uint8 | bit 7 = side-to-move for `input_format>=3` |
| **`played_idx`** | **8344** | **uint16** | **index of the move actually played** |

**Record size = 8356 bytes.**

We only read `played_idx`, look it up in the 1858-move table, and convert to
absolute coordinates. lc0 always orients the board from the **side-to-move's**
point of view (Black's board is flipped vertically), so for a Black move we
mirror ranks (`rank → 9 − rank`); files and any promotion suffix are unchanged.
Everything else in the record (planes, full policy distribution, value,
moves-left) is discarded for this project.

Correctness was verified by inspection — the parser reproduces real games, e.g.
a Ruy Lopez `e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 …` with Black's replies properly
mirrored. Castling appears in lc0's king-to-rook encoding (`e1h1` = White O-O).

---

## 3. Our model (dart_pytorch `GPT`)

A decoder-only, GPT-2-style causal language model over move tokens:

```
move ids ──► token embedding (tied) ──► + learned positional embedding
         ──► embedding dropout
         ──► N × pre-LN TransformerBlock:  causal multi-head attention
                                           + residual
                                           + MLP (GELU)
                                           + residual
         ──► final LayerNorm
         ──► linear head (weights tied to the token embedding)
         ──► logits over the move vocabulary (~1900)
```

- **Objective:** next-token cross-entropy over move sequences.
- **Optimizer:** Adam + `LinearWarmupCosineDecay`, global grad-norm clip.
- **Generation:** KV-cached `generate()` with greedy / temperature / top-k.
- **Config used:** vocab ≈ 1908, `maxCtx` 48–64, `embedDim` 128–160,
  `numLayers` 3–4, `numHeads` 4–6, dropout 0, tied weights. ≈ 0.8–2M params.

The model sees only a 1-D sequence of moves. It has **no board, no history
planes, no value head, no search, and no legality constraint** — it can in
principle emit an illegal or out-of-turn move; quality comes purely from how
well it has modelled the move distribution.

---

## 4. Comparison with the lc0 network

The data came from lc0, but the two models solve different problems.

**lc0 is a position evaluator used inside MCTS.** Given a single board (plus a
short history) it outputs a policy over all legal moves and a value, and a
PUCT/MCTS search uses many such evaluations to pick a move.

**Ours is a move-sequence language model.** Given the move history it directly
predicts the next move, with no board and no search.

| Aspect | This project | lc0 |
|---|---|---|
| Task | Predict next move from move history | Evaluate a position → policy + value |
| Input | Sequence of move tokens (UCI) | 112 planes of 8×8 (8-ply history + castling/stm/rule50/aux) |
| Representation | 1-D sequence | 2-D spatial board (64 squares) |
| Architecture | Decoder-only causal transformer (GPT) | ResNet of 3×3 convs + SE (classic nets, e.g. 128 filters × 10 blocks) **or** a transformer **encoder** over the 64 squares with smolgen/attention policy (modern T8x nets) |
| Positional info | Learned positional embedding over sequence index | Conv translation-equivariance (classic) or square position encoding (transformer) |
| Outputs | One next-move distribution over the vocab | Policy over **1858** moves + value **WDL** + moves-left head |
| Training target | Observed next move (cross-entropy) | MCTS visit-count distribution (policy) + game result (value) + plies-left |
| Move selection | Direct sampling/greedy, **no search** | **MCTS (PUCT)** guided by the network |
| Legality | None (pure LM) | Policy masked to legal moves at play time |
| Side to move | Learned implicitly from the sequence | Board explicitly oriented to the side to move |
| Scale | ≈ 0.8–2M params, CPU | millions to 100M+ params, GPU |

**Shared ingredient.** Both can be built from the same primitives — multi-head
attention, LayerNorm, residual MLPs, embeddings — all of which `dart_pytorch`
provides (and which it also uses for its own lc0 *inference* demo,
[`bin/lc0_demo.dart`](https://github.com/KellyKinyama/dart-pytorch/blob/main/bin/lc0_demo.dart)).
The difference is **what the tokens are** (moves here vs. board squares in lc0's
transformer nets) and **the learning objective** (language modelling vs.
policy+value distillation from search).

**Use of the same data, differently.** We train on lc0 self-play games but keep
only the single played move per position, throwing away the rich policy
distribution and value that lc0 learns from. A natural next step toward an
lc0-like model would be to feed the **board planes** (not move history) into a
transformer encoder and add **policy + value** heads trained on the full V6
targets — i.e. reconstruct lc0's objective on top of `dart_pytorch`.

---

## 5. Running it

```powershell
$dart = 'C:\flutter\flutter\bin\dart.bat'

# 1) (one-time) download data
./download_lc0_training_data.ps1

# 2) parse V6 tarballs into move sequences
& $dart run bin/parse_lc0.dart --max-games 30000 --out data/games.txt

# 3) train
& $dart run bin/train_moves.dart --steps 3000 --batch 16 --ctx 64 --embed 160 --layers 4 --heads 5

# 4) predict
& $dart run bin/predict_move.dart e2e4 e7e5 g1f3
```

### Known limitations / next steps
- Lightly trained CPU models sometimes ignore whose turn it is — train longer / bigger.
- No legality masking; pair with the `chess` package to filter to legal moves.
- To approach lc0 quality, switch to a board-plane encoder with policy + value heads and MCTS.
