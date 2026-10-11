#!/usr/bin/env python3
"""One-time, offline conversion of DeepMind searchless-chess Orbax checkpoints
to a flat ``params.npz`` that the pure-Dart loader (lib/searchless.dart) reads.

This is the ONLY Python needed, and only to repackage weights — inference runs
entirely in Dart. The 9M model is already available as an npz on HuggingFace
(dbest-isi/searchless-chess-9M-selfplay); use this for 136M / 270M, which
DeepMind only publishes in Orbax format.

Prereqs (CPU-only is fine):
    pip install "jax[cpu]" orbax-checkpoint numpy

Usage:
    # 1) download + unzip the checkpoint (≈1.1 GB for 270M)
    #    https://storage.googleapis.com/searchless_chess/checkpoints/270M.zip
    #    unzip 270M.zip        -> creates ./270M/<step>/params/...
    # 2) convert
    python scripts/convert_searchless_checkpoint.py --ckpt ./270M --out params_270M.npz

The npz stores the leaves in jax.tree_util flatten order (dict keys sorted
lexicographically), which is exactly what the Dart loader reconstructs.
"""
import argparse
import pathlib

import jax
import numpy as np
import orbax.checkpoint as ocp


def find_params_dir(ckpt_root: pathlib.Path) -> pathlib.Path:
    """Locates the `.../<step>/params` directory inside an unzipped checkpoint."""
    if (ckpt_root / "params").is_dir():
        return ckpt_root / "params"
    steps = [p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()]
    if steps:
        step = max(steps, key=lambda p: int(p.name))
        if (step / "params").is_dir():
            return step / "params"
        return step
    # Fall back to the first subdir that contains a params folder.
    for p in ckpt_root.rglob("params"):
        if p.is_dir():
            return p
    raise FileNotFoundError(f"no params directory found under {ckpt_root}")


def restore(params_dir: pathlib.Path):
    """Restores an Orbax PyTree checkpoint to a nested dict of numpy arrays."""
    params_dir = params_dir.resolve()
    handler = ocp.PyTreeCheckpointHandler()
    try:
        return handler.restore(params_dir)
    except Exception:  # noqa: BLE001 - newer orbax needs an explicit args object
        return handler.restore(params_dir, args=ocp.args.PyTreeRestore())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="unzipped checkpoint root (e.g. ./270M)")
    ap.add_argument("--out", required=True, help="output .npz path")
    args = ap.parse_args()

    params_dir = find_params_dir(pathlib.Path(args.ckpt))
    print(f"restoring {params_dir} ...")
    restored = restore(params_dir)

    leaves, _ = jax.tree_util.tree_flatten(restored)
    arrays = [np.asarray(x, dtype="<f4") for x in leaves]
    print(f"flattened {len(arrays)} arrays; "
          f"first shape {arrays[0].shape}, dim={arrays[0].shape[-1]}, "
          f"inferred layers={(len(arrays) - 6) // 11}")
    np.savez(args.out, *arrays)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
