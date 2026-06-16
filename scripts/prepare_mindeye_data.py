#!/usr/bin/env python
"""Prepare MindEye2-ready iEEG + bigG CLIP tensors (rep-averaged conditions).

Reuses the eval-layer data path verbatim: loads the raw reshaped array for one modality,
runs the same pipeline as ``clip_decode_compare`` (smooth -> top-NCSNR select -> rep-average),
aligns to the token-level bigG CLIP target, drops NaN feature columns (channels absent from
some runs), splits train/test exactly like the evals (``test_size=0.2, seed=42`` over images),
and saves four ``.pt`` files consumed by ``iEEGDataset``. NCSNR is read from the cache, never
recomputed.

Run:
  uv run python scripts/prepare_mindeye_data.py --modality hfb --window 25 --stride 10 --top-k 2000
"""

import argparse
import warnings

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ieeg_preproc.data import assert_aligned
from template_matching_compare import DERIV, MODES, load_modalities
from clip_decode_compare import make_pipeline_factory

BIGG_PATH = DERIV / "images_clipemb_bigG"   # (2900, 256, 1664) fp16, unique-image order
OUT_DIR = DERIV / "mindeye"
OUT_DIR_ST = DERIV / "mindeye_single_trial"  # Tier-1 single-trial variant


def averaged_and_keepmask(X3d):
    """From rep-resolved ``(conditions, features, reps)`` -> ``(X_avg, keep)``.

    ``X_avg`` is the rep-averaged matrix (``np.nanmean`` over reps, == the ``average_reps=True``
    output). ``keep`` is the feature-column mask dropping channels NaN for any image -- the
    *same* rule as the averaged path / ``CLIPDecodingEval`` -- so single-trial train and
    averaged test end up with identical columns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN repetition slices
        X_avg = np.nanmean(X3d, axis=2)
    keep = ~np.isnan(X_avg).any(axis=0)
    return X_avg, keep


def expand_train_trials(X3d, image_ids, keep):
    """Expand the given images' valid repetitions to one ``(x_trial, image)`` row each.

    Padded repetitions (all-NaN slices for under-sampled images) are dropped; only the
    ``image_ids`` requested contribute (so passing the train split cannot leak test trials).
    Returns ``(X_trials (n, n_keep) float32, image_idx (n,) int64)``.
    """
    rows, imgs = [], []
    for c in image_ids:
        feats = X3d[c][keep, :]                 # (n_keep, reps)
        for r in range(feats.shape[1]):
            v = feats[:, r]
            if not np.isnan(v).any():           # skip padded / partial repetitions
                rows.append(v)
                imgs.append(int(c))
    return np.stack(rows).astype(np.float32), np.asarray(imgs, dtype=np.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modality", choices=tuple(MODES), default="hfb")
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None,
                   help="subset to the first N images (lighter smoke run; separate NCSNR cache).")
    args = p.parse_args()

    datasets, ncsnr_by_name = load_modalities(args.window, args.stride, args.max_images)
    if args.modality not in datasets:
        raise SystemExit(f"modality {args.modality!r} has no reshaped array; found {list(datasets)}")
    X4d, stim_info = datasets[args.modality]

    # Same pipeline as clip_decode_compare (rep-averaged 2D output).
    make_pipeline = make_pipeline_factory(args.window, args.stride, args.top_k, ncsnr_by_name)
    X = make_pipeline(args.modality).fit_transform(X4d, stim_info=stim_info)   # (N, F)
    N = X.shape[0]

    # bigG token-level CLIP target, already in unique-image order (== conditions axis).
    y3d = torch.load(BIGG_PATH, weights_only=True)                              # (2900, 256, 1664)
    if args.max_images is not None:
        y3d = y3d[:args.max_images]
    assert_aligned(stim_info, N, y=y3d.reshape(y3d.shape[0], -1).numpy())

    # Drop feature columns NaN for any kept image -- same rule as CLIPDecodingEval.
    bad = np.isnan(X).any(axis=0)
    if bad.any():
        print(f"[nan] dropping {int(bad.sum())}/{X.shape[1]} feature(s) with NaNs "
              f"(channels missing from some runs)")
        X = X[:, ~bad]
    print(f"[shape] X(iEEG)={X.shape}  y(bigG)={tuple(y3d.shape)}")

    idx = np.arange(N)
    train, test = train_test_split(idx, test_size=args.test_size,
                                   random_state=args.seed, shuffle=True)
    train_t, test_t = torch.from_numpy(train), torch.from_numpy(test)
    print(f"[split] train={len(train)} test={len(test)} (test_size={args.test_size}, seed={args.seed})")

    Xt = torch.from_numpy(np.ascontiguousarray(X)).float()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(Xt[train_t], OUT_DIR / "ieeg_train.pt")
    torch.save(Xt[test_t], OUT_DIR / "ieeg_test.pt")
    torch.save(y3d[train_t], OUT_DIR / "clip_train.pt")
    torch.save(y3d[test_t], OUT_DIR / "clip_test.pt")
    torch.save({"train": train_t, "test": test_t, "modality": args.modality,
                "window": args.window, "stride": args.stride, "top_k": args.top_k,
                "feature_dim": int(Xt.shape[1]), "max_images": args.max_images},
               OUT_DIR / "split_meta.pt")
    print(f"[saved] {OUT_DIR}/")
    print(f"        ieeg_train.pt {tuple(Xt[train_t].shape)}  ieeg_test.pt {tuple(Xt[test_t].shape)}")
    print(f"        clip_train.pt {tuple(y3d[train_t].shape)}  clip_test.pt {tuple(y3d[test_t].shape)}")


if __name__ == "__main__":
    main()
