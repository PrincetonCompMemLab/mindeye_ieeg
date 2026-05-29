"""Data loading and the NCSNR cache.

Only the rawest array per modality is persisted
(``data/derivatives/<modality>/reshaped_electrode_data.npy``); every preprocessing
variant is produced on the fly by the pipeline. The one derived artifact worth keeping is
NCSNR, which is slow and data-dependent: ``get_ncsnr`` computes it at most once per
*data-signature* (modality + smoothing) and serves later requests from disk, so it is
never recomputed unless the underlying data changes.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from utils import compute_ncsnr_all_timepoints


def load_reshaped(modality_dir, filename="reshaped_electrode_data.npy"):
    """Load the raw 4D array ``(channels, conditions, timepoints, repetitions)`` + stim_info."""
    modality_dir = Path(modality_dir)
    X = np.load(modality_dir / filename)
    with open(modality_dir / "stim_info.pkl", "rb") as f:
        stim_info = pickle.load(f)
    return X, stim_info


def assert_aligned(stim_info, n_conditions, y=None):
    """Guard that conditions ↔ stim_info ↔ targets line up before modeling.

    Silent mis-ordering (a past reshape bug) yields chance-level results that look like a
    modeling failure, so this is a hard check, not a warning.
    """
    n_unique = stim_info["n_unique_images"]
    n_stim = len(stim_info["unique_stimuli"])
    if not (n_conditions == n_unique == n_stim):
        raise ValueError(
            f"conditions misaligned: pipeline produced {n_conditions} conditions but "
            f"stim_info has n_unique_images={n_unique}, len(unique_stimuli)={n_stim}.")
    if y is not None and y.shape[0] != n_conditions:
        raise ValueError(
            f"targets misaligned: y has {y.shape[0]} rows but there are {n_conditions} "
            f"conditions. Embedding rows must be in unique-image order.")


def smoothing_signature(smoothing):
    """Short data-signature string for the cache key.

    ``smoothing`` is ``None`` (raw) or a ``(window, stride)`` pair.
    """
    if smoothing is None:
        return "raw"
    window, stride = smoothing
    return f"smooth-w{window}-s{stride}"


def ncsnr_from_4d(X, times):
    """NCSNR ``(channels, timepoints)`` from pipeline-order ``X = (C, K, T, R)``.

    ``compute_ncsnr_all_timepoints`` wants ``(conditions, channels, trials, timepoints)``.
    """
    ncsnr, _ = compute_ncsnr_all_timepoints(X.transpose(1, 0, 3, 2), times)
    return ncsnr


def get_ncsnr(cache_dir, signature, compute):
    """Load cached NCSNR for ``signature`` or compute (and cache) it on a miss.

    ``compute`` is a zero-argument callable returning a ``(channels, timepoints)`` array;
    it is invoked **only** on a cache miss, so a cache hit costs one ``np.load`` and never
    touches the raw data.
    """
    cache_dir = Path(cache_dir)
    path = cache_dir / f"ncsnr__{signature}.npy"
    if path.exists():
        return np.load(path)

    ncsnr = np.asarray(compute())
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(path, ncsnr)
    _update_manifest(cache_dir, signature, ncsnr.shape)
    return ncsnr


def _update_manifest(cache_dir, signature, shape):
    """Record ``signature -> {shape}`` provenance in ``cache/manifest.json``."""
    manifest_path = Path(cache_dir) / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    manifest[signature] = {"shape": list(shape)}
    manifest_path.write_text(json.dumps(manifest, indent=2))
