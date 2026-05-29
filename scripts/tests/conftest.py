"""Shared pytest fixtures for the iEEG preprocessing pipeline tests.

Provides a reproducible mock 4D array ``X`` of shape
``(channels, conditions, timepoints, repetitions)`` together with a matching
``stim_info`` dict, mirroring the contract produced by
``utils.reshape_electrode_data_by_stimuli``. The repetitions axis is padded with
NaNs for conditions with fewer than ``max_repeats`` trials, so tests exercise the
NaN-aware behaviour the real data requires.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Make ``scripts/`` importable so tests (and the package itself) can do
# ``from ieeg_preproc... import ...`` and ``from utils import ...`` exactly as the
# analysis notebooks do.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture
def dims():
    """Small dimensions for fast tests: (channels, conditions, timepoints, repeats)."""
    return dict(n_channels=4, n_conditions=6, n_timepoints=10, max_repeats=3)


@pytest.fixture
def reps_per_condition(dims):
    """Number of valid (non-NaN) repetitions per condition.

    Conditions 2 and 5 are under-sampled so the last repetition slot is NaN
    padding, matching uneven repeats in the real experiment.
    """
    counts = np.full(dims["n_conditions"], dims["max_repeats"], dtype=int)
    counts[2] = dims["max_repeats"] - 1
    counts[5] = dims["max_repeats"] - 1
    return counts


@pytest.fixture
def mock_X(dims, reps_per_condition):
    """Reproducible mock ``X`` of shape (channels, conditions, timepoints, repeats).

    Each (channel, condition) has a deterministic true mean (signal) that also
    varies across timepoints; per-repetition Gaussian noise is added. Signal
    strength increases with channel index and peaks mid-window in time, so NCSNR
    varies across both channels and timepoints. Unused repetition slots are NaN.
    """
    rng = np.random.default_rng(0)
    C, K, T, R = (dims["n_channels"], dims["n_conditions"],
                  dims["n_timepoints"], dims["max_repeats"])

    # Per-condition signal that differs across conditions (the thing NCSNR sees).
    cond_level = np.linspace(-1.0, 1.0, K)  # (K,)
    # Channel gain grows with channel index -> later channels carry more signal.
    chan_gain = np.linspace(0.2, 2.0, C)  # (C,)
    # Temporal envelope peaks in the middle of the window.
    t_env = np.exp(-0.5 * ((np.arange(T) - T / 2) / (T / 4)) ** 2)  # (T,)

    signal = (chan_gain[:, None, None, None]
              * cond_level[None, :, None, None]
              * t_env[None, None, :, None])  # (C, K, T, 1)
    signal = np.broadcast_to(signal, (C, K, T, R)).copy()

    noise = rng.normal(0.0, 0.5, size=(C, K, T, R))
    X = (signal + noise).astype(np.float64)

    # Apply NaN padding for under-sampled conditions.
    for k, n in enumerate(reps_per_condition):
        if n < R:
            X[:, k, :, n:] = np.nan

    return X


@pytest.fixture
def times(dims):
    """Timepoint axis in milliseconds (length = n_timepoints)."""
    return np.linspace(-100.0, 400.0, dims["n_timepoints"])


@pytest.fixture
def rep_resolved_signal():
    """Rep-resolved ``(conditions, features, reps)`` where reps share a per-condition pattern.

    Each condition has a distinct latent pattern; every repetition is that pattern plus
    small noise, so the two half-averages (A, B) are highly correlated and template
    matching should retrieve far above chance. Two conditions are NaN-padded to one rep
    (dropped) and a few to three reps (odd split), mirroring real uneven repeats.
    """
    rng = np.random.default_rng(7)
    K, F, R = 80, 40, 4
    patterns = rng.normal(size=(K, F))
    X = patterns[:, :, None] + 0.1 * rng.normal(size=(K, F, R))
    X[0, :, 1:] = np.nan   # 1 rep -> dropped
    X[1, :, 1:] = np.nan   # 1 rep -> dropped
    X[2, :, 3:] = np.nan   # 3 reps -> odd split
    X[3, :, 3:] = np.nan   # 3 reps -> odd split
    return X.astype(np.float64)


@pytest.fixture
def rep_resolved_noise():
    """Rep-resolved ``(conditions, features, reps)`` of independent noise.

    Repetitions share no per-condition structure, so the half-averages are uncorrelated
    and retrieval should sit near chance.
    """
    rng = np.random.default_rng(8)
    K, F, R = 80, 40, 4
    return rng.normal(size=(K, F, R)).astype(np.float64)


@pytest.fixture
def decoding_dataset():
    """Rep-averaged ``(conditions, features)`` X and target ``y`` that is a linear
    function of X plus small noise, so brain->embedding decoding retrieves far above
    chance on held-out images.
    """
    rng = np.random.default_rng(11)
    K, F, D = 200, 30, 16
    X = rng.normal(size=(K, F))
    W = rng.normal(size=(F, D))
    y = X @ W + 0.1 * rng.normal(size=(K, D))
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def decoding_noise_dataset():
    """X and y with no shared structure -> decoding retrieval sits near chance."""
    rng = np.random.default_rng(12)
    K, F, D = 200, 30, 16
    X = rng.normal(size=(K, F))
    y = rng.normal(size=(K, D))
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def stim_info(dims, reps_per_condition):
    """``stim_info`` dict matching ``mock_X``, with all six contract keys."""
    K = dims["n_conditions"]
    unique_stimuli = [f"stim_{i:03d}" for i in range(K)]
    stimulus_counts = {s: int(n) for s, n in zip(unique_stimuli, reps_per_condition)}

    # Fabricate trial indices consistent with the per-condition counts.
    stimulus_to_trials = {}
    trial_idx = 0
    for s, n in zip(unique_stimuli, reps_per_condition):
        stimulus_to_trials[s] = list(range(trial_idx, trial_idx + int(n)))
        trial_idx += int(n)

    return {
        "unique_stimuli": unique_stimuli,
        "stimulus_counts": stimulus_counts,
        "stimulus_to_trials": stimulus_to_trials,
        "max_repeats": dims["max_repeats"],
        "n_unique_images": K,
        "nsd_id_mapping": {s: i for i, s in enumerate(unique_stimuli)},
    }
