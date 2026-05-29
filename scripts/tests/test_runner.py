"""Tests for the comparison runner and the order-alignment guard."""

import numpy as np
import pytest

from ieeg_preproc.data import assert_aligned
from ieeg_preproc.evals import TemplateMatchingEval, run_comparison
from ieeg_preproc.pipeline import IEEGPreprocessingPipeline
from ieeg_preproc.transformers import DomainFeaturizer, FinalFormatter


def _stim_info(n):
    stims = [f"stim_{i:03d}" for i in range(n)]
    return {
        "unique_stimuli": stims,
        "stimulus_counts": {s: 4 for s in stims},
        "stimulus_to_trials": {s: [0, 1, 2, 3] for s in stims},
        "max_repeats": 4,
        "n_unique_images": n,
        "nsd_id_mapping": {s: i for i, s in enumerate(stims)},
    }


def _make_4d_signal(seed, C=3, K=40, T=4, R=4):
    """4D (channels, conditions, timepoints, reps) with reps sharing a per-condition pattern."""
    rng = np.random.default_rng(seed)
    pattern = rng.normal(size=(C, K, T))
    X = pattern[:, :, :, None] + 0.1 * rng.normal(size=(C, K, T, R))
    return X.astype(np.float64)


def test_assert_aligned_passes_when_consistent():
    si = _stim_info(40)
    assert_aligned(si, 40)                       # no y
    assert_aligned(si, 40, y=np.zeros((40, 5)))  # y rows match


def test_assert_aligned_raises_on_condition_mismatch():
    si = _stim_info(40)
    with pytest.raises(ValueError):
        assert_aligned(si, 39)


def test_assert_aligned_raises_on_y_mismatch():
    si = _stim_info(40)
    with pytest.raises(ValueError):
        assert_aligned(si, 40, y=np.zeros((39, 5)))


def test_run_comparison_over_datasets():
    datasets = {
        "HFB": (_make_4d_signal(1), _stim_info(40)),
        "voltage": (_make_4d_signal(2), _stim_info(40)),
    }

    def make_pipeline(name):
        return IEEGPreprocessingPipeline([
            ("featurization", DomainFeaturizer(mode="HFB" if name == "HFB" else "voltage")),
            ("formatter", FinalFormatter(average_reps=False)),
        ])

    ev = TemplateMatchingEval(top_ks=(1, 5), test_size=0.25, seed=1, verbose=False)
    results = run_comparison(datasets, make_pipeline, ev)

    assert set(results) == {"HFB", "voltage"}
    for name in datasets:
        assert "top_k_correlation_test" in results[name]
        assert results[name]["n_train"] + results[name]["n_test"] == 40
