"""Tests for the template-matching denoiser proxy eval.

Half-split each image's reps into A and B, fit RidgeCV A->B on a train subset of images,
and measure top-K retrieval on held-out images vs. a no-model baseline and chance.
"""

import numpy as np
import pytest

from ieeg_preproc.evals import TemplateMatchingEval, half_split


def test_half_split_shapes_and_drops_singletons(rep_resolved_signal):
    rng = np.random.default_rng(0)
    A, B, kept_idx = half_split(rep_resolved_signal, rng)
    K = rep_resolved_signal.shape[0]
    assert len(kept_idx) == K - 2          # two 1-rep conditions dropped
    assert A.shape == B.shape == (K - 2, rep_resolved_signal.shape[1])
    assert np.isfinite(A).all() and np.isfinite(B).all()
    assert 0 not in kept_idx and 1 not in kept_idx


def test_half_split_reproducible(rep_resolved_signal):
    a1, b1, k1 = half_split(rep_resolved_signal, np.random.default_rng(3))
    a2, b2, k2 = half_split(rep_resolved_signal, np.random.default_rng(3))
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    np.testing.assert_array_equal(k1, k2)


def test_run_keys_ranges_and_split_sizes(rep_resolved_signal):
    ev = TemplateMatchingEval(top_ks=(1, 5, 10), test_size=0.25, seed=1)
    res = ev.run(rep_resolved_signal)

    for key in ("n_train", "n_test", "top_k_correlation_test", "top_k_neg_mse_test",
                "top_k_baseline_test", "chance", "top_k_correlation_train"):
        assert key in res

    n_kept = rep_resolved_signal.shape[0] - 2
    assert res["n_train"] + res["n_test"] == n_kept
    for k, v in res["top_k_correlation_test"].items():
        assert 0.0 <= v <= 1.0
    assert res["chance"][1] == pytest.approx(1 / res["n_test"])


def test_run_topk_monotonic_in_k(rep_resolved_signal):
    ev = TemplateMatchingEval(top_ks=(1, 5, 10), test_size=0.25, seed=1)
    res = ev.run(rep_resolved_signal)
    accs = [res["top_k_correlation_test"][k] for k in (1, 5, 10)]
    assert accs == sorted(accs)


def test_signal_retrieval_beats_chance(rep_resolved_signal):
    ev = TemplateMatchingEval(top_ks=(1, 5), test_size=0.25, seed=1)
    res = ev.run(rep_resolved_signal)
    chance1 = res["chance"][1]
    # Strong shared structure -> both the model and the no-model baseline retrieve well.
    assert res["top_k_correlation_test"][1] > 0.5
    assert res["top_k_correlation_test"][1] > 5 * chance1
    assert res["top_k_baseline_test"][1] > 5 * chance1


def test_noise_retrieval_near_chance(rep_resolved_noise):
    ev = TemplateMatchingEval(top_ks=(1, 5), test_size=0.25, seed=1)
    res = ev.run(rep_resolved_noise)
    # No shared structure -> top-1 stays low (loose bound for sampling noise).
    assert res["top_k_correlation_test"][1] < 0.3
