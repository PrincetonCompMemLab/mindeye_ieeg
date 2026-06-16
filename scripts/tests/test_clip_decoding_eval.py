"""Tests for the CLIP-decoding eval.

Fit RidgeCV brain->embedding on a train subset of images and measure cosine top-K
retrieval of the predicted embedding against the held-out true embeddings vs. chance.
"""

import numpy as np
import pytest

from ieeg_preproc.evals import CLIPDecodingEval


def test_run_requires_targets(decoding_dataset):
    X, _ = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5), seed=1, verbose=False)
    with pytest.raises(ValueError):
        ev.run(X, y=None)


def test_run_keys_ranges_and_split_sizes(decoding_dataset):
    X, y = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5, 10), test_size=0.25, seed=1, verbose=False)
    res = ev.run(X, y=y)

    for key in ("n_train", "n_test", "top_k_retrieval_test", "top_k_retrieval_train",
                "chance", "test_r2"):
        assert key in res

    assert res["n_train"] + res["n_test"] == X.shape[0]
    for v in res["top_k_retrieval_test"].values():
        assert 0.0 <= v <= 1.0
    assert res["chance"][1] == pytest.approx(1 / res["n_test"])


def test_run_topk_monotonic_in_k(decoding_dataset):
    X, y = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5, 10), test_size=0.25, seed=1, verbose=False)
    res = ev.run(X, y=y)
    accs = [res["top_k_retrieval_test"][k] for k in (1, 5, 10)]
    assert accs == sorted(accs)


def test_signal_decoding_beats_chance(decoding_dataset):
    X, y = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5), test_size=0.25, seed=1, verbose=False)
    res = ev.run(X, y=y)
    assert res["top_k_retrieval_test"][1] > 0.5
    assert res["top_k_retrieval_test"][1] > 5 * res["chance"][1]


def test_noise_decoding_near_chance(decoding_noise_dataset):
    X, y = decoding_noise_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5), test_size=0.25, seed=1, verbose=False)
    res = ev.run(X, y=y)
    assert res["top_k_retrieval_test"][1] < 0.2


def test_target_pca_reduces_and_decodes(decoding_dataset):
    # A high-dim target reduced via target_pca should still decode above chance, and
    # retrieval happens in the reduced space (the path used for bigG token embeddings).
    X, y = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1, 5), test_size=0.25, target_pca=8, seed=1,
                          verbose=False)
    res = ev.run(X, y=y)
    assert "target_pca_components" in res and res["target_pca_components"] == 8
    assert res["top_k_retrieval_test"][1] > 5 * res["chance"][1]


def test_target_pca_variance_threshold(decoding_dataset):
    # A float target_pca is treated as a retained-variance threshold.
    X, y = decoding_dataset
    ev = CLIPDecodingEval(top_ks=(1,), test_size=0.25, target_pca=0.9, seed=1,
                          verbose=False)
    res = ev.run(X, y=y)
    assert 1 <= res["target_pca_components"] <= y.shape[1]


def test_no_target_pca_by_default(decoding_dataset):
    X, y = decoding_dataset
    res = CLIPDecodingEval(top_ks=(1,), seed=1, verbose=False).run(X, y=y)
    assert res["target_pca_components"] is None


def test_drops_nan_feature_columns(decoding_dataset):
    X, y = decoding_dataset
    X = X.copy()
    X[:, 0] = np.nan  # a channel missing from every run
    ev = CLIPDecodingEval(top_ks=(1,), test_size=0.25, seed=1, verbose=False)
    res = ev.run(X, y=y)  # must not raise; NaN column dropped
    assert np.isfinite(res["top_k_retrieval_test"][1])
