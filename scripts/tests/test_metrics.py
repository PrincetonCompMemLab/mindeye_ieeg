"""Tests for retrieval metrics: similarity matrices, ranks, top-K accuracy.

These are the pure functions the eval layer (template matching, CLIP decoding) builds
on, ported from the exploratory ``split_half.py``.
"""

import numpy as np
import pytest

from ieeg_preproc.metrics import (
    correlation_matrix,
    cosine_matrix,
    neg_mse_matrix,
    ranks_from_sim,
    topk_accuracy_from_ranks,
    topk_from_similarity,
)


def test_correlation_matrix_shape_and_self_diagonal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 8))
    Y = rng.normal(size=(7, 8))
    out = correlation_matrix(X, Y)
    assert out.shape == (5, 7)

    self_corr = correlation_matrix(X, X)
    np.testing.assert_allclose(np.diag(self_corr), 1.0, atol=1e-6)


def test_correlation_matrix_anticorrelated_is_minus_one():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    y = -x
    np.testing.assert_allclose(correlation_matrix(x, y), -1.0, atol=1e-6)


def test_cosine_matrix_shape_and_self_diagonal():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(5, 8))
    Y = rng.normal(size=(7, 8))
    out = cosine_matrix(X, Y)
    assert out.shape == (5, 7)
    np.testing.assert_allclose(np.diag(cosine_matrix(X, X)), 1.0, atol=1e-6)


def test_cosine_matrix_ignores_magnitude_not_direction():
    # cosine is scale-invariant: scaling a row leaves the similarity unchanged.
    x = np.array([[1.0, 0.0, 0.0]])
    y = np.array([[5.0, 0.0, 0.0]])
    np.testing.assert_allclose(cosine_matrix(x, y), 1.0, atol=1e-6)
    # but unlike correlation, cosine does not center -> orthogonal vectors give 0.
    np.testing.assert_allclose(cosine_matrix(np.array([[1.0, 0.0]]),
                                             np.array([[0.0, 1.0]])), 0.0, atol=1e-6)


def test_neg_mse_matrix_self_diagonal_is_zero():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(6, 10))
    out = neg_mse_matrix(X, X)
    assert out.shape == (6, 6)
    np.testing.assert_allclose(np.diag(out), 0.0, atol=1e-9)


def test_neg_mse_matrix_known_constant_offset():
    # rows differ by a constant c -> per-feature MSE = c^2, so neg_mse = -c^2.
    x = np.zeros((1, 4))
    y = np.full((1, 4), 3.0)
    np.testing.assert_allclose(neg_mse_matrix(x, y), -9.0, atol=1e-9)


def test_ranks_from_sim_perfect_diagonal_all_zero():
    # diagonal strictly the largest in each row -> every true target is rank 0.
    sim = np.array([
        [1.0, 0.1, 0.2],
        [0.3, 1.0, 0.0],
        [0.2, 0.1, 1.0],
    ])
    np.testing.assert_array_equal(ranks_from_sim(sim, higher_is_better=True), [0, 0, 0])


def test_ranks_from_sim_counts_off_target_winners():
    # row 0: two off-target entries beat the diagonal (0.5) -> rank 2.
    sim = np.array([
        [0.5, 0.9, 0.7],
        [0.1, 1.0, 0.2],
    ])
    ranks = ranks_from_sim(sim, higher_is_better=True)
    assert ranks[0] == 2
    assert ranks[1] == 0


def test_ranks_from_sim_lower_is_better():
    # distance matrix: smaller diagonal = better. Diagonal smallest -> rank 0.
    dist = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 0.0, 1.0],
    ])
    np.testing.assert_array_equal(ranks_from_sim(dist, higher_is_better=False), [0, 0])


def test_topk_accuracy_from_ranks():
    ranks = np.array([0, 1, 5])
    acc = topk_accuracy_from_ranks(ranks, [1, 2, 10])
    assert acc[1] == pytest.approx(1 / 3)
    assert acc[2] == pytest.approx(2 / 3)
    assert acc[10] == pytest.approx(1.0)


def test_topk_from_similarity_perfect_retrieval():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 12))
    acc = topk_from_similarity(correlation_matrix(X, X), [1, 5], higher_is_better=True)
    assert acc[1] == pytest.approx(1.0)
    assert acc[5] == pytest.approx(1.0)
