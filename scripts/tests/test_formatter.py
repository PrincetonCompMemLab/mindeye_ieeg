"""Tests for FinalFormatter: average over repetitions, collapse to 2D (conditions, features)."""

import numpy as np

from ieeg_preproc.transformers import FinalFormatter


def test_4d_input_collapses_to_conditions_by_features(mock_X, stim_info, dims):
    out = FinalFormatter().fit_transform(mock_X, stim_info=stim_info)
    C, K, T = dims["n_channels"], dims["n_conditions"], dims["n_timepoints"]
    assert out.shape == (K, C * T)
    assert out.ndim == 2
    assert np.isfinite(out).all()


def test_4d_values_match_nanmean_over_reps(mock_X, stim_info, dims):
    out = FinalFormatter().fit_transform(mock_X, stim_info=stim_info)
    # Reference: nanmean over reps -> (C, K, T) -> (K, C, T) -> (K, C*T)
    averaged = np.nanmean(mock_X, axis=-1)              # (C, K, T)
    expected = averaged.transpose(1, 0, 2).reshape(dims["n_conditions"], -1)
    np.testing.assert_allclose(out, expected)


def test_3d_selected_input_collapses_to_conditions_by_features():
    # Post-selection shape: (features, conditions, repetitions) with NaN padding.
    rng = np.random.default_rng(1)
    F, K, R = 5, 6, 3
    X = rng.normal(size=(F, K, R))
    X[:, 2, 2] = np.nan  # under-sampled condition
    out = FinalFormatter().fit_transform(X)
    assert out.shape == (K, F)
    expected = np.nanmean(X, axis=-1).T  # (K, F)
    np.testing.assert_allclose(out, expected)
    assert np.isfinite(out).all()


def test_average_reps_false_keeps_reps_4d(mock_X, stim_info, dims):
    # Rep-resolved output for the template-matching eval: (conditions, features, reps).
    out = FinalFormatter(average_reps=False).fit_transform(mock_X, stim_info=stim_info)
    C, K, T, R = mock_X.shape
    assert out.shape == (K, C * T, R)
    # Feature layout matches the averaged path's (C, T) C-order flatten, per rep.
    expected = mock_X.transpose(1, 0, 2, 3).reshape(K, C * T, R)
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(expected))
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))


def test_average_reps_false_keeps_reps_3d():
    # Post-selection (features, conditions, reps) -> (conditions, features, reps).
    rng = np.random.default_rng(3)
    F, K, R = 5, 6, 3
    X = rng.normal(size=(F, K, R))
    X[:, 2, 2] = np.nan
    out = FinalFormatter(average_reps=False).fit_transform(X)
    assert out.shape == (K, F, R)
    expected = X.transpose(1, 0, 2)
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(expected))
