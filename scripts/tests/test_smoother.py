"""Tests for TemporalSmoother: NaN-aware centered moving average over timepoints."""

import numpy as np
import pytest

from ieeg_preproc.transformers import TemporalSmoother


def test_even_or_nonpositive_window_raises():
    with pytest.raises(ValueError):
        TemporalSmoother(window=4)  # even
    with pytest.raises(ValueError):
        TemporalSmoother(window=0)


def test_window_1_is_identity(mock_X, stim_info):
    out = TemporalSmoother(window=1).fit_transform(mock_X, stim_info=stim_info)
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(mock_X))
    assert out.shape == mock_X.shape


def test_shape_and_nan_padding_preserved(mock_X, stim_info):
    out = TemporalSmoother(window=3).fit_transform(mock_X, stim_info=stim_info)
    assert out.shape == mock_X.shape
    np.testing.assert_array_equal(np.isnan(out), np.isnan(mock_X))


def test_interior_point_is_mean_of_neighbors():
    # One channel/condition/rep, ramp over time: [0,1,2,3,4].
    X = np.arange(5, dtype=float).reshape(1, 1, 5, 1)
    out = TemporalSmoother(window=3).transform(X)
    # Interior: centered mean of 3; edges: partial window (NaN-aware).
    expected = np.array([0.5, 1.0, 2.0, 3.0, 3.5]).reshape(1, 1, 5, 1)
    np.testing.assert_allclose(out, expected)


def test_smoothing_reduces_variance(mock_X, stim_info):
    out = TemporalSmoother(window=5).fit_transform(mock_X, stim_info=stim_info)
    # Averaging neighbors should not increase the temporal variance of valid trials.
    assert np.nanvar(out) <= np.nanvar(mock_X)


def test_nonpositive_stride_raises():
    with pytest.raises(ValueError):
        TemporalSmoother(window=3, stride=0)


def test_stride_default_is_one(mock_X, stim_info):
    # Default stride must leave the timepoint axis length unchanged.
    out = TemporalSmoother(window=3, stride=1).fit_transform(mock_X, stim_info=stim_info)
    assert out.shape == mock_X.shape


def test_stride_decimates_timepoints(mock_X, stim_info, dims):
    T = dims["n_timepoints"]
    stride = 3
    out = TemporalSmoother(window=3, stride=stride).fit_transform(mock_X, stim_info=stim_info)
    C, K, _, R = mock_X.shape
    expected_T = -(-T // stride)  # ceil(T / stride)
    assert out.shape == (C, K, expected_T, R)


def test_stride_takes_every_sth_after_smoothing():
    # ramp [0,1,2,3,4], window=3 -> smoothed [0.5,1,2,3,3.5]; stride 2 -> idx 0,2,4.
    X = np.arange(5, dtype=float).reshape(1, 1, 5, 1)
    out = TemporalSmoother(window=3, stride=2).transform(X)
    expected = np.array([0.5, 2.0, 3.5]).reshape(1, 1, 3, 1)
    np.testing.assert_allclose(out, expected)


def test_stride_with_window_1_just_decimates():
    X = np.arange(6, dtype=float).reshape(1, 1, 6, 1)
    out = TemporalSmoother(window=1, stride=2).transform(X)
    expected = np.array([0.0, 2.0, 4.0]).reshape(1, 1, 3, 1)
    np.testing.assert_allclose(out, expected)


def test_stride_preserves_rep_nan_padding(mock_X, stim_info):
    out = TemporalSmoother(window=3, stride=2).fit_transform(mock_X, stim_info=stim_info)
    # A fully-padded repetition slot stays fully NaN after smooth + decimate.
    padded = np.all(np.isnan(mock_X[:, 5, :, -1]))  # condition 5 last rep is padding
    assert padded
    assert np.all(np.isnan(out[:, 5, :, -1]))
