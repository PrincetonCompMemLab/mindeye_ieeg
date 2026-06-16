"""Tests for IEEGNormalizer: methods x granularities, leakage guardrail, fit/transform."""

import warnings

import numpy as np
import pytest

from ieeg_preproc.transformers import IEEGNormalizer


def _nanmean_ignoring_empty(a, axis):
    """nanmean that tolerates all-NaN (padding) slices without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def test_invalid_method_or_level_raises():
    with pytest.raises(ValueError):
        IEEGNormalizer(method="bogus", level="electrode")
    with pytest.raises(ValueError):
        IEEGNormalizer(method="zscore", level="bogus")


def test_baseline_requires_window():
    with pytest.raises(ValueError):
        IEEGNormalizer(method="baseline", level="trial")  # no baseline_window


def test_global_raises_without_allow_global(mock_X, stim_info):
    norm = IEEGNormalizer(method="zscore", level="global")
    with pytest.raises(RuntimeError):
        norm.fit(mock_X, stim_info=stim_info)


def test_global_allowed_with_flag(mock_X, stim_info):
    norm = IEEGNormalizer(method="zscore", level="global", allow_global=True)
    out = norm.fit_transform(mock_X, stim_info=stim_info)
    assert out.shape == mock_X.shape
    np.testing.assert_allclose(np.nanmean(out), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.nanstd(out), 1.0, atol=1e-9)


def test_mean_center_electrode_zeros_per_channel(mock_X, stim_info):
    out = IEEGNormalizer(method="mean_center", level="electrode").fit_transform(
        mock_X, stim_info=stim_info)
    per_channel = np.nanmean(out, axis=(1, 2, 3))
    np.testing.assert_allclose(per_channel, 0.0, atol=1e-9)


def test_zscore_electrode_unit_variance_per_channel(mock_X, stim_info):
    out = IEEGNormalizer(method="zscore", level="electrode").fit_transform(
        mock_X, stim_info=stim_info)
    np.testing.assert_allclose(np.nanmean(out, axis=(1, 2, 3)), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.nanstd(out, axis=(1, 2, 3)), 1.0, atol=1e-9)


def test_trial_level_mean_center_zeros_each_trial_timecourse(mock_X, stim_info):
    out = IEEGNormalizer(method="mean_center", level="trial").fit_transform(
        mock_X, stim_info=stim_info)
    # Each (channel, condition, rep) timecourse should average to 0 over timepoints.
    per_trial = _nanmean_ignoring_empty(out, axis=2)  # (C, K, R)
    np.testing.assert_allclose(np.nan_to_num(per_trial), 0.0, atol=1e-9)


def test_baseline_trial_zeros_baseline_window(mock_X, stim_info):
    window = slice(0, 3)
    out = IEEGNormalizer(method="baseline", level="trial",
                         baseline_window=window).fit_transform(mock_X, stim_info=stim_info)
    baseline_mean = _nanmean_ignoring_empty(out[:, :, window, :], axis=2)  # (C, K, R)
    np.testing.assert_allclose(np.nan_to_num(baseline_mean), 0.0, atol=1e-9)


def test_fit_transform_separation_uses_fit_stats(mock_X, stim_info):
    # Electrode-level stats learned on train X are applied to a different test X.
    train, test = mock_X[:, :4], mock_X[:, 4:]
    norm = IEEGNormalizer(method="zscore", level="electrode")
    norm.fit(train, stim_info=stim_info)
    out = norm.transform(test, stim_info=stim_info)

    mean = np.nanmean(train, axis=(1, 2, 3), keepdims=True)
    std = np.nanstd(train, axis=(1, 2, 3), keepdims=True)
    expected = (test - mean) / std
    np.testing.assert_allclose(np.nan_to_num(out), np.nan_to_num(expected))


def test_nan_padding_preserved(mock_X, stim_info):
    out = IEEGNormalizer(method="zscore", level="electrode").fit_transform(
        mock_X, stim_info=stim_info)
    np.testing.assert_array_equal(np.isnan(out), np.isnan(mock_X))
