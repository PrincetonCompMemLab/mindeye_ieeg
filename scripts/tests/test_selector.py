"""Tests for NCSNRSelector: slicing-based feature selection on flattened channel x time."""

import numpy as np
import pytest

from ieeg_preproc.transformers import NCSNRSelector
from utils import compute_ncsnr_all_timepoints


def _reference_ncsnr(X, times):
    """NCSNR via the existing util; X is (C, K, T, R) -> util wants (K, C, R, T)."""
    ncsnr, _ = compute_ncsnr_all_timepoints(X.transpose(1, 0, 3, 2), times)
    return ncsnr  # (channels, timepoints)


def test_requires_exactly_one_criterion():
    with pytest.raises(ValueError):
        NCSNRSelector()  # neither threshold nor top_k
    with pytest.raises(ValueError):
        NCSNRSelector(threshold=1.0, top_k=5)  # both


def test_fit_computes_ncsnr_matching_util(mock_X, stim_info, times, dims):
    sel = NCSNRSelector(threshold=0.5, times=times)
    sel.fit(mock_X, stim_info=stim_info)
    expected = _reference_ncsnr(mock_X, times)
    assert sel.ncsnr_.shape == (dims["n_channels"], dims["n_timepoints"])
    np.testing.assert_allclose(sel.ncsnr_, expected)


def test_threshold_mask_and_transform_shape(mock_X, stim_info, times, dims):
    thr = 1.0
    sel = NCSNRSelector(threshold=thr, times=times)
    out = sel.fit_transform(mock_X, stim_info=stim_info)

    expected_mask = _reference_ncsnr(mock_X, times).reshape(-1) >= thr
    np.testing.assert_array_equal(sel.support_, expected_mask)

    n_sel = int(expected_mask.sum())
    K, R = dims["n_conditions"], dims["max_repeats"]
    assert out.shape == (n_sel, K, R)
    assert n_sel > 0  # fixture has high-NCSNR channels above 1.0


def test_transform_values_are_correct_channel_time_slices(mock_X, stim_info, times):
    sel = NCSNRSelector(threshold=1.0, times=times)
    out = sel.fit_transform(mock_X, stim_info=stim_info)
    # Reference: flatten (C, T) features in C-order -> (C*T, K, R), then mask.
    C, K, T, R = mock_X.shape
    flat = mock_X.transpose(0, 2, 1, 3).reshape(C * T, K, R)
    expected = flat[sel.support_]
    np.testing.assert_array_equal(np.nan_to_num(out), np.nan_to_num(expected))


def test_top_k_selects_highest_ncsnr(mock_X, stim_info, times):
    k = 7
    sel = NCSNRSelector(top_k=k, times=times)
    sel.fit(mock_X, stim_info=stim_info)
    assert sel.support_.sum() == k
    flat = _reference_ncsnr(mock_X, times).reshape(-1)
    expected_idx = set(np.argsort(flat)[::-1][:k].tolist())
    assert set(sel.selected_indices_.tolist()) == expected_idx


def test_precomputed_ncsnr_override_skips_computation(mock_X, stim_info, dims):
    C, T = dims["n_channels"], dims["n_timepoints"]
    # A custom matrix where only one feature clears the threshold.
    custom = np.zeros((C, T))
    custom[1, 3] = 5.0
    sel = NCSNRSelector(threshold=1.0, ncsnr=custom)
    out = sel.fit_transform(mock_X, stim_info=stim_info)
    np.testing.assert_allclose(sel.ncsnr_, custom)
    assert sel.support_.sum() == 1
    assert sel.selected_indices_.tolist() == [1 * T + 3]
    assert out.shape[0] == 1


def test_fit_on_train_transform_heldout_uses_train_mask(mock_X, stim_info, times):
    # Split conditions into train/test (sample axis is conditions = axis 1).
    train, test = mock_X[:, :4], mock_X[:, 4:]
    sel = NCSNRSelector(top_k=6, times=times)
    sel.fit(train, stim_info=stim_info)
    out = sel.transform(test, stim_info=stim_info)
    assert out.shape == (6, test.shape[1], test.shape[3])  # (n_sel, K_test, R)
