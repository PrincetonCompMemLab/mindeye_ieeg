"""Tests for DomainFeaturizer: a routing/validation passthrough at the pipeline entry."""

import numpy as np
import pytest

from ieeg_preproc.transformers import DomainFeaturizer


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        DomainFeaturizer(mode="bogus")


@pytest.mark.parametrize("mode", ["voltage", "HFB"])
def test_fit_returns_self(mock_X, stim_info, mode):
    feat = DomainFeaturizer(mode=mode)
    assert feat.fit(mock_X, stim_info=stim_info) is feat


def test_transform_is_passthrough(mock_X, stim_info):
    feat = DomainFeaturizer(mode="HFB")
    out = feat.fit_transform(mock_X, stim_info=stim_info)
    assert out.shape == mock_X.shape
    assert out.dtype == mock_X.dtype
    assert np.array_equal(out, mock_X, equal_nan=True)


def test_non_4d_input_raises(stim_info):
    feat = DomainFeaturizer(mode="HFB")
    with pytest.raises(ValueError):
        feat.fit_transform(np.zeros((4, 6, 10)), stim_info=stim_info)


def test_non_nan_values_must_be_finite(mock_X, stim_info):
    bad = mock_X.copy()
    bad[0, 0, 0, 0] = np.inf  # an inf in a valid (non-padding) slot
    feat = DomainFeaturizer(mode="HFB")
    with pytest.raises(ValueError):
        feat.fit_transform(bad, stim_info=stim_info)
