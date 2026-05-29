"""Tests for IEEGPreprocessingPipeline composition and stim_info threading."""

import numpy as np

from ieeg_preproc.base import TransformerStep
from ieeg_preproc.pipeline import IEEGPreprocessingPipeline
from ieeg_preproc.transformers import (
    DomainFeaturizer,
    FinalFormatter,
    IEEGNormalizer,
    NCSNRSelector,
    TemporalSmoother,
)


def _spec_example_steps(times):
    """The canonical SPEC step list: featurize -> select -> normalize -> format."""
    return [
        ("featurization", DomainFeaturizer(mode="HFB")),
        ("selection", NCSNRSelector(threshold=1.0, times=times)),
        ("normalization", IEEGNormalizer(method="zscore", level="electrode")),
        ("formatter", FinalFormatter()),
    ]


class _SpyStep(TransformerStep):
    """Records what stim_info it was handed; passes data through unchanged."""

    def __init__(self):
        self.seen_stim_info = None

    def transform(self, X, stim_info=None):
        self.seen_stim_info = stim_info
        return X


def test_skeleton_pipeline_produces_2d_finite_output(mock_X, stim_info, dims):
    pipe = IEEGPreprocessingPipeline([
        ("featurization", DomainFeaturizer(mode="HFB")),
        ("formatter", FinalFormatter()),
    ])
    out = pipe.fit_transform(mock_X, stim_info=stim_info)
    C, K, T = dims["n_channels"], dims["n_conditions"], dims["n_timepoints"]
    assert out.shape == (K, C * T)
    assert np.isfinite(out).all()


def test_fit_transform_equals_fit_then_transform(mock_X, stim_info):
    steps = lambda: [("featurization", DomainFeaturizer(mode="HFB")),
                     ("formatter", FinalFormatter())]
    a = IEEGPreprocessingPipeline(steps()).fit_transform(mock_X, stim_info=stim_info)
    b = IEEGPreprocessingPipeline(steps()).fit(mock_X, stim_info=stim_info).transform(
        mock_X, stim_info=stim_info)
    np.testing.assert_allclose(a, b)


def test_stim_info_threaded_to_every_step(mock_X, stim_info):
    spy = _SpyStep()
    pipe = IEEGPreprocessingPipeline([
        ("featurization", DomainFeaturizer(mode="HFB")),
        ("spy", spy),
        ("formatter", FinalFormatter()),
    ])
    pipe.fit_transform(mock_X, stim_info=stim_info)
    assert spy.seen_stim_info is stim_info


def test_full_spec_example_pipeline(mock_X, stim_info, times, dims):
    pipe = IEEGPreprocessingPipeline(_spec_example_steps(times))
    out = pipe.fit_transform(mock_X, stim_info=stim_info)
    n_sel = pipe.steps[1][1].support_.sum()
    assert n_sel > 0
    assert out.shape == (dims["n_conditions"], n_sel)
    assert np.isfinite(out).all()


def test_smoothing_before_selection_runs_end_to_end(mock_X, stim_info, times, dims):
    steps = [("featurization", DomainFeaturizer(mode="HFB")),
             ("smoothing", TemporalSmoother(window=3)),
             ("selection", NCSNRSelector(top_k=8, times=times)),
             ("normalization", IEEGNormalizer(method="mean_center", level="electrode")),
             ("formatter", FinalFormatter())]
    out = IEEGPreprocessingPipeline(steps).fit_transform(mock_X, stim_info=stim_info)
    assert out.shape == (dims["n_conditions"], 8)
    assert np.isfinite(out).all()


def test_cv_fit_on_train_transform_heldout(mock_X, stim_info, times):
    # Sample axis is conditions (axis 1): fit on train conditions, apply to held-out.
    train, test = mock_X[:, :4], mock_X[:, 4:]
    pipe = IEEGPreprocessingPipeline(_spec_example_steps(times))
    train_out = pipe.fit_transform(train, stim_info=stim_info)
    test_out = pipe.transform(test, stim_info=stim_info)

    n_sel = pipe.steps[1][1].support_.sum()
    assert train_out.shape == (train.shape[1], n_sel)
    assert test_out.shape == (test.shape[1], n_sel)
    assert np.isfinite(train_out).all() and np.isfinite(test_out).all()
