"""Preprocessing steps for the iEEG pipeline.

All steps operate on the 4D array ``X`` of shape
``(channels, conditions, timepoints, repetitions)`` and thread the ``stim_info`` dict
through unchanged. The repetitions axis may contain NaN padding for under-sampled
conditions, so every step uses NaN-aware operations.

Steps are kept in one module so the available processing choices are visible
side-by-side. They are: ``DomainFeaturizer`` (entry validator/router),
``TemporalSmoother``, ``NCSNRSelector`` (feature selection), ``IEEGNormalizer``, and
``FinalFormatter`` (terminal 2D conversion).
"""

import warnings

import numpy as np

from utils import compute_ncsnr_all_timepoints

from .base import TransformerStep


def _check_finite(X):
    """Validate that X's non-NaN entries are finite (no +/-inf outside NaN padding)."""
    finite_or_nan = np.isfinite(X) | np.isnan(X)
    if not finite_or_nan.all():
        raise ValueError("X contains non-finite values (e.g. +/-inf) outside the NaN "
                         "repetition padding.")


def _check_4d(X):
    """Validate that X is a 4D array whose non-NaN entries are finite."""
    if X.ndim != 4:
        raise ValueError(f"Expected a 4D array (channels, conditions, timepoints, "
                         f"repetitions); got {X.ndim}D with shape {X.shape}.")
    _check_finite(X)


class DomainFeaturizer(TransformerStep):
    """Entry-point routing toggle that validates the target signal type.

    HFB is assumed to be computed upstream, so this stage validates the array and
    passes it through unchanged; ``mode`` records the intended signal type
    (``"voltage"`` or ``"HFB"``) for downstream provenance.
    """

    VALID_MODES = ("voltage", "HFB")

    def __init__(self, mode="HFB"):
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}; got {mode!r}.")
        self.mode = mode

    def transform(self, X, stim_info=None):
        _check_4d(X)
        return X


class TemporalSmoother(TransformerStep):
    """Centered moving-average smoothing (and optional decimation) over timepoints.

    Uses an odd ``window`` so the average is centered. The average is NaN-aware: at the
    window edges it shrinks to the available timepoints, and fully-NaN repetition
    padding stays NaN. ``stride > 1`` then keeps every ``stride``-th smoothed timepoint
    (``ceil(T / stride)`` outputs), which lets this step reproduce a downsampled
    "window/stride" derivative from the raw array rather than baking it into a file.
    Apply this *before* ``NCSNRSelector``, which flattens (and so destroys) the
    timepoint axis.
    """

    def __init__(self, window=1, stride=1):
        if window < 1 or window % 2 == 0:
            raise ValueError(f"window must be a positive odd integer; got {window}.")
        if stride < 1:
            raise ValueError(f"stride must be a positive integer; got {stride}.")
        self.window = window
        self.stride = stride

    def transform(self, X, stim_info=None):
        _check_4d(X)
        if self.window == 1:
            smoothed = X
        else:
            smoothed = self._moving_average(X)
        if self.stride > 1:
            return smoothed[:, :, ::self.stride, :].copy()
        return smoothed.copy()

    def _moving_average(self, X):
        """NaN-aware centered moving average over the time axis via cumulative sums.

        O(N) memory: no per-window expansion (the naive ``sliding_window_view`` +
        ``nanmean`` materialises a ``window``x copy, which OOMs on full-resolution data).
        At the edges the window shrinks to the available timepoints, and timepoints with
        no valid samples (all-NaN repetition padding) stay NaN — same semantics as a
        centered ``nanmean`` over the window.
        """
        half = self.window // 2
        T = X.shape[2]
        mask = ~np.isnan(X)
        # Keep the input dtype (float32) end-to-end; promoting to float64 doubles the
        # peak footprint and OOMs on the full-resolution array.
        Xf = X.copy()
        Xf[~mask] = 0
        csum = np.cumsum(Xf, axis=2)
        ccnt = np.cumsum(mask, axis=2, dtype=X.dtype)

        t = np.arange(T)
        hi = np.minimum(t + half, T - 1)        # inclusive window end
        lo = t - half - 1                       # index just before window start
        lo_clip = np.maximum(lo, 0)
        keep_prefix = (lo >= 0).reshape(1, 1, T, 1)  # zero the prefix at the left edge

        def windowed(cum):
            return np.take(cum, hi, axis=2) - np.take(cum, lo_clip, axis=2) * keep_prefix

        wsum = windowed(csum)
        wcnt = windowed(ccnt)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(wcnt > 0, wsum / wcnt, np.nan)


class NCSNRSelector(TransformerStep):
    """Select channel x timepoint features by noise-ceiling SNR.

    Flattens the spatial and temporal axes together so a 1D boolean mask drops
    sub-threshold (channel, timepoint) pairs, keeping the remaining matrix dense and
    NaN-free along the feature axis. Exactly one of ``threshold`` or ``top_k`` selects
    the criterion.

    Parameters
    ----------
    threshold : float, optional
        Keep features whose NCSNR is >= this value.
    top_k : int, optional
        Keep the ``top_k`` features with the highest NCSNR.
    ncsnr : ndarray, optional
        Precomputed NCSNR matrix of shape (channels, timepoints). If given, ``fit``
        uses it instead of computing NCSNR from the training data.
    times : ndarray, optional
        Timepoint axis passed to the NCSNR computation (used only for progress
        reporting); defaults to ``arange(n_timepoints)``.

    Attributes
    ----------
    ncsnr_ : ndarray (channels, timepoints)
        The NCSNR matrix used for selection.
    support_ : ndarray of bool (channels * timepoints,)
        Boolean mask over flattened (channel, timepoint) features.
    selected_indices_ : ndarray of int
        Flattened indices of the selected features (C-order: ``channel * T + t``).
    """

    def __init__(self, threshold=None, top_k=None, ncsnr=None, times=None):
        if (threshold is None) == (top_k is None):
            raise ValueError("Specify exactly one of `threshold` or `top_k`.")
        self.threshold = threshold
        self.top_k = top_k
        self.ncsnr = ncsnr
        self.times = times

    def fit(self, X, y=None, stim_info=None):
        _check_4d(X)
        if self.ncsnr is not None:
            self.ncsnr_ = np.asarray(self.ncsnr, dtype=float)
        else:
            # util wants (conditions, channels, trials, timepoints); X is (C, K, T, R).
            data_4d = X.transpose(1, 0, 3, 2)
            times = self.times if self.times is not None else np.arange(X.shape[2])
            self.ncsnr_, _ = compute_ncsnr_all_timepoints(data_4d, times)

        flat = self.ncsnr_.reshape(-1)  # C-order: index = channel * T + t
        if self.threshold is not None:
            self.support_ = flat >= self.threshold
        else:
            # Rank with NaNs treated as lowest so they are never selected.
            ranked = np.argsort(np.nan_to_num(flat, nan=-np.inf))[::-1]
            self.support_ = np.zeros(flat.shape, dtype=bool)
            self.support_[ranked[:self.top_k]] = True
        self.selected_indices_ = np.where(self.support_)[0]
        return self

    def transform(self, X, stim_info=None):
        _check_4d(X)
        C, K, T, R = X.shape
        if self.support_.shape[0] != C * T:
            raise ValueError(
                f"Selector was fit on {self.support_.shape[0]} features but X has "
                f"{C * T} (channels x timepoints).")
        # (C, K, T, R) -> (C, T, K, R) -> (C*T, K, R), C-order matches ncsnr flatten.
        flat = X.transpose(0, 2, 1, 3).reshape(C * T, K, R)
        return flat[self.support_]


class IEEGNormalizer(TransformerStep):
    """Center / scale the 4D array at a chosen granularity, with a leakage guardrail.

    ``method`` chooses the transform; ``level`` chooses the granularity at which the
    statistics are pooled (which axes are reduced):

    - ``electrode``: one statistic per channel (reduce conditions, timepoints, reps).
    - ``trial``: one statistic per channel and presentation (reduce timepoints only),
      i.e. each (channel, condition, repetition) timecourse is normalized on its own.
    - ``global``: a single statistic over the whole array.

    ``method``:

    - ``mean_center``: subtract the pooled mean.
    - ``zscore``: subtract the pooled mean and divide by the pooled std.
    - ``baseline``: subtract the pooled mean computed over ``baseline_window`` only
      (a pre-stimulus index range), leaving scale untouched.

    Leakage guardrail: ``level="global"`` pools statistics across the entire array,
    which leaks information if done outside a cross-validation partition. ``fit`` raises
    ``RuntimeError`` for global unless ``allow_global=True`` is set explicitly to
    acknowledge a safe (within-fold) context.

    Statistics are learned in ``fit`` and applied in ``transform`` so train-derived
    scaling can be applied to held-out data without leakage.

    Works on the 4D array ``(channels, conditions, timepoints, repetitions)`` and, for
    ``electrode``/``global`` levels, on the 3D post-selection array
    ``(features, conditions, repetitions)`` (axis 0 is the feature axis in both).
    ``trial`` and ``baseline`` need the timepoint axis and so require the 4D array.
    """

    VALID_METHODS = ("mean_center", "zscore", "baseline")
    VALID_LEVELS = ("trial", "electrode", "global")

    def __init__(self, method="zscore", level="electrode", baseline_window=None,
                 allow_global=False):
        if method not in self.VALID_METHODS:
            raise ValueError(f"method must be one of {self.VALID_METHODS}; got {method!r}.")
        if level not in self.VALID_LEVELS:
            raise ValueError(f"level must be one of {self.VALID_LEVELS}; got {level!r}.")
        if method == "baseline" and baseline_window is None:
            raise ValueError("method='baseline' requires a `baseline_window` "
                             "(a pre-stimulus timepoint index range).")
        self.method = method
        self.level = level
        self.baseline_window = baseline_window
        self.allow_global = allow_global

    def _reduce_axes(self, ndim):
        """Axes to pool statistics over, given the array's dimensionality."""
        if self.level == "global":
            return tuple(range(ndim))
        if self.level == "electrode":
            return tuple(range(1, ndim))  # all but the feature axis (axis 0)
        # trial level normalizes each presentation's timecourse -> needs the time axis.
        if ndim != 4:
            raise ValueError("level='trial' requires the 4D (channels, conditions, "
                             "timepoints, repetitions) array; apply it before selection.")
        return (2,)

    def fit(self, X, y=None, stim_info=None):
        _check_finite(X)
        if X.ndim not in (3, 4):
            raise ValueError(f"Expected a 4D or 3D array; got {X.ndim}D.")
        if self.level == "global" and not self.allow_global:
            raise RuntimeError(
                "level='global' pools statistics across the entire array and leaks "
                "information outside a cross-validation partition. Pass "
                "allow_global=True only when fitting within a proper train partition, "
                "or use level='electrode'/'trial'.")

        axes = self._reduce_axes(X.ndim)
        if self.method == "baseline":
            if X.ndim != 4:
                raise ValueError("method='baseline' requires the 4D array (the "
                                 "baseline_window indexes timepoints).")
            source = X[:, :, self.baseline_window, :]
        else:
            source = X
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN padding slices
            self.center_ = np.nanmean(source, axis=axes, keepdims=True)
            if self.method == "zscore":
                std = np.nanstd(X, axis=axes, keepdims=True)
                self.scale_ = np.where(std == 0, 1.0, std)
            else:
                self.scale_ = np.ones_like(self.center_)
        return self

    def transform(self, X, stim_info=None):
        _check_finite(X)
        return (X - self.center_) / self.scale_


class FinalFormatter(TransformerStep):
    """Terminal step: put conditions first and collapse the feature dims.

    With ``average_reps=True`` (default) repetitions are averaged with ``np.nanmean``
    (ignoring NaN padding) and the output is the 2D ``(conditions, features)`` matrix the
    sklearn/PyTorch modeling blocks consume. With ``average_reps=False`` the repetitions
    axis is kept, giving the rep-resolved ``(conditions, features, repetitions)`` array
    the template-matching eval needs to split halves itself.

    Handles both the 4D array ``(channels, conditions, timepoints, repetitions)`` (feature
    axis = channels x timepoints, C-order) and the 3D post-selection array
    ``(features, conditions, repetitions)``.
    """

    def __init__(self, average_reps=True):
        self.average_reps = average_reps

    def _conditions_first(self, X):
        """Reorder to conditions-first, feature dims flattened (reps last if present)."""
        if X.ndim == 4:
            C, K, T, R = X.shape
            # (C, K, T, R) -> (K, C, T, R) -> (K, C*T, R)
            return X.transpose(1, 0, 2, 3).reshape(K, C * T, R)
        # 3D (F, K, R) -> (K, F, R)
        return X.transpose(1, 0, 2)

    def transform(self, X, stim_info=None):
        if X.ndim not in (3, 4):
            raise ValueError(f"FinalFormatter expects a 4D or 3D array; got {X.ndim}D.")
        rep_resolved = self._conditions_first(X)  # (conditions, features, reps)
        if not self.average_reps:
            return rep_resolved
        # A condition with no valid repetitions for a feature averages to NaN (e.g. a
        # channel missing from some runs); that is surfaced rather than filled.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN repetition slices
            return np.nanmean(rep_resolved, axis=-1)  # (conditions, features)
