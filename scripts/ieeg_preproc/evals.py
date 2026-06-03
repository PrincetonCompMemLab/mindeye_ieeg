"""Pluggable analysis/eval layer over the iEEG pipeline.

An ``Evaluation`` consumes a pipeline representation and returns a headline metrics dict,
so different *processing choices* can be compared with one runner. The first eval is the
template-matching denoiser proxy; ``CLIPDecodingEval`` (image-embedding decoding) follows
the same seam.
"""

import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import assert_aligned
from .metrics import (
    correlation_matrix, cosine_matrix, neg_mse_matrix, topk_from_similarity,
)


def run_comparison(datasets, make_pipeline, evaluation, y=None, seed=42):
    """Run one ``Evaluation`` across named datasets and collect per-dataset metrics.

    Parameters
    ----------
    datasets : dict ``{name: (X4d, stim_info)}``
        Each ``X4d`` is the raw ``(channels, conditions, timepoints, repetitions)`` array.
    make_pipeline : callable ``name -> IEEGPreprocessingPipeline``
        Builds the (fixed) processing pipeline for a dataset; the swept knob is encoded
        per name (e.g. voltage vs HFB).
    evaluation : Evaluation
    y : ndarray, optional
        Shared targets (e.g. CLIP embeddings) passed to each ``run``; ``None`` for template
        matching. Alignment to the conditions axis is asserted.

    Returns ``{name: metrics_dict}``.
    """
    results = {}
    for name, (X, stim_info) in datasets.items():
        repr_ = make_pipeline(name).fit_transform(X, stim_info=stim_info)
        assert_aligned(stim_info, repr_.shape[0], y=y)
        results[name] = evaluation.run(repr_, stim_info=stim_info, y=y,
                                       rng=np.random.default_rng(seed))
    return results


class Evaluation:
    """Interface: turn a pipeline representation into a metrics dict.

    Implementations define ``run``. Template matching uses the rep-resolved
    ``(conditions, features, reps)`` array and ignores ``y``; decoding evals use the
    rep-averaged ``(conditions, features)`` array plus targets ``y``.
    """

    def run(self, X, stim_info=None, y=None, rng=None):
        raise NotImplementedError


def half_split(X, rng):
    """Split each image's repetitions into two balanced half-averages.

    Parameters
    ----------
    X : ndarray (conditions, features, repetitions)
        Rep-resolved features, NaN-padded for under-sampled conditions.
    rng : numpy.random.Generator

    Returns
    -------
    A, B : ndarray (n_kept, features)
        Half-averages for the conditions with >= 2 valid repetitions.
    kept_idx : ndarray (n_kept,)
        Indices into the original conditions axis.

    Odd rep counts split ``n//2`` vs ``n - n//2`` with a per-image coin flip for which
    half gets the extra trial, so the mix of 1-trial and 2-trial averages is balanced
    across A and B on average.
    """
    K, F, R = X.shape
    valid = ~np.all(np.isnan(X), axis=1)          # (K, R): non-padding rep slots
    rep_counts = valid.sum(axis=1)
    kept_idx = np.where(rep_counts >= 2)[0]

    A = np.empty((len(kept_idx), F), dtype=float)
    B = np.empty((len(kept_idx), F), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # missing-channel all-NaN cols
        for i, k in enumerate(kept_idx):
            slots = np.where(valid[k])[0]
            perm = rng.permutation(slots)
            n = len(perm)
            cut = n // 2 + (1 if (n % 2 and rng.random() < 0.5) else 0)
            A[i] = np.nanmean(X[k][:, perm[:cut]], axis=1)
            B[i] = np.nanmean(X[k][:, perm[cut:]], axis=1)
    return A, B, kept_idx


def _drop_nan_features(A, B):
    """Drop feature columns that are NaN in any kept image (e.g. channels absent in some runs)."""
    bad = np.isnan(A).any(axis=0) | np.isnan(B).any(axis=0)
    if bad.any():
        warnings.warn(f"dropping {int(bad.sum())} feature(s) with NaNs "
                      f"(channels missing from some runs).", stacklevel=2)
        A, B = A[:, ~bad], B[:, ~bad]
    return A, B


class _FeaturePrep:
    """Z-score + PCA fit on a **train** feature matrix, applied to any split.

    PCA component count is fixed (``n_pcs``) or chosen to retain ``pca_var_threshold``
    of variance. Fitting on train only keeps the test split leakage-free.
    """

    def __init__(self, zscore=True, do_pca=True, n_pcs=None, pca_var_threshold=0.98):
        self.zscore = zscore
        self.do_pca = do_pca
        self.n_pcs = n_pcs
        self.pca_var_threshold = pca_var_threshold

    def fit(self, M_train):
        self.scaler_ = StandardScaler().fit(M_train) if self.zscore else None
        if self.do_pca:
            scaled = self.scaler_.transform(M_train) if self.zscore else M_train
            n_pcs = self.n_pcs
            if n_pcs is None:
                cumvar = np.cumsum(PCA().fit(scaled).explained_variance_ratio_)
                n_pcs = int(np.searchsorted(cumvar, self.pca_var_threshold) + 1)
            self.pca_ = PCA(n_components=n_pcs).fit(scaled)
        else:
            self.pca_ = None
        return self

    def transform(self, M):
        if self.scaler_ is not None:
            M = self.scaler_.transform(M)
        if self.pca_ is not None:
            M = self.pca_.transform(M)
        return M


class _PooledPrep(_FeaturePrep):
    """``_FeaturePrep`` fit on the **pooled** (A, B) train halves (shared subspace)."""

    def fit(self, A_train, B_train):
        return super().fit(np.vstack([A_train, B_train]))


class TemplateMatchingEval(Evaluation):
    """RidgeCV denoiser proxy: fit A->B on train images, retrieve on held-out images.

    Returns top-K retrieval (Pearson and -MSE) on the test (and train) split, the no-model
    baseline (half A correlated directly with half B), and chance ``K / n``.
    """

    def __init__(self, top_ks=(1, 5, 20, 100), test_size=0.2, zscore=True, do_pca=True,
                 n_pcs=None, pca_var_threshold=0.98, ridge_alphas=None, seed=42,
                 verbose=True):
        self.top_ks = tuple(top_ks)
        self.test_size = test_size
        self.zscore = zscore
        self.do_pca = do_pca
        self.n_pcs = n_pcs
        self.pca_var_threshold = pca_var_threshold
        self.ridge_alphas = (np.logspace(-4, 10, 80) if ridge_alphas is None
                             else np.asarray(ridge_alphas))
        self.seed = seed
        self.verbose = verbose

    def run(self, X, stim_info=None, y=None, rng=None):
        if rng is None:
            rng = np.random.default_rng(self.seed)

        A, B, _ = half_split(X, rng)
        A, B = _drop_nan_features(A, B)

        idx = np.arange(A.shape[0])
        train, test = train_test_split(idx, test_size=self.test_size,
                                       random_state=self.seed, shuffle=True)

        prep = _PooledPrep(self.zscore, self.do_pca, self.n_pcs,
                           self.pca_var_threshold).fit(A[train], B[train])
        A_tr, B_tr = prep.transform(A[train]), prep.transform(B[train])
        A_te, B_te = prep.transform(A[test]), prep.transform(B[test])

        ridge = RidgeCV(alphas=self.ridge_alphas, alpha_per_target=True)
        ridge.fit(A_tr, B_tr)

        res = {"n_train": int(len(train)), "n_test": int(len(test)),
               "chance": {k: k / len(test) for k in self.top_ks},
               "test_r2": float(ridge.score(A_te, B_te))}
        res.update(self._retrieval(ridge.predict(A_tr), A_tr, B_tr, "train"))
        res.update(self._retrieval(ridge.predict(A_te), A_te, B_te, "test"))

        if self.verbose:
            self._print(res)
        return res

    def _retrieval(self, B_pred, A, B, split):
        """Top-K for model corr / model -MSE / no-model baseline on one split."""
        return {
            f"top_k_correlation_{split}": topk_from_similarity(
                correlation_matrix(B_pred, B), self.top_ks),
            f"top_k_neg_mse_{split}": topk_from_similarity(
                neg_mse_matrix(B_pred, B), self.top_ks),
            f"top_k_baseline_{split}": topk_from_similarity(
                correlation_matrix(A, B), self.top_ks),
        }

    def _print(self, res):
        print(f"\ntemplate matching: n_train={res['n_train']} n_test={res['n_test']} "
              f"(test R^2={res['test_r2']:.4f})")
        header = f"  {'K':>4} {'chance':>8} {'corr(tr)':>9} {'corr(te)':>9} " \
                 f"{'-MSE(te)':>9} {'base(te)':>9}"
        print(header)
        for k in self.top_ks:
            print(f"  {k:>4d} {res['chance'][k]:>8.2%} "
                  f"{res['top_k_correlation_train'][k]:>9.2%} "
                  f"{res['top_k_correlation_test'][k]:>9.2%} "
                  f"{res['top_k_neg_mse_test'][k]:>9.2%} "
                  f"{res['top_k_baseline_test'][k]:>9.2%}")


class CLIPDecodingEval(Evaluation):
    """RidgeCV brain->embedding decoder: fit on train images, retrieve on held-out images.

    Expects rep-averaged ``X = (conditions, features)`` and targets ``y =
    (conditions, embed_dim)`` (e.g. CLIP embeddings), aligned on the conditions axis.
    Returns cosine top-K retrieval of the predicted embedding against the true ones on
    the test (and train) split vs. chance ``K / n``.
    """

    def __init__(self, top_ks=(1, 5, 20, 100), test_size=0.2, zscore=True, do_pca=True,
                 n_pcs=None, pca_var_threshold=0.98, target_pca=None, ridge_alphas=None,
                 seed=42, verbose=True):
        self.top_ks = tuple(top_ks)
        self.test_size = test_size
        self.zscore = zscore
        self.do_pca = do_pca
        self.n_pcs = n_pcs
        self.pca_var_threshold = pca_var_threshold
        self.target_pca = target_pca
        self.ridge_alphas = (np.logspace(-4, 10, 80) if ridge_alphas is None
                             else np.asarray(ridge_alphas))
        self.seed = seed
        self.verbose = verbose

    def run(self, X, stim_info=None, y=None, rng=None):
        if y is None:
            raise ValueError("CLIPDecodingEval requires targets y (the embeddings).")
        y = np.asarray(y, dtype=float)

        bad = np.isnan(X).any(axis=0)              # channels missing from some runs
        if bad.any():
            warnings.warn(f"dropping {int(bad.sum())} feature(s) with NaNs "
                          f"(channels missing from some runs).", stacklevel=2)
            X = X[:, ~bad]

        idx = np.arange(X.shape[0])
        train, test = train_test_split(idx, test_size=self.test_size,
                                       random_state=self.seed, shuffle=True)

        prep = _FeaturePrep(self.zscore, self.do_pca, self.n_pcs,
                            self.pca_var_threshold).fit(X[train])
        X_tr, X_te = prep.transform(X[train]), prep.transform(X[test])

        # Optionally reduce the (possibly huge, e.g. bigG token-level) target with a PCA
        # fit on the train embeddings only; decode and retrieve in that reduced space.
        y_prep = self._fit_target_prep(y[train])
        y_tr, y_te = y_prep.transform(y[train]), y_prep.transform(y[test])

        ridge = RidgeCV(alphas=self.ridge_alphas, alpha_per_target=True)
        ridge.fit(X_tr, y_tr)

        n_components = None if y_prep.pca_ is None else int(y_prep.pca_.n_components_)
        res = {"n_train": int(len(train)), "n_test": int(len(test)),
               "chance": {k: k / len(test) for k in self.top_ks},
               "test_r2": float(ridge.score(X_te, y_te)),
               "target_pca_components": n_components}
        res["top_k_retrieval_train"] = topk_from_similarity(
            cosine_matrix(ridge.predict(X_tr), y_tr), self.top_ks)
        res["top_k_retrieval_test"] = topk_from_similarity(
            cosine_matrix(ridge.predict(X_te), y_te), self.top_ks)

        if self.verbose:
            self._print(res)
        return res

    def _fit_target_prep(self, y_train):
        """Target-side PCA fit on train embeddings: ``None`` -> identity (no reduction);
        ``int`` -> that many components; ``float in (0, 1]`` -> retained-variance threshold."""
        if self.target_pca is None:
            return _FeaturePrep(zscore=False, do_pca=False).fit(y_train)
        if isinstance(self.target_pca, float) and 0.0 < self.target_pca <= 1.0:
            return _FeaturePrep(zscore=False, do_pca=True, n_pcs=None,
                                pca_var_threshold=self.target_pca).fit(y_train)
        return _FeaturePrep(zscore=False, do_pca=True,
                            n_pcs=int(self.target_pca)).fit(y_train)

    def _print(self, res):
        tgt = (f" target_pca={res['target_pca_components']}"
               if res["target_pca_components"] is not None else "")
        print(f"\nclip decoding: n_train={res['n_train']} n_test={res['n_test']}{tgt} "
              f"(test R^2={res['test_r2']:.4f})")
        print(f"  {'K':>4} {'chance':>8} {'retr(tr)':>9} {'retr(te)':>9}")
        for k in self.top_ks:
            print(f"  {k:>4d} {res['chance'][k]:>8.2%} "
                  f"{res['top_k_retrieval_train'][k]:>9.2%} "
                  f"{res['top_k_retrieval_test'][k]:>9.2%}")
