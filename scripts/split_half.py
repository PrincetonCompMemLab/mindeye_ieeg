#!/usr/bin/env python
# coding: utf-8

"""
Template-matching analysis: A -> B half-split ridge regression.

For each image with >=2 repetitions, split its trials into two independent halves
(Half A, Half B), average within each half, then fit a RidgeCV mapping
A -> B across all 2100 images. Evaluate via top-K retrieval on the same set
(in-sample, regularized by internal LOO-CV over alphas).

Noise-balancing for 3-rep images: half-split is 1-vs-2; per-image coin flip
decides which side gets 2 trials, so on average both halves contain the same
mix of 1-trial and 2-trial averages.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


# ----------------------------- config -----------------------------

SEED = 42
np.random.seed(SEED)

BASE_DIR = '/home/ri99/ieeg'
DATA_DIR = f'{BASE_DIR}/data/hfb'
OUTPUT_DIR = f'{BASE_DIR}/outputs/hfb'

# match the smoothing suffix used to save the reduced array
SMOOTHING = True
WINDOW_SIZE = 25
STRIDE = 10
SMOOTH_SUFFIX = f'_window{WINDOW_SIZE}_stride{STRIDE}' if SMOOTHING else ''

REDUCED_DATA_PATH = f'{OUTPUT_DIR}/top_ncsnr_electrodes_timepoints{SMOOTH_SUFFIX}.npy'
STIM_INFO_PATH = f'{DATA_DIR}/stim_info.pkl'

# pipeline knobs
DO_PCA = True
PCA_VAR_THRESHOLD = 0.98       # used only if DO_PCA and N_PCS is None
N_PCS = None                   # if set, overrides PCA_VAR_THRESHOLD
ZSCORE = True                  # z-score per feature using pooled (A, B) stats

TOP_KS = [1, 2, 5, 10, 20, 50, 100]
N_PERMUTATIONS = 1000

RIDGE_ALPHAS = np.logspace(-4, 10, 80)  # extended upper end; previous run pinned at 1e6

N_CLUSTERS_FOR_SIM_PLOT = 20   # k-means clusters used to sort the similarity matrix


# ----------------------------- data loading -----------------------------

def load_reduced_data():
    """
    Load the reduced (top-ncsnr electrodes, top timepoints, smoothed) array.

    Expected shape: (channels, n_unique_images, n_timepoints, max_reps),
    with NaN padding for images that have fewer than max_reps trials.
    """
    print(f'loading reduced_data from {REDUCED_DATA_PATH}')
    reduced = np.load(REDUCED_DATA_PATH)
    print(f'  reduced_data shape: {reduced.shape}')

    print(f'loading stim_info from {STIM_INFO_PATH}')
    with open(STIM_INFO_PATH, 'rb') as f:
        stim_info = pickle.load(f)
    print(f"  n_unique_images: {stim_info['n_unique_images']}")

    return reduced, stim_info


def get_rep_counts(reduced_data, stim_info):
    """
    For each unique image, count the number of non-NaN repetitions.
    Cross-checks against stim_info['stimulus_to_trials'] for sanity.
    """
    n_images = reduced_data.shape[1]
    # use channel 0 timepoint 0 to detect non-NaN reps; consistent across (channel, timepoint)
    rep_counts = np.sum(~np.isnan(reduced_data[0, :, 0, :]), axis=-1).astype(int)

    expected = np.array(
        [len(stim_info['stimulus_to_trials'][k])
         for k in stim_info['stimulus_to_trials']]
    )
    assert n_images == len(expected), 'image count mismatch with stim_info'
    assert np.array_equal(rep_counts, expected), \
        'per-image rep count mismatch between reduced_data NaN pattern and stim_info'

    return rep_counts


# ----------------------------- half-splitting -----------------------------

def split_trials_into_halves(reduced_data, rep_counts, rng):
    """
    For each image with >=2 reps, randomly assign trials to Half A and Half B,
    then average within each half.

    3-rep images: split is 1-vs-2; per-image coin flip decides which half gets 2.
    On average, ~half of 3-rep images contribute 2 trials to A and 1 to B,
    and vice versa, so the global mix of 1-avg vs 2-avg conditions is balanced
    across halves.

    2-rep images: 1-vs-1.
    4-rep images: 2-vs-2.

    Returns
    -------
    pattern_A, pattern_B : (n_kept_images, channels, timepoints)
    kept_image_idx       : (n_kept_images,) indices into the original image axis
    kept_rep_counts      : (n_kept_images,) original rep count per kept image
    """
    n_channels, n_images, n_timepoints, max_reps = reduced_data.shape

    keep_mask = rep_counts >= 2
    kept_image_idx = np.where(keep_mask)[0]
    kept_rep_counts = rep_counts[keep_mask]

    n_kept = len(kept_image_idx)
    print(f'  keeping {n_kept} images with >=2 reps '
          f'(dropped {n_images - n_kept} with 1 rep)')
    for r in sorted(np.unique(kept_rep_counts)):
        print(f'    {int((kept_rep_counts == r).sum())} images with {r} reps')

    pattern_A = np.empty((n_kept, n_channels, n_timepoints), dtype=np.float32)
    pattern_B = np.empty((n_kept, n_channels, n_timepoints), dtype=np.float32)

    for i, img_idx in enumerate(kept_image_idx):
        n_reps = kept_rep_counts[i]
        # the first n_reps slots along the rep axis are the valid trials
        trials = reduced_data[:, img_idx, :, :n_reps]   # (channels, timepoints, n_reps)

        perm = rng.permutation(n_reps)

        if n_reps % 2 == 0:
            half = n_reps // 2
            a_idx = perm[:half]
            b_idx = perm[half:]
        else:
            # odd (n_reps==3 in practice): coin flip for which half gets 2 trials
            big_first = rng.random() < 0.5
            if big_first:
                a_idx = perm[:2]
                b_idx = perm[2:]
            else:
                a_idx = perm[:1]
                b_idx = perm[1:]

        pattern_A[i] = trials[:, :, a_idx].mean(axis=-1)
        pattern_B[i] = trials[:, :, b_idx].mean(axis=-1)

    assert not np.isnan(pattern_A).any(), 'NaNs in pattern_A'
    assert not np.isnan(pattern_B).any(), 'NaNs in pattern_B'

    return pattern_A, pattern_B, kept_image_idx, kept_rep_counts


# ----------------------------- preprocessing -----------------------------

def flatten_and_preprocess(pattern_A, pattern_B, zscore=True, do_pca=True,
                          n_pcs=None, pca_var_threshold=0.98):
    """
    Flatten (n, channels, timepoints) -> (n, channels*timepoints).
    Z-score per feature using pooled (A, B) statistics.
    Optionally PCA on pooled (A, B) to keep the basis symmetric for input and target.

    Returns A_proc, B_proc, info dict.
    """
    n = pattern_A.shape[0]
    A_flat = pattern_A.reshape(n, -1)
    B_flat = pattern_B.reshape(n, -1)
    info = {'n_features_raw': A_flat.shape[1]}

    if zscore:
        # pool A and B since they are exchangeable estimates of the same patterns
        scaler = StandardScaler()
        scaler.fit(np.vstack([A_flat, B_flat]))
        A_flat = scaler.transform(A_flat)
        B_flat = scaler.transform(B_flat)
        info['scaler'] = scaler

    if do_pca:
        pooled = np.vstack([A_flat, B_flat])
        if n_pcs is None:
            full_pca = PCA().fit(pooled)
            cumvar = np.cumsum(full_pca.explained_variance_ratio_)
            n_pcs = int(np.searchsorted(cumvar, pca_var_threshold) + 1)
            print(f'  selected {n_pcs} PCs to capture {pca_var_threshold} of variance '
                  f'(out of {len(cumvar)})')
        pca = PCA(n_components=n_pcs).fit(pooled)
        A_proc = pca.transform(A_flat)
        B_proc = pca.transform(B_flat)
        info['pca'] = pca
        info['n_pcs'] = n_pcs
    else:
        A_proc = A_flat
        B_proc = B_flat

    info['n_features_final'] = A_proc.shape[1]
    return A_proc, B_proc, info


# ----------------------------- retrieval -----------------------------

def ranks_from_sim(sim_matrix, higher_is_better=True):
    """
    sim_matrix[i, j] = similarity between predicted_i and true_j.
    Returns a length-n array where ranks[i] = number of off-target entries in
    row i that beat the diagonal (rank 0 = top-1 correct).
    """
    diag = np.diag(sim_matrix)[:, None]
    if higher_is_better:
        return np.sum(sim_matrix > diag, axis=1)
    else:
        return np.sum(sim_matrix < diag, axis=1)


def topk_accuracy_from_ranks(ranks, ks):
    """Top-K means rank < K (rank 0 = top-1). Returns dict k -> accuracy."""
    return {k: float(np.mean(ranks < k)) for k in ks}


def topk_accuracy_from_sim(sim_matrix, ks, higher_is_better=True):
    """Convenience wrapper: compute ranks, then top-K accuracy."""
    return topk_accuracy_from_ranks(
        ranks_from_sim(sim_matrix, higher_is_better=higher_is_better), ks
    )


def correlation_matrix(X, Y):
    """Row-wise Pearson correlation between rows of X and rows of Y. Returns (nX, nY)."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def neg_mse_matrix(X, Y):
    """Negative pairwise MSE between rows of X and rows of Y. Higher = closer."""
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)            # (nX, 1)
    y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T          # (1, nY)
    xy = X @ Y.T                                             # (nX, nY)
    sq_dist = x_sq + y_sq - 2 * xy
    n_features = X.shape[1]
    return -sq_dist / n_features


def rankdata_rows(X):
    """
    Per-row ranks (0..n_features-1). Ties broken arbitrarily by argsort.
    Used to convert Pearson correlation -> Spearman correlation by passing
    the ranked rows into correlation_matrix.

    For continuous-valued float data, exact ties are vanishingly rare so the
    arbitrary tie-breaking doesn't matter in practice.
    """
    order = np.argsort(X, axis=1)
    ranks = np.empty_like(order, dtype=np.int32)
    cols = np.broadcast_to(np.arange(X.shape[1], dtype=np.int32), X.shape)
    np.put_along_axis(ranks, order, cols, axis=1)
    return ranks.astype(np.float32)


# ----------------------------- main pipeline -----------------------------

def run_pipeline():
    rng = np.random.default_rng(SEED)

    # --- load ---
    reduced_data, stim_info = load_reduced_data()
    rep_counts = get_rep_counts(reduced_data, stim_info)

    # --- half split ---
    print('\n[1/5] splitting trials into halves...')
    pattern_A, pattern_B, kept_idx, kept_reps = split_trials_into_halves(
        reduced_data, rep_counts, rng
    )
    print(f'  pattern_A shape: {pattern_A.shape}')
    print(f'  pattern_B shape: {pattern_B.shape}')

    # free the big array
    del reduced_data

    # --- preprocess ---
    print('\n[2/5] preprocessing (z-score + PCA)...')
    A_proc, B_proc, prep_info = flatten_and_preprocess(
        pattern_A, pattern_B,
        zscore=ZSCORE, do_pca=DO_PCA,
        n_pcs=N_PCS, pca_var_threshold=PCA_VAR_THRESHOLD,
    )
    print(f'  A_proc shape: {A_proc.shape}')
    print(f'  B_proc shape: {B_proc.shape}')

    # --- fit RidgeCV: A -> B ---
    print('\n[3/5] fitting RidgeCV (A -> B)...')
    print(f'  alphas: {len(RIDGE_ALPHAS)} values from '
          f'{RIDGE_ALPHAS.min():.0e} to {RIDGE_ALPHAS.max():.0e}')
    ridge = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True)
    ridge.fit(A_proc, B_proc)

    if hasattr(ridge, 'alpha_') and np.ndim(ridge.alpha_) > 0:
        print(f'  per-target alpha summary: '
              f'median={np.median(ridge.alpha_):.3g}, '
              f'min={ridge.alpha_.min():.3g}, max={ridge.alpha_.max():.3g}')
    else:
        print(f'  selected alpha: {ridge.alpha_:.3g}')

    r2 = ridge.score(A_proc, B_proc)
    print(f'  in-sample R^2 (A -> B): {r2:.4f}')

    # --- predict and score ---
    print('\n[4/5] predicting B from A and scoring retrieval...')
    B_pred = ridge.predict(A_proc)

    # similarity matrices: rows = predictions, cols = true targets
    print('  computing model correlation matrix (B_pred vs B)...')
    corr_sim = correlation_matrix(B_pred, B_proc)
    print('  computing model -MSE matrix (B_pred vs B)...')
    mse_sim = neg_mse_matrix(B_pred, B_proc)

    corr_ranks = ranks_from_sim(corr_sim, higher_is_better=True)
    mse_ranks = ranks_from_sim(mse_sim, higher_is_better=True)
    corr_acc = topk_accuracy_from_ranks(corr_ranks, TOP_KS)
    mse_acc = topk_accuracy_from_ranks(mse_ranks, TOP_KS)

    # rank-correlation retrieval: less sensitive to amplitude shrinkage and
    # to a few high-magnitude features dominating the dot product. if Spearman
    # gives substantially better K=1 than Pearson, the anchor effect we're
    # seeing is partly a metric artifact; if it's similar, the anchor effect
    # is real signal-structure.
    print('  computing Spearman (rank) similarity matrix...')
    B_pred_ranks = rankdata_rows(B_pred)
    B_proc_ranks = rankdata_rows(B_proc)
    spear_sim = correlation_matrix(B_pred_ranks, B_proc_ranks)
    spear_ranks_arr = ranks_from_sim(spear_sim, higher_is_better=True)
    spear_acc = topk_accuracy_from_ranks(spear_ranks_arr, TOP_KS)

    # --- noise-ceiling retrieval: A directly vs B, no model ---
    # this is the empirical upper bound given the SNR of single half-splits;
    # if a model could extract every bit of generalizable signal, it could not
    # exceed this, since A and B are independent draws of the same image.
    print('  computing noise-ceiling correlation matrix (A vs B, no model)...')
    ceil_corr_sim = correlation_matrix(A_proc, B_proc)
    ceil_corr_ranks = ranks_from_sim(ceil_corr_sim, higher_is_better=True)
    ceil_corr_acc = topk_accuracy_from_ranks(ceil_corr_ranks, TOP_KS)

    n = B_pred.shape[0]
    print(f'\n  retrieval over {n} images (chance = K/{n}):')
    print(f'  {"K":>4} {"chance":>8} {"corr-acc":>10} {"-MSE acc":>10} '
          f'{"spear-acc":>10} {"ceil corr":>10}')
    for k in TOP_KS:
        chance = k / n
        print(f'  {k:>4d} {chance:>8.2%} {corr_acc[k]:>10.2%} '
              f'{mse_acc[k]:>10.2%} {spear_acc[k]:>10.2%} '
              f'{ceil_corr_acc[k]:>10.2%}')

    # --- permutation test (using correlation, top-5 by default) ---
    print(f'\n[5/5] permutation test ({N_PERMUTATIONS} shuffles, correlation, top-5)...')
    k_perm = 5
    true_acc = corr_acc[k_perm]
    null_accs = np.empty(N_PERMUTATIONS)
    for i in tqdm(range(N_PERMUTATIONS)):
        perm = rng.permutation(n)
        shuffled_sim = corr_sim[perm]   # shuffles which prediction is paired with which row index
        # under shuffle, the "diagonal" entry no longer matches the true target;
        # equivalent to: for each row i of corr_sim, ask whether sim[i, perm_inv[i]] is in top-K
        # easier: shuffle the columns of corr_sim, the diagonal is now randomized
        shuffled_sim = corr_sim[:, perm]
        diag = np.diag(shuffled_sim)[:, None]
        ranks = np.sum(shuffled_sim > diag, axis=1)
        null_accs[i] = float(np.mean(ranks < k_perm))

    p_val = float(np.mean(null_accs >= true_acc))
    print(f'  true top-{k_perm} acc: {true_acc:.2%}')
    print(f'  null mean: {null_accs.mean():.2%}, '
          f'null 95th pct: {np.percentile(null_accs, 95):.2%}')
    print(f'  p-value: {p_val:.4f}')

    # --- plots ---
    print('\nplotting...')

    # top-K curve with noise ceiling overlay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = np.array(TOP_KS)
    ax.plot(ks, [ceil_corr_acc[k] for k in TOP_KS], '^-', color='C2',
            label='baseline (correlate predictor and target, no model)')
    ax.plot(ks, [corr_acc[k] for k in TOP_KS], 'o-', color='C0',
            label='model (B_pred vs B, corr)')
    ax.plot(ks, [spear_acc[k] for k in TOP_KS], 'D-', color='C3',
            label='model (B_pred vs B, Spearman)')
    ax.plot(ks, [mse_acc[k] for k in TOP_KS], 's-', color='C1',
            label='model (B_pred vs B, -MSE)')
    ax.plot(ks, ks / n, 'k--', alpha=0.5, label='chance')
    ax.set_xscale('log')
    ax.set_xlabel('K')
    ax.set_ylabel('Top-K retrieval accuracy')
    ax.set_title(f'Template-matching retrieval (n={n} images)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_topk.png'
    plt.savefig(out_path, dpi=120)
    print(f'  saved {out_path}')
    plt.close()

    # per-image rank histogram (model, correlation), stratified by rep count
    # rank 0 = top-1 hit; rank n-1 = worst possible. Plot on log-x to see the
    # spike at low ranks if it exists.
    unique_reps = sorted(np.unique(kept_reps).tolist())
    fig, axes = plt.subplots(1, len(unique_reps) + 1,
                             figsize=(4 * (len(unique_reps) + 1), 4),
                             sharey=False)

    # bin edges in log space so a spike near rank 0 is visible;
    # use rank+1 to avoid log(0)
    bins = np.logspace(0, np.log10(n), 40)

    # all images
    ax = axes[0]
    ax.hist(corr_ranks + 1, bins=bins, color='C0', edgecolor='white')
    ax.axvline(5, color='red', linestyle=':', label='top-5')
    ax.axvline(n / 2, color='gray', linestyle='--', alpha=0.6, label='chance median')
    ax.set_xscale('log')
    ax.set_xlabel('rank of true image (1 = top-1)')
    ax.set_ylabel('count')
    ax.set_title(f'all images (n={len(corr_ranks)}, '
                 f'median rank={int(np.median(corr_ranks)) + 1})')
    ax.legend(fontsize=8)

    # per rep count
    for j, r in enumerate(unique_reps):
        ax = axes[j + 1]
        mask = kept_reps == r
        sub = corr_ranks[mask]
        ax.hist(sub + 1, bins=bins, color=f'C{j+1}', edgecolor='white')
        ax.axvline(5, color='red', linestyle=':')
        ax.axvline(n / 2, color='gray', linestyle='--', alpha=0.6)
        ax.set_xscale('log')
        ax.set_xlabel('rank of true image')
        top1 = float(np.mean(sub == 0))
        top5 = float(np.mean(sub < 5))
        ax.set_title(f'{r} reps (n={mask.sum()})\n'
                     f'top-1={top1:.1%}, top-5={top5:.1%}')

    fig.suptitle('Per-image retrieval rank (model, correlation)', y=1.02)
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_rank_histogram.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # also report top-K stratified by rep count in the console
    print('\n  per-rep-count retrieval (model, correlation):')
    print(f'  {"reps":>5} {"n":>6} {"top-1":>8} {"top-5":>8} {"top-20":>8} '
          f'{"top-100":>8}')
    rep_stratified = {}
    for r in unique_reps:
        mask = kept_reps == r
        sub = corr_ranks[mask]
        row = {k: float(np.mean(sub < k)) for k in TOP_KS}
        rep_stratified[int(r)] = {'n': int(mask.sum()), **{f'top_{k}': v for k, v in row.items()}}
        print(f'  {r:>5d} {mask.sum():>6d} {row[1]:>8.2%} {row[5]:>8.2%} '
              f'{row[20]:>8.2%} {row[100]:>8.2%}')

    # ---- sorted similarity matrix ----
    # cluster images on B_proc with k-means, sort by cluster label, and show the
    # 2100x2100 model correlation matrix. block structure on the diagonal =
    # category-level signal (model identifies the right neighborhood but not
    # the right exemplar within it). thin lone diagonal = exemplar-level signal.
    print(f'\n  clustering images into {N_CLUSTERS_FOR_SIM_PLOT} groups for '
          f'sorted similarity plot...')
    km = KMeans(n_clusters=N_CLUSTERS_FOR_SIM_PLOT, n_init=10,
                random_state=SEED).fit(B_proc)
    cluster_labels = km.labels_
    sort_order = np.argsort(cluster_labels, kind='stable')
    sorted_corr = corr_sim[sort_order][:, sort_order]
    sorted_ceil = ceil_corr_sim[sort_order][:, sort_order]

    cluster_boundaries = np.where(np.diff(cluster_labels[sort_order]) != 0)[0] + 1

    # use a robust color range so the diagonal isn't blown out by outliers
    vmax = float(np.percentile(np.abs(sorted_corr), 99))
    vmin = -vmax

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in [
        (axes[0], sorted_corr, 'Model: corr(B_pred, B), sorted by B_proc cluster'),
        (axes[1], sorted_ceil, 'Ceiling: corr(A, B), sorted by B_proc cluster'),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='RdBu_r',
                       interpolation='nearest', aspect='equal')
        for b in cluster_boundaries:
            ax.axhline(b, color='k', linewidth=0.3, alpha=0.4)
            ax.axvline(b, color='k', linewidth=0.3, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('true (B) image index (sorted)')
        ax.set_ylabel('predicted (B_pred or A) image index (sorted)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_sorted_similarity.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # ---- per-target R^2 distribution ----
    # sklearn r2_score with multioutput='raw_values' returns one R^2 per target dim.
    # PCs (when DO_PCA=True) are ordered by descending input variance, NOT by
    # how predictable the target is, so we plot in PC order AND sorted.
    per_target_r2 = r2_score(B_proc, B_pred, multioutput='raw_values')
    n_targets = len(per_target_r2)
    print(f'  per-target R^2: median={np.median(per_target_r2):.4f}, '
          f'max={per_target_r2.max():.4f}, '
          f'frac > 0: {float(np.mean(per_target_r2 > 0)):.2%}, '
          f'frac > 0.01: {float(np.mean(per_target_r2 > 0.01)):.2%}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.bar(np.arange(n_targets), per_target_r2, color='steelblue')
    ax.axhline(0, color='k', linewidth=0.7)
    ax.set_xlabel('target dimension (PC index, in input-variance order)' if DO_PCA
                  else 'target dimension index')
    ax.set_ylabel('R^2')
    ax.set_title('Per-target R^2, in target-PC order')

    ax = axes[1]
    sorted_r2 = np.sort(per_target_r2)[::-1]
    ax.plot(np.arange(n_targets), sorted_r2, '.-', color='steelblue')
    ax.axhline(0, color='k', linewidth=0.7)
    ax.axhline(0.01, color='red', linestyle=':', label='R^2 = 0.01')
    ax.set_xlabel('target rank (best -> worst)')
    ax.set_ylabel('R^2')
    ax.set_title('Per-target R^2, sorted')
    ax.legend()

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_per_target_r2.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # ---- per-target alpha histogram ----
    # given the spread (4.8e4 to 1e10 with median 8e6), expect bimodality:
    # one cluster of "real signal" targets at intermediate alpha, one cluster of
    # "noise" targets pinned at the grid ceiling.
    if hasattr(ridge, 'alpha_') and np.ndim(ridge.alpha_) > 0:
        alphas_per_target = np.asarray(ridge.alpha_)
        log_alphas = np.log10(alphas_per_target)
        log_grid = np.log10(RIDGE_ALPHAS)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(log_alphas, bins=40, color='steelblue', edgecolor='white')
        ax.axvline(log_grid.min(), color='gray', linestyle=':',
                   alpha=0.7, label='grid floor')
        ax.axvline(log_grid.max(), color='red', linestyle=':',
                   alpha=0.7, label='grid ceiling')
        n_at_ceiling = int(np.sum(alphas_per_target >= RIDGE_ALPHAS.max() * 0.999))
        ax.set_xlabel('log10(alpha) selected by RidgeCV')
        ax.set_ylabel('count (target dims)')
        ax.set_title(f'Per-target alpha (n={n_targets} targets, '
                     f'{n_at_ceiling} pinned at ceiling)')
        ax.legend()
        plt.tight_layout()
        out_path = f'{OUTPUT_DIR}/template_matching_alpha_histogram.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f'  saved {out_path}')
        plt.close()
    else:
        alphas_per_target = None
        n_at_ceiling = 0

    # ---- SVD spectrum of the ridge weight matrix ----
    # ridge.coef_ has shape (n_targets, n_features). Its singular values describe
    # the effective rank of the A -> B mapping. If the model operates in a
    # low-dimensional subspace (as suggested by the alpha histogram and per-target
    # R^2 plot), the singular value spectrum should drop off sharply around the
    # same dimensionality. A knee at ~10 would line up with the ~10 predictable
    # target PCs.
    print('\n  computing SVD of ridge weight matrix...')
    coef = ridge.coef_  # (n_targets, n_features)
    print(f'    coef shape: {coef.shape}')
    U_svd, sing_vals, Vt_svd = np.linalg.svd(coef, full_matrices=False)
    # U_svd: (n_targets_pca, k) — output (B-PCA) directions
    # Vt_svd: (k, n_features_pca) — input (A-PCA) directions
    print(f'    largest 5 singular values: {sing_vals[:5]}')
    print(f'    smallest 5 singular values: {sing_vals[-5:]}')

    # define an "effective rank" by the number of singular values within 10x of the largest
    eff_rank_10x = int(np.sum(sing_vals > sing_vals[0] / 10))
    eff_rank_100x = int(np.sum(sing_vals > sing_vals[0] / 100))
    print(f'    effective rank (s > s_max/10):  {eff_rank_10x}')
    print(f'    effective rank (s > s_max/100): {eff_rank_100x}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(sing_vals) + 1), sing_vals,
                'o-', markersize=4, color='steelblue')
    ax.axvline(eff_rank_10x, color='red', linestyle=':',
               label=f's > s_max/10  ({eff_rank_10x} dims)')
    ax.axvline(eff_rank_100x, color='gray', linestyle=':',
               label=f's > s_max/100 ({eff_rank_100x} dims)')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('singular value (log scale)')
    ax.set_title('SVD spectrum of ridge weight matrix')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cum_energy = np.cumsum(sing_vals ** 2) / np.sum(sing_vals ** 2)
    ax.plot(np.arange(1, len(sing_vals) + 1), cum_energy,
            '.-', color='steelblue')
    for thresh, color in [(0.9, 'red'), (0.95, 'orange'), (0.99, 'gray')]:
        n_for_thresh = int(np.searchsorted(cum_energy, thresh) + 1)
        ax.axhline(thresh, color=color, linestyle=':', alpha=0.7,
                   label=f'{thresh:.0%} energy ({n_for_thresh} dims)')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('cumulative spectral energy')
    ax.set_title('Cumulative SVD energy of ridge weight matrix')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_weight_svd.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # ---- top singular vectors back in channel x timepoint space ----
    # the right singular vectors (rows of Vt) are input directions in A-PCA
    # space; the left singular vectors (columns of U) are output directions
    # in B-PCA space. we used a single PCA fit on the pooled (A, B) data, so
    # the same components_ matrix maps both back to flat (channels*timepoints)
    # feature space, which we then reshape to (channels, timepoints) for
    # visualization.
    #
    # what to look for: are the top-N predictable directions concentrated on a
    # few electrodes / timepoints (suggesting the signal lives in a tight
    # spatiotemporal locus), or smeared across many channels (suggesting it's
    # distributed)? are the input and output directions for the same SV pair
    # similar (model is essentially identity on that subspace) or different
    # (model is doing nontrivial spatial/temporal remapping)?
    if DO_PCA and 'pca' in prep_info:
        n_sv_to_show = min(6, len(sing_vals), eff_rank_10x + 2)
        n_channels, n_timepoints = pattern_A.shape[1], pattern_A.shape[2]
        components = prep_info['pca'].components_   # (n_pcs, n_features)

        # project SVs back to flat feature space, then reshape
        # input directions: V[k] is in input-PCA space (n_pcs,)
        v_feat = Vt_svd @ components               # (k, n_features)
        u_feat = U_svd.T @ components              # (k, n_features)
        v_maps = v_feat.reshape(-1, n_channels, n_timepoints)
        u_maps = u_feat.reshape(-1, n_channels, n_timepoints)

        fig, axes = plt.subplots(n_sv_to_show, 2,
                                 figsize=(11, 1.6 * n_sv_to_show + 1),
                                 sharex=True, sharey=True)
        if n_sv_to_show == 1:
            axes = axes.reshape(1, -1)

        for k in range(n_sv_to_show):
            v_map = v_maps[k]
            u_map = u_maps[k]
            # symmetric color range per row (input/output share the SV's scale)
            vmax = max(np.abs(v_map).max(), np.abs(u_map).max())

            ax = axes[k, 0]
            im = ax.imshow(v_map, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           interpolation='nearest', origin='upper')
            ax.set_ylabel(f'SV {k+1}\nσ={sing_vals[k]:.3f}')
            if k == 0:
                ax.set_title('Input direction (right SV, A-side)')
            if k == n_sv_to_show - 1:
                ax.set_xlabel('timepoint index')

            ax = axes[k, 1]
            im = ax.imshow(u_map, aspect='auto', cmap='RdBu_r',
                           vmin=-vmax, vmax=vmax,
                           interpolation='nearest', origin='upper')
            if k == 0:
                ax.set_title('Output direction (left SV, B-side)')
            if k == n_sv_to_show - 1:
                ax.set_xlabel('timepoint index')

        # one shared colorbar label hint
        axes[0, 0].set_ylabel(
            f'channel index\n\nSV 1\nσ={sing_vals[0]:.3f}', fontsize=9)
        fig.suptitle('Top singular vectors of ridge weight matrix '
                     '(channel × timepoint)', y=1.005)
        plt.tight_layout()
        out_path = f'{OUTPUT_DIR}/template_matching_weight_svd_maps.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        print(f'  saved {out_path}')
        plt.close()
    else:
        v_maps = None
        u_maps = None
        print('  (skipping SV channel-time maps because PCA was disabled)')

    # ---- stripe-source analysis ----
    # for each true image (column j of corr_sim), count how many predictions
    # (rows) put it in their top-1 / top-5. if the model produces low-rank,
    # shrunken outputs, a small handful of "generic" images at the centroid of
    # the predictable subspace will be top-1 for many predictions -> a few
    # columns will hog the predictions, producing vertical stripes in the
    # similarity matrix.
    #
    # also do the symmetric thing for rows: how many true patterns is each
    # prediction the top-1 *for* (i.e., how often does prediction i look like
    # the most likely match for many true images). this catches "generic
    # predictions" that the model emits for many inputs.
    print('  computing stripe-source distributions...')
    # rank along columns: for each row, what's the rank of each column?
    # we already have corr_ranks for the diagonal; here we need full argsort.
    # to get top-1 column-wise per row: argmax along axis=1
    top1_col_per_row = np.argmax(corr_sim, axis=1)        # (n,) which true img each prediction picks
    top1_row_per_col = np.argmax(corr_sim, axis=0)        # (n,) which prediction each true img is closest to

    col_top1_counts = np.bincount(top1_col_per_row, minlength=n)  # how often each true img is the top-1 pick
    row_top1_counts = np.bincount(top1_row_per_col, minlength=n)  # how often each prediction is the top-1 pick

    # uniform expectation: each image is top-1 for exactly 1 prediction
    n_zero_col = int(np.sum(col_top1_counts == 0))
    max_col = int(col_top1_counts.max())
    print(f'    column top-1 hits: {n_zero_col}/{n} images are never picked as top-1; '
          f'max picks for a single image = {max_col}')
    n_zero_row = int(np.sum(row_top1_counts == 0))
    max_row = int(row_top1_counts.max())
    print(f'    row top-1 hits:    {n_zero_row}/{n} predictions are never the closest; '
          f'max for a single prediction = {max_row}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # column stripe sources (which true images are over-picked)
    ax = axes[0]
    max_count = max(col_top1_counts.max(), row_top1_counts.max())
    bins = np.arange(0, max_count + 2) - 0.5
    ax.hist(col_top1_counts, bins=bins, color='C0', edgecolor='white')
    ax.axvline(1, color='red', linestyle=':', label='uniform expectation = 1')
    ax.set_xlabel('# predictions with this true image as top-1')
    ax.set_ylabel('count (true images)')
    ax.set_title(f'Column stripe sources\n'
                 f'({n_zero_col} unpicked, max={max_col})')
    ax.legend()
    ax.set_yscale('log')

    # row stripe sources (which predictions are over-emitted)
    ax = axes[1]
    ax.hist(row_top1_counts, bins=bins, color='C1', edgecolor='white')
    ax.axvline(1, color='red', linestyle=':', label='uniform expectation = 1')
    ax.set_xlabel('# true images for which this prediction is top-1')
    ax.set_ylabel('count (predictions)')
    ax.set_title(f'Row stripe sources\n'
                 f'({n_zero_row} never top-1, max={max_row})')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_stripe_sources.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # ---- stripe-source characterization ----
    # what's special about the images that get over-picked? two angles:
    #   (1) energy in the predictable subspace — anchors should be high-norm
    #       rows of B_proc, since correlation against a low-rank shrunken
    #       prediction is dominated by whichever target rows have the most
    #       energy in the predictable directions.
    #   (2) rep-count distribution — if 4-rep images dominate the anchors,
    #       it suggests cleaner half-averages produce more extreme
    #       coordinates in the predictable subspace.
    # also save the kept_idx values for the top anchor images so they can be
    # mapped back to filenames externally.
    print('  characterizing stripe-source anchors...')

    # row-norms of B_proc — total energy per image after preprocessing
    b_energy = np.linalg.norm(B_proc, axis=1)

    # energy restricted to the top-eff_rank_10x predictable directions in B
    # (left singular vectors of the weight matrix)
    if DO_PCA:
        n_top_dirs = max(eff_rank_10x, 1)
        # project B_proc onto top output directions
        b_proj = B_proc @ U_svd[:, :n_top_dirs]   # (n, n_top_dirs)
        b_energy_predictable = np.linalg.norm(b_proj, axis=1)
    else:
        b_energy_predictable = b_energy

    # report the top-10 most-picked images
    top_anchor_local = np.argsort(col_top1_counts)[::-1][:20]
    top_anchor_global = kept_idx[top_anchor_local]
    print(f'    top-20 anchor images (local idx -> original image idx, picks, '
          f'rep_count, |B|, |B|_predictable):')
    for li, gi in zip(top_anchor_local, top_anchor_global):
        print(f'      local={li:>5d}  orig={gi:>5d}  picks={col_top1_counts[li]:>4d}  '
              f'reps={kept_reps[li]}  '
              f'|B|={b_energy[li]:>7.2f}  '
              f'|B|_pred={b_energy_predictable[li]:>7.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (1) scatter: predictable energy vs picks
    ax = axes[0]
    ax.scatter(b_energy_predictable, col_top1_counts,
               s=10, alpha=0.5, color='steelblue')
    # annotate top-5 anchors
    for li in top_anchor_local[:5]:
        ax.annotate(f'{kept_idx[li]}',
                    (b_energy_predictable[li], col_top1_counts[li]),
                    fontsize=8, ha='left', va='bottom', color='red')
    ax.axhline(1, color='red', linestyle=':', alpha=0.5,
               label='uniform expectation')
    if eff_rank_10x > 0:
        try:
            corr_e = float(np.corrcoef(b_energy_predictable, col_top1_counts)[0, 1])
        except Exception:
            corr_e = float('nan')
        ax.set_title(f'Predictable-subspace energy vs anchor strength\n'
                     f'r={corr_e:.2f} (top-{n_top_dirs} dirs)')
    else:
        ax.set_title('Predictable-subspace energy vs anchor strength')
    ax.set_xlabel(f'||B_proc projected onto top-{n_top_dirs} output SVs||')
    ax.set_ylabel('# top-1 picks (anchor strength)')
    ax.legend()

    # (2) col_top1_counts stratified by rep count, side-by-side log-y histograms
    ax = axes[1]
    bin_edges = np.arange(0, col_top1_counts.max() + 2) - 0.5
    for j, r in enumerate(unique_reps):
        mask = kept_reps == r
        ax.hist(col_top1_counts[mask], bins=bin_edges,
                histtype='step', linewidth=1.5,
                label=f'{r} reps (n={mask.sum()})', color=f'C{j+1}')
    ax.axvline(1, color='red', linestyle=':', alpha=0.5)
    ax.set_yscale('log')
    ax.set_xlabel('# top-1 picks')
    ax.set_ylabel('count (true images)')
    ax.set_title('Anchor strength stratified by rep count')
    ax.legend()

    # (3) rep-count composition of the top-50 vs all images
    ax = axes[2]
    top_n_anchor = 50
    top_anchors = np.argsort(col_top1_counts)[::-1][:top_n_anchor]
    rep_in_top = np.array([float(np.mean(kept_reps[top_anchors] == r))
                           for r in unique_reps])
    rep_overall = np.array([float(np.mean(kept_reps == r)) for r in unique_reps])
    x_pos = np.arange(len(unique_reps))
    width = 0.35
    ax.bar(x_pos - width / 2, rep_overall, width,
           color='steelblue', label='all images')
    ax.bar(x_pos + width / 2, rep_in_top, width,
           color='salmon', label=f'top-{top_n_anchor} anchors')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'{r} reps' for r in unique_reps])
    ax.set_ylabel('fraction')
    ax.set_title('Rep-count composition: anchors vs whole set')
    ax.legend()

    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_anchor_characterization.png'
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f'  saved {out_path}')
    plt.close()

    # permutation histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_accs, bins=30, color='steelblue', edgecolor='white',
            label='Null distribution')
    ax.axvline(true_acc, color='red', linewidth=2, linestyle='--',
               label=f'True top-{k_perm} ({true_acc:.2%})')
    ax.axvline(np.percentile(null_accs, 95), color='gray', linewidth=1.5,
               linestyle=':', label='95th pct null')
    ax.set_xlabel(f'Top-{k_perm} retrieval accuracy')
    ax.set_ylabel('Count')
    ax.set_title(f'Permutation test (p={p_val:.4f})')
    ax.legend()
    plt.tight_layout()
    out_path = f'{OUTPUT_DIR}/template_matching_permutation.png'
    plt.savefig(out_path, dpi=120)
    print(f'  saved {out_path}')
    plt.close()

    # results summary
    results = {
        'config': {
            'seed': SEED,
            'zscore': ZSCORE,
            'do_pca': DO_PCA,
            'n_pcs': prep_info.get('n_pcs'),
            'pca_var_threshold': PCA_VAR_THRESHOLD if N_PCS is None else None,
            'n_features_raw': prep_info['n_features_raw'],
            'n_features_final': prep_info['n_features_final'],
            'smoothing': SMOOTHING,
            'window_size': WINDOW_SIZE if SMOOTHING else None,
            'stride': STRIDE if SMOOTHING else None,
        },
        'n_images': int(n),
        'rep_count_distribution': {int(r): int((kept_reps == r).sum())
                                   for r in np.unique(kept_reps)},
        'in_sample_r2': float(r2),
        'top_k_correlation': {int(k): float(v) for k, v in corr_acc.items()},
        'top_k_neg_mse': {int(k): float(v) for k, v in mse_acc.items()},
        'top_k_spearman': {int(k): float(v) for k, v in spear_acc.items()},
        'top_k_noise_ceiling_correlation': {int(k): float(v)
                                            for k, v in ceil_corr_acc.items()},
        'per_image_ranks_correlation': corr_ranks.astype(int),
        'per_image_rep_count': kept_reps.astype(int),
        'top_k_by_rep_count_correlation': rep_stratified,
        'per_target_r2': per_target_r2.astype(np.float32),
        'per_target_alpha': (alphas_per_target.astype(np.float32)
                             if alphas_per_target is not None else None),
        'cluster_labels': cluster_labels.astype(int),
        'weight_singular_values': sing_vals.astype(np.float32),
        'weight_effective_rank_10x': eff_rank_10x,
        'weight_effective_rank_100x': eff_rank_100x,
        'weight_input_sv_maps': (v_maps.astype(np.float32)
                                 if v_maps is not None else None),
        'weight_output_sv_maps': (u_maps.astype(np.float32)
                                  if u_maps is not None else None),
        'col_top1_counts': col_top1_counts.astype(int),
        'row_top1_counts': row_top1_counts.astype(int),
        'top_anchor_local_idx': top_anchor_local.astype(int),
        'top_anchor_global_idx': top_anchor_global.astype(int),
        'b_energy': b_energy.astype(np.float32),
        'b_energy_predictable': b_energy_predictable.astype(np.float32),
        'kept_image_idx': kept_idx.astype(int),
        'permutation': {
            'k': k_perm,
            'metric': 'correlation',
            'n_permutations': int(N_PERMUTATIONS),
            'true_acc': float(true_acc),
            'null_mean': float(null_accs.mean()),
            'null_95th': float(np.percentile(null_accs, 95)),
            'p_value': float(p_val),
        },
    }

    out_path = f'{OUTPUT_DIR}/template_matching_results.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'  saved results to {out_path}')

    return results


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = run_pipeline()
    print('\ndone.')