"""Retrieval metrics shared by the evaluation layer.

Pure functions, ported from the exploratory ``split_half.py``. A *similarity matrix* has
rows = predictions and columns = candidate true targets; ``sim[i, j]`` scores how well
prediction ``i`` matches target ``j``. Retrieval asks where the diagonal (the true pair)
ranks within each row.
"""

import numpy as np


def correlation_matrix(X, Y):
    """Row-wise Pearson correlation between rows of ``X`` and ``Y`` -> ``(nX, nY)``."""
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    Xn = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def cosine_matrix(X, Y):
    """Row-wise cosine similarity between rows of ``X`` and ``Y`` -> ``(nX, nY)``.

    Like ``correlation_matrix`` but without centering, matching the CLIP retrieval
    convention (``utils.do_retrieval`` uses ``sklearn``'s ``cosine_similarity``).
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
    return Xn @ Yn.T


def neg_mse_matrix(X, Y):
    """Negative per-feature MSE between rows of ``X`` and ``Y`` -> ``(nX, nY)``.

    Higher (closer to 0) means more similar. Uses
    ``||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y``.
    """
    x_sq = np.sum(X ** 2, axis=1, keepdims=True)        # (nX, 1)
    y_sq = np.sum(Y ** 2, axis=1, keepdims=True).T      # (1, nY)
    sq_dist = x_sq + y_sq - 2 * (X @ Y.T)
    return -sq_dist / X.shape[1]


def ranks_from_sim(sim_matrix, higher_is_better=True):
    """Per-row rank of the diagonal (true) target. Rank 0 == top-1 correct.

    ``ranks[i]`` counts how many off-target entries in row ``i`` beat the diagonal.
    """
    diag = np.diag(sim_matrix)[:, None]
    if higher_is_better:
        return np.sum(sim_matrix > diag, axis=1)
    return np.sum(sim_matrix < diag, axis=1)


def topk_accuracy_from_ranks(ranks, ks):
    """Top-K accuracy from ranks: fraction with ``rank < k``. Returns ``{k: acc}``."""
    ranks = np.asarray(ranks)
    return {k: float(np.mean(ranks < k)) for k in ks}


def topk_from_similarity(sim_matrix, ks, higher_is_better=True):
    """Convenience: ranks then top-K accuracy from a similarity matrix."""
    ranks = ranks_from_sim(sim_matrix, higher_is_better=higher_is_better)
    return topk_accuracy_from_ranks(ranks, ks)
