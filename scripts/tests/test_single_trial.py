"""Tests for the single-trial (rep-resolved) expansion used by Tier-1 augmentation.

The averaged path collapses each image's repetitions to one row; single-trial training
instead keeps every valid repetition as its own ``(x_trial, y_image)`` pair (MindEye2's
native fMRI regime), so the model sees many noisy views of each image rather than one clean
memorizable vector. These tests pin the two pure helpers in ``prepare_mindeye_data`` on a
synthetic rep-resolved array: the NaN-column rule must match the averaged path, padded
repetitions must be dropped, and only the requested images may appear (no train/test leak).
"""

import numpy as np

from prepare_mindeye_data import averaged_and_keepmask, expand_train_trials


def _synthetic():
    """(conditions=4, features=5, reps=3), NaN-padded.

    - image 0 rep 2 is a padded (all-NaN) repetition;
    - feature col 4 is a channel missing for image 1 (NaN across all its reps) -> col is
      NaN in the average -> must be dropped, exactly like the averaged path;
    - all other (image, kept-feature, rep) entries are finite.
    """
    X = np.arange(4 * 5 * 3, dtype=np.float64).reshape(4, 5, 3)
    X[0, :, 2] = np.nan            # image 0: only 2 valid repetitions
    X[1, 4, :] = np.nan            # image 1: channel (feature 4) absent in its run
    return X


def test_keepmask_matches_averaged_nan_rule():
    X = _synthetic()
    X_avg, keep = averaged_and_keepmask(X)
    assert X_avg.shape == (4, 5)
    # Only feature 4 is NaN in the average (image 1) -> dropped; the rest kept.
    assert keep.tolist() == [True, True, True, True, False]
    # The padded rep must not poison image 0's average (nanmean ignores it).
    assert not np.isnan(X_avg[0, :4]).any()


def test_expand_drops_padded_reps_and_nan_cols():
    X = _synthetic()
    _, keep = averaged_and_keepmask(X)
    Xtr, img = expand_train_trials(X, image_ids=[0, 1, 2, 3], keep=keep)
    # image 0 -> 2 trials (one rep padded); images 1,2,3 -> 3 each => 11 trials, 4 kept feats.
    assert Xtr.shape == (11, 4)
    assert not np.isnan(Xtr).any()
    assert img.tolist() == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_expand_respects_image_subset_no_leak():
    """Only the requested images (e.g. the train split) may contribute trials."""
    X = _synthetic()
    _, keep = averaged_and_keepmask(X)
    Xtr, img = expand_train_trials(X, image_ids=[2, 3], keep=keep)
    assert set(img.tolist()) == {2, 3}
    assert Xtr.shape == (6, 4)
