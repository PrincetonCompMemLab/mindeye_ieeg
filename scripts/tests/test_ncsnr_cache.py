"""Tests for the NCSNR cache helper: compute once per data-signature, never recompute."""

import json

import numpy as np

from ieeg_preproc.data import get_ncsnr, smoothing_signature


def _spy_compute(value):
    """Return a zero-arg compute thunk plus a mutable call counter."""
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return value

    return compute, calls


def test_smoothing_signature():
    assert smoothing_signature(None) == "raw"
    assert smoothing_signature((25, 10)) == "smooth-w25-s10"
    assert smoothing_signature((3, 1)) == "smooth-w3-s1"


def test_cache_miss_computes_and_writes(tmp_path):
    val = np.arange(6, dtype=float).reshape(2, 3)
    compute, calls = _spy_compute(val)

    out = get_ncsnr(tmp_path, "raw", compute)

    assert calls["n"] == 1
    np.testing.assert_array_equal(out, val)
    assert (tmp_path / "ncsnr__raw.npy").exists()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert "raw" in manifest


def test_cache_hit_does_not_recompute(tmp_path):
    val = np.arange(6, dtype=float).reshape(2, 3)
    compute, calls = _spy_compute(val)

    first = get_ncsnr(tmp_path, "raw", compute)
    second = get_ncsnr(tmp_path, "raw", compute)

    assert calls["n"] == 1  # second call served from disk
    np.testing.assert_array_equal(first, second)


def test_distinct_signatures_are_separate(tmp_path):
    a, calls_a = _spy_compute(np.zeros((2, 2)))
    b, calls_b = _spy_compute(np.ones((2, 2)))

    out_a = get_ncsnr(tmp_path, "raw", a)
    out_b = get_ncsnr(tmp_path, "smooth-w25-s10", b)

    assert calls_a["n"] == 1 and calls_b["n"] == 1
    np.testing.assert_array_equal(out_a, np.zeros((2, 2)))
    np.testing.assert_array_equal(out_b, np.ones((2, 2)))
    assert (tmp_path / "ncsnr__raw.npy").exists()
    assert (tmp_path / "ncsnr__smooth-w25-s10.npy").exists()
