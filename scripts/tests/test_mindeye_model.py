"""Characterization tests for the vendored MindEye2 model classes.

The classes already exist in ``scripts/mindeye_offline/MindEye2.py`` (so these are GREEN
from the start); they pin the input/output shapes the iEEG training driver relies on:
ridge maps ``(B, 1, F) -> (B, 1, h)`` and the backbone emits both a ``backbone`` tensor and
a ``clip_voxels`` retrieval tensor of shape ``(B, clip_seq, clip_size)``. Small dims keep it
fast; the real 1024-hid / 256x1664 config is exercised in the smoke run.
"""

import sys
from pathlib import Path

import torch

# MindEye2 lives in the vendored package dir, not on the default test path.
MINDEYE_DIR = Path(__file__).resolve().parent.parent / "mindeye_offline"
if str(MINDEYE_DIR) not in sys.path:
    sys.path.append(str(MINDEYE_DIR))

from MindEye2 import BrainNetwork, MindEyeModule, RidgeRegression  # noqa: E402

H, CLIP_SEQ, CLIP_SIZE, N_BLOCKS = 16, 4, 8, 2


def _backbone():
    return BrainNetwork(h=H, in_dim=H, out_dim=CLIP_SEQ * CLIP_SIZE, seq_len=1,
                        n_blocks=N_BLOCKS, clip_size=CLIP_SIZE, clip_scale=1)


def test_ridge_shape():
    """RidgeRegression maps (B, 1, F) -> (B, 1, h) for a single subject."""
    F = 20
    ridge = RidgeRegression([F], out_features=H)
    out = ridge(torch.randn(3, 1, F), 0)
    assert out.shape == (3, 1, H)


def test_backbone_shapes_and_finite():
    """Backbone returns backbone + clip_voxels of (B, clip_seq, clip_size), all finite."""
    backbone, clip_voxels, _ = _backbone()(torch.randn(3, 1, H))
    assert backbone.shape == (3, CLIP_SEQ, CLIP_SIZE)
    assert clip_voxels.shape == (3, CLIP_SEQ, CLIP_SIZE)
    assert torch.isfinite(backbone).all() and torch.isfinite(clip_voxels).all()


def test_ridge_then_backbone_end_to_end():
    """The composed ridge -> backbone path produces the retrieval tensor."""
    F = 20
    model = MindEyeModule()
    model.ridge = RidgeRegression([F], out_features=H)
    model.backbone = _backbone()
    _, clip_voxels, _ = model.backbone(model.ridge(torch.randn(5, 1, F), 0))
    assert clip_voxels.shape == (5, CLIP_SEQ, CLIP_SIZE)
