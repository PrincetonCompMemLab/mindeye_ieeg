"""Tests for selecting/loading the pretrained backbone from a MindEye2 checkpoint.

The transferable part of the NSD-pretrained checkpoint is the shared ``backbone.*``; the
``ridge.*`` (per-subject/voxel) and ``diffusion_prior.*`` keys must be dropped. These tests
pin that filtering on a synthetic state dict, so they need neither the real (deepspeed)
checkpoint nor a GPU.
"""

import sys
from pathlib import Path

import torch

MINDEYE_DIR = Path(__file__).resolve().parent.parent / "mindeye_offline"
if str(MINDEYE_DIR) not in sys.path:
    sys.path.append(str(MINDEYE_DIR))

from MindEye2 import BrainNetwork  # noqa: E402

from me_pretrained import filter_backbone  # noqa: E402

H, CLIP_SEQ, CLIP_SIZE, N_BLOCKS = 16, 4, 8, 2


def _backbone():
    return BrainNetwork(h=H, in_dim=H, out_dim=CLIP_SEQ * CLIP_SIZE, seq_len=1,
                        n_blocks=N_BLOCKS, clip_size=CLIP_SIZE, clip_scale=1)


def test_filter_keeps_only_backbone_and_strips_prefix():
    sd = {
        "ridge.linears.0.weight": torch.zeros(2),
        "ridge.linears.3.bias": torch.zeros(2),
        "backbone.backbone_linear.weight": torch.zeros(2),
        "backbone.clip_proj.0.weight": torch.zeros(2),
        "diffusion_prior.net.foo": torch.zeros(2),
    }
    bb = filter_backbone(sd)
    assert set(bb.keys()) == {"backbone_linear.weight", "clip_proj.0.weight"}


def test_filtered_backbone_loads_with_no_missing_or_unexpected():
    """A real backbone's own (prefixed) weights, plus junk ridge/prior keys, must load
    cleanly into a fresh backbone after filtering."""
    src = _backbone()
    state = {f"backbone.{k}": v for k, v in src.state_dict().items()}
    state["ridge.linears.0.weight"] = torch.randn(H, 99)      # wrong shape on purpose
    state["diffusion_prior.net.x"] = torch.randn(3)
    filtered = filter_backbone(state)

    fresh = _backbone()
    missing, unexpected = fresh.load_state_dict(filtered, strict=False)
    assert len(missing) == 0 and len(unexpected) == 0
