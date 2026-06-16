"""Load the transferable backbone weights from a MindEye2 checkpoint.

The released checkpoint is an accelerate/deepspeed dict whose ``optimizer_state_dict``
pickles ``deepspeed.*`` objects; unpickling those triggers deepspeed's CUDA op-builder
checks, which raise on CPU-only nodes (``CUDA_HOME does not exist``). We only need
``model_state_dict``, so we shadow ``deepspeed`` with a no-op stub during the load — this
works identically on CPU and GPU regardless of whether real deepspeed is installed.

``filter_backbone`` keeps the shared ``backbone.*`` weights (stripping the prefix) and drops
the per-subject/voxel ``ridge.*`` and the ``diffusion_prior.*`` keys (out of scope here).
"""

import importlib.abc
import importlib.machinery
import sys
import types

import torch


def filter_backbone(state_dict):
    """Keep only ``backbone.*`` entries, with the ``backbone.`` prefix stripped."""
    prefix = "backbone."
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


class _DeepspeedStub(types.ModuleType):
    """Package-like module that fabricates a no-op class for any attribute access."""

    __path__ = []  # marks it as a package so ``deepspeed.x.y`` submodules resolve

    def __getattr__(self, name):
        cls = type(name, (), {
            "__init__": lambda self, *a, **k: None,
            "__setstate__": lambda self, state: self.__dict__.update(
                state if isinstance(state, dict) else {}),
        })
        setattr(self, name, cls)
        return cls


class _DeepspeedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == "deepspeed" or fullname.startswith("deepspeed."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return _DeepspeedStub(spec.name)

    def exec_module(self, module):
        pass


def load_model_state_dict(ckpt_path, map_location="cpu"):
    """Return a MindEye2 checkpoint's ``model_state_dict``, stubbing deepspeed during load."""
    shadowed = {k: sys.modules.pop(k) for k in list(sys.modules)
                if k == "deepspeed" or k.startswith("deepspeed.")}
    finder = _DeepspeedFinder()
    sys.meta_path.insert(0, finder)
    try:
        ckpt = torch.load(ckpt_path, map_location=map_location,
                          weights_only=False, mmap=True)
    finally:
        sys.meta_path.remove(finder)
        sys.modules.update(shadowed)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_pretrained_backbone(backbone, ckpt_path, map_location="cpu"):
    """Load the pretrained ``backbone.*`` weights into ``backbone`` (``strict=False``).

    Returns a report dict: kept count, missing/unexpected key lists, and how many
    ``ridge.*`` / ``diffusion_prior.*`` keys were skipped.
    """
    sd = load_model_state_dict(ckpt_path, map_location)
    bb = filter_backbone(sd)
    missing, unexpected = backbone.load_state_dict(bb, strict=False)
    return {
        "kept": len(bb),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped_ridge": sum(1 for k in sd if k.startswith("ridge.")),
        "skipped_prior": sum(1 for k in sd if k.startswith("diffusion_prior.")),
    }
