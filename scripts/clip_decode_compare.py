#!/usr/bin/env python
"""Compare modalities (voltage vs HFB) on CLIP-embedding decoding.

Fits RidgeCV from rep-averaged brain features to the 1024-d pooled CLIP image
embeddings (``data/derivatives/images_clipemb``), and reports cosine top-K retrieval of
the predicted embedding against held-out true embeddings vs. chance, per modality. Reuses
the same raw-array loading, smoothing-as-a-pipeline-step, and cached-once NCSNR selection
as the template-matching driver.

Embedding rows are in unique-image order (the order ``unique_image_names`` was encoded in
``train.py``), which matches the conditions axis order in ``stim_info['unique_stimuli']``;
``assert_aligned`` enforces the row count and (for subsets) we slice the same prefix.

Run:  uv run python clip_decode_compare.py [--window 25 --stride 10 --top-k 2000]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ieeg_preproc.data import assert_aligned
from ieeg_preproc.evals import CLIPDecodingEval, run_comparison
from ieeg_preproc.pipeline import IEEGPreprocessingPipeline
from ieeg_preproc.transformers import (
    DomainFeaturizer, FinalFormatter, NCSNRSelector, TemporalSmoother,
)
from template_matching_compare import DERIV, MODES, load_modalities

EMB_PATH = DERIV / "images_clipemb"
OUTPUT_DIR = Path("outputs/clip")


def load_embeddings(max_images=None):
    """Load the 1024-d pooled CLIP embeddings (unique-image order)."""
    y = torch.load(EMB_PATH, weights_only=True).numpy().astype(np.float64)
    if max_images is not None:
        y = y[:max_images]
    print(f"[load] images_clipemb y={y.shape}")
    return y


def make_pipeline_factory(window, stride, top_k, ncsnr_by_name):
    def make_pipeline(name):
        return IEEGPreprocessingPipeline([
            ("featurization", DomainFeaturizer(mode=MODES[name])),
            ("smoothing", TemporalSmoother(window, stride)),
            ("selection", NCSNRSelector(ncsnr=ncsnr_by_name[name], top_k=top_k)),
            ("formatter", FinalFormatter(average_reps=True)),
        ])
    return make_pipeline


def plot_comparison(results, top_ks, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, res) in enumerate(results.items()):
        ax.plot(top_ks, [res["top_k_retrieval_test"][k] for k in top_ks],
                "o-", color=colors[i % len(colors)], label=f"{name} (test R^2={res['test_r2']:.3f})")
    any_res = next(iter(results.values()))
    ax.plot(top_ks, [any_res["chance"][k] for k in top_ks], "k:", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K CLIP retrieval accuracy (held-out test)")
    ax.set_title("CLIP-embedding decoding: voltage vs HFB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"\nsaved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--window", type=int, default=25)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--top-k", type=int, default=2000)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=None,
                   help="subset to the first N images (lighter smoke run; separate cache).")
    args = p.parse_args()

    top_ks = (1, 2, 5, 10, 20, 50, 100)
    datasets, ncsnr_by_name = load_modalities(args.window, args.stride, args.max_images)
    if not datasets:
        raise SystemExit("no modalities with a raw reshaped array found.")
    y = load_embeddings(args.max_images)
    # Guard order alignment against every dataset's conditions axis before modeling.
    for name, (_, stim_info) in datasets.items():
        assert_aligned(stim_info, stim_info["n_unique_images"], y=y)

    make_pipeline = make_pipeline_factory(args.window, args.stride, args.top_k, ncsnr_by_name)
    ev = CLIPDecodingEval(top_ks=top_ks, test_size=args.test_size, seed=args.seed)
    results = run_comparison(datasets, make_pipeline, ev, y=y, seed=args.seed)

    plot_comparison(results, top_ks, OUTPUT_DIR / "clip_decode_voltage_vs_hfb.png")


if __name__ == "__main__":
    main()
