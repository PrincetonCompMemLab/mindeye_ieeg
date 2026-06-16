#!/usr/bin/env python
"""Compare modalities (voltage vs HFB) on the template-matching denoiser proxy.

Starts from the rawest reshaped array per modality, applies smoothing as a *pipeline
step*, selects top-NCSNR channel x timepoint features (NCSNR cached per data-signature,
never recomputed), and runs ``TemplateMatchingEval``. Prints per-modality top-K retrieval
(Pearson and -MSE) vs. the no-model baseline vs. chance, and saves a comparison plot.

Run:  uv run python template_matching_compare.py [--window 25 --stride 10 --top-k 2000]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ieeg_preproc.data import (
    assert_aligned, get_ncsnr, load_reshaped, ncsnr_from_4d, smoothing_signature,
)
from ieeg_preproc.evals import TemplateMatchingEval, run_comparison
from ieeg_preproc.pipeline import IEEGPreprocessingPipeline
from ieeg_preproc.transformers import (
    DomainFeaturizer, FinalFormatter, NCSNRSelector, TemporalSmoother,
)

DERIV = Path("data/derivatives")
OUTPUT_DIR = Path("outputs/hfb")
MODES = {"hfb": "HFB", "voltage": "voltage"}


def _subset_stim_info(stim_info, n):
    """First-``n``-images view of stim_info, keeping the contract keys aligned."""
    stims = stim_info["unique_stimuli"][:n]
    keep = set(stims)
    return {
        "unique_stimuli": stims,
        "stimulus_counts": {s: stim_info["stimulus_counts"][s] for s in stims},
        "stimulus_to_trials": {s: stim_info["stimulus_to_trials"][s] for s in stims},
        "max_repeats": stim_info["max_repeats"],
        "n_unique_images": n,
        "nsd_id_mapping": {s: v for s, v in stim_info["nsd_id_mapping"].items()
                           if s in keep},
    }


def load_modalities(window, stride, max_images=None):
    """Load each modality with a raw reshaped array; compute+cache its (smoothed) NCSNR."""
    smoothing = None if (window == 1 and stride == 1) else (window, stride)
    sig = smoothing_signature(smoothing)
    if max_images is not None:
        sig = f"{sig}-sub{max_images}"   # keep smoke caches separate from full ones
    datasets, ncsnr_by_name = {}, {}
    for name in MODES:
        modality_dir = DERIV / name
        if not (modality_dir / "reshaped_electrode_data.npy").exists():
            print(f"[skip] {name}: no reshaped_electrode_data.npy")
            continue
        print(f"[load] {name} ...")
        X, stim_info = load_reshaped(modality_dir)
        if max_images is not None:
            X = X[:, :max_images]
            stim_info = _subset_stim_info(stim_info, max_images)
        assert_aligned(stim_info, X.shape[1])
        smoother = TemporalSmoother(window, stride)

        def compute(X=X, smoother=smoother):
            sm = smoother.transform(X)
            return ncsnr_from_4d(sm, np.arange(sm.shape[2]))

        ncsnr_by_name[name] = get_ncsnr(modality_dir / "cache", sig, compute)
        datasets[name] = (X, stim_info)
        print(f"  X={X.shape}  ncsnr={ncsnr_by_name[name].shape} (sig={sig})")
    return datasets, ncsnr_by_name


def make_pipeline_factory(window, stride, top_k, ncsnr_by_name):
    def make_pipeline(name):
        return IEEGPreprocessingPipeline([
            ("featurization", DomainFeaturizer(mode=MODES[name])),
            ("smoothing", TemporalSmoother(window, stride)),
            ("selection", NCSNRSelector(ncsnr=ncsnr_by_name[name], top_k=top_k)),
            ("formatter", FinalFormatter(average_reps=False)),
        ])
    return make_pipeline


def plot_comparison(results, top_ks, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, res) in enumerate(results.items()):
        c = colors[i % len(colors)]
        ax.plot(top_ks, [res["top_k_correlation_test"][k] for k in top_ks],
                "o-", color=c, label=f"{name} model (corr)")
        ax.plot(top_ks, [res["top_k_baseline_test"][k] for k in top_ks],
                "^--", color=c, alpha=0.7, label=f"{name} baseline (no model)")
    any_res = next(iter(results.values()))
    ax.plot(top_ks, [any_res["chance"][k] for k in top_ks], "k:", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K retrieval accuracy (held-out test)")
    ax.set_title("Template-matching retrieval: voltage vs HFB")
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

    make_pipeline = make_pipeline_factory(args.window, args.stride, args.top_k, ncsnr_by_name)
    ev = TemplateMatchingEval(top_ks=top_ks, test_size=args.test_size, seed=args.seed)
    results = run_comparison(datasets, make_pipeline, ev, seed=args.seed)

    plot_comparison(results, top_ks, OUTPUT_DIR / "template_matching_voltage_vs_hfb.png")


if __name__ == "__main__":
    main()
