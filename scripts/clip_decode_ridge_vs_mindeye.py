#!/usr/bin/env python
"""Compare HFB->bigG CLIP decoding: RidgeCV vs fine-tuned MindEye2, at varying top-K.

Companion to ``clip_decode_compare.py`` (which compares modalities). Here the modality is
fixed (HFB) and the *decoder* is swept: the linear RidgeCV baseline vs the NSD-pretrained
MindEye2 encoder fine-tuned on iEEG (and the from-scratch ablation). All curves are top-K
cosine retrieval of the predicted bigG embedding against the held-out true ones on the
**same** test images.

Both decoders consume the prepared tensors in ``data/derivatives/mindeye/`` (the exact HFB
pipeline output -- smooth -> top-NCSNR -> rep-average -> NaN-drop -- split with
``test_size=0.2, seed=42``). MindEye2 was trained on them directly; the ridge curve is
computed here from the same split, reusing ``CLIPDecodingEval``'s own transform helpers
(z-score+PCA brain features, target PCA, RidgeCV, cosine top-K) so the metric is identical to
the existing bigG plot -- without reloading the ~26 GB raw arrays. Because the split indices
match, the held-out set is identical and ``chance = K/n_test`` is shared.

Note: ridge retrieves in the PCA-reduced target space; MindEye2 in the full 256x1664 token
space. Both are cosine top-K on the identical held-out images -- the honest "each decoder in
its native space" comparison (labels state the space). MindEye2 curves come from each run's
best-by-test-top1 checkpoint (``mindeye_ieeg_best.pt``), since retrieval peaks early then
overfits.

Run:
  uv run python scripts/clip_decode_ridge_vs_mindeye.py
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV

from ieeg_preproc.evals import _FeaturePrep
from ieeg_preproc.metrics import cosine_matrix, topk_from_similarity
from clip_decode_compare import OUTPUT_DIR

TOP_KS = (1, 2, 5, 10, 20, 50, 100)
DATA_DIR = Path("data/derivatives/mindeye")


def ridge_curve(data_dir, top_ks, target_pca=512, seed=42):
    """RidgeCV HFB->bigG test top-K, replicating CLIPDecodingEval on the prepared split."""
    X_tr = torch.load(data_dir / "ieeg_train.pt", weights_only=True).float().numpy()
    X_te = torch.load(data_dir / "ieeg_test.pt", weights_only=True).float().numpy()
    # bigG targets are fp16 on disk; keep fp32 (halves the target-PCA memory).
    y_tr = torch.load(data_dir / "clip_train.pt", weights_only=True).float().reshape(len(X_tr), -1).numpy()
    y_te = torch.load(data_dir / "clip_test.pt", weights_only=True).float().reshape(len(X_te), -1).numpy()

    # Brain features: z-score + PCA(98% var), fit on train (CLIPDecodingEval defaults).
    xprep = _FeaturePrep(zscore=True, do_pca=True, n_pcs=None, pca_var_threshold=0.98).fit(X_tr)
    Xtr, Xte = xprep.transform(X_tr), xprep.transform(X_te)
    # Target-side PCA fit on train embeddings only (matches the bigG plot's target_pca).
    yprep = _FeaturePrep(zscore=False, do_pca=True, n_pcs=int(target_pca)).fit(y_tr)
    Ytr, Yte = yprep.transform(y_tr), yprep.transform(y_te)

    ridge = RidgeCV(alphas=np.logspace(-4, 10, 80), alpha_per_target=True).fit(Xtr, Ytr)
    curve = topk_from_similarity(cosine_matrix(ridge.predict(Xte), Yte), top_ks)
    return curve, len(X_te), int(yprep.pca_.n_components_), float(ridge.score(Xte, Yte))


def mindeye_curve(ckpt_path, top_ks):
    """Read the saved best-checkpoint test top-K (forward) curve from a training run."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    fwd = ckpt["test_fwd"]                       # {k: acc}; int keys (torch.save preserves them)
    return {k: float(fwd[k]) for k in top_ks}, int(ckpt.get("epoch", -1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--target-pca", type=int, default=512,
                   help="target-side PCA for the ridge bigG decoder (matches the bigG plot).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-dir", type=Path, default=Path("outputs/mindeye/pretrained_30ep"))
    p.add_argument("--scratch-dir", type=Path, default=Path("outputs/mindeye/scratch_30ep"),
                   help="from-scratch ablation run dir; pass '' to omit that curve.")
    p.add_argument("--out", type=Path,
                   default=OUTPUT_DIR / "clip_decode_bigG_ridge_vs_mindeye.png")
    args = p.parse_args()

    rc, n_test, npc, r2 = ridge_curve(args.data_dir, TOP_KS, args.target_pca, args.seed)
    chance = {k: k / n_test for k in TOP_KS}
    curves = [(f"ridge (bigG PCA={npc}, R²={r2:.3f})", rc)]

    me_specs = [("MindEye2 pretrained", args.pretrained_dir)]
    if str(args.scratch_dir):
        me_specs.append(("MindEye2 from-scratch", args.scratch_dir))
    for label, d in me_specs:
        ckpt = d / "mindeye_ieeg_best.pt"
        if not ckpt.exists():
            print(f"[warn] {ckpt} missing -- skipping '{label}'")
            continue
        curve, ep = mindeye_curve(ckpt, TOP_KS)
        curves.append((f"{label} (full bigG, best@ep{ep})", curve))

    # --- plot (style mirrors clip_decode_compare.plot_comparison) ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, curve) in enumerate(curves):
        ax.plot(TOP_KS, [curve[k] for k in TOP_KS], "o-",
                color=colors[i % len(colors)], label=label)
    ax.plot(TOP_KS, [chance[k] for k in TOP_KS], "k:", label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("Top-K CLIP retrieval accuracy (held-out test)")
    ax.set_title("HFB->bigG decoding: ridge vs MindEye2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)

    print(f"\n[n_test={n_test}] curves:")
    for label, curve in curves:
        print(f"  {label:<46} " + " ".join(f"K{k}={curve[k]:.1%}" for k in TOP_KS))
    print(f"  {'chance':<46} " + " ".join(f"K{k}={chance[k]:.1%}" for k in TOP_KS))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
