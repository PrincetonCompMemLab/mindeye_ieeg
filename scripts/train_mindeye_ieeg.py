#!/usr/bin/env python
"""Fine-tune the NSD-pretrained MindEye2 encoder on iEEG -> bigG CLIP retrieval.

Builds the MindEye2 ridge + ``BrainNetwork`` via the vendored ``utils.prepare_model_and_training``
(``use_prior=False`` -> retrieval head only), initializes the shared backbone from the
multisubject NSD checkpoint (the per-subject ridge is reinitialized for the iEEG feature
dimension), and trains with MindEye2's own recipe (BiMixCo -> SoftCLIP, AdamW + OneCycleLR).
Each epoch prints train AND test top-K retrieval vs. chance; the best (by test top-1) model
and the metric curves are saved under ``outputs/mindeye/``.

Run:
  uv run python scripts/train_mindeye_ieeg.py --num_epochs 150
  uv run python scripts/train_mindeye_ieeg.py --num_epochs 2          # smoke
  uv run python scripts/train_mindeye_ieeg.py --no-pretrained         # from-scratch ablation
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parent.parent
MINDEYE = REPO / "scripts" / "mindeye_offline"
DATA_DIR = REPO / "data" / "derivatives" / "mindeye"
OUT_DIR = REPO / "outputs" / "mindeye"
CKPT_DEFAULT = (REPO / "data" / "MindEyeV2" / "src" / "train_logs"
                / "multisubject_subj01_1024hid_nolow_300ep" / "last.pth")

# MindEye2 architecture config, pinned by the checkpoint (1024hid, bigG token-level).
HIDDEN_DIM, N_BLOCKS, CLIP_EMB_DIM, CLIP_SEQ_DIM, CLIP_SCALE = 1024, 4, 1664, 256, 1.0
TOP_KS = (1, 2, 5, 10, 20, 50, 100)


def load_as(name, path):
    """Load a module from an explicit file path under ``name`` and pin it in sys.modules.

    Used for ``scripts/mindeye_offline/utils.py`` so it (not ``scripts/utils.py``) answers
    every ``import utils`` in this process, regardless of sys.path order.
    """
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    d = os.path.dirname(str(path))
    if d not in sys.path:
        sys.path.append(d)          # so its sibling `models` / `MindEye2` resolve too
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def retrieval_topk(P, T, ks, device="cpu"):
    """Top-K retrieval accuracy (both directions) for normalized preds P vs targets T."""
    import utils  # the pinned MindEye utils
    P, T = P.to(device), T.to(device)
    labels = torch.arange(len(P), device=device)
    fwd = utils.batchwise_cosine_similarity(P, T)   # rank predictions for each target
    bwd = utils.batchwise_cosine_similarity(T, P)   # rank targets for each prediction
    return ({k: utils.topk(fwd, labels, k=k).item() for k in ks},
            {k: utils.topk(bwd, labels, k=k).item() for k in ks})


@torch.no_grad()
def collect_embeddings(model, dl, device, autocast_ctx):
    """Run the encoder over a loader; return L2-normalized flattened (pred, target) on CPU."""
    model.eval()
    preds, targs = [], []
    for voxel, clip_target in dl:
        voxel = voxel.to(device)
        with autocast_ctx():
            _, clip_voxels, _ = model.backbone(model.ridge(voxel, 0))
        preds.append(nn.functional.normalize(clip_voxels.flatten(1).float(), dim=-1).cpu())
        targs.append(nn.functional.normalize(clip_target.flatten(1).float(), dim=-1).cpu())
    return torch.cat(preds), torch.cat(targs)


def print_table(tag, fwd, bwd, n, ks):
    print(f"  [{tag}] K     chance     fwd(brain->img)  bwd(img->brain)")
    for k in ks:
        print(f"        {k:>4d}  {k / n:>7.2%}        {fwd[k]:>7.2%}          {bwd[k]:>7.2%}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--max-lr", type=float, default=3e-4)
    p.add_argument("--mixup-pct", type=float, default=0.33)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pretrained-path", type=Path, default=CKPT_DEFAULT)
    p.add_argument("--no-pretrained", action="store_true",
                   help="skip loading the pretrained backbone (from-scratch ablation).")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR,
                   help="where to write best ckpt / history.json / plot (per-run dir avoids clobber).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true",
                   help="log config + per-epoch metrics to Weights & Biases (off by default).")
    p.add_argument("--wandb-project", default="mindeye_ieeg")
    p.add_argument("--wandb-name", default=None, help="optional run name (defaults to wandb auto).")
    args = p.parse_args()

    device = torch.device(args.device)
    utils = load_as("utils", MINDEYE / "utils.py")   # MindEye losses/retrieval/seed/model builder
    from ieeg_dataset import iEEGDataset
    from me_pretrained import load_pretrained_backbone

    utils.seed_everything(args.seed)

    # --- data (train computes z-score stats; test reuses them) ---
    train_ds = iEEGDataset(args.data_dir / "ieeg_train.pt", args.data_dir / "clip_train.pt")
    test_ds = iEEGDataset(args.data_dir / "ieeg_test.pt", args.data_dir / "clip_test.pt",
                          mean=train_ds.mean, std=train_ds.std)
    F = train_ds.ieeg.shape[1]
    print(f"[data] train={len(train_ds)} test={len(test_ds)} feature_dim={F}")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    # Equal-sized train subset for a like-for-like train retrieval readout each epoch.
    n_eval = len(test_ds)
    train_eval_dl = DataLoader(Subset(train_ds, list(range(n_eval))),
                               batch_size=args.batch_size, shuffle=False)

    # --- model (vendored helper; retrieval head only) ---
    model = utils.prepare_model_and_training(
        num_voxels_list=[F], n_blocks=N_BLOCKS, hidden_dim=HIDDEN_DIM,
        clip_emb_dim=CLIP_EMB_DIM, clip_seq_dim=CLIP_SEQ_DIM, clip_scale=CLIP_SCALE,
        use_prior=False,
    )
    if not args.no_pretrained:
        rep = load_pretrained_backbone(model.backbone, args.pretrained_path, map_location="cpu")
        print(f"[pretrained] {args.pretrained_path.name}: kept={rep['kept']} "
              f"missing={len(rep['missing'])} unexpected={len(rep['unexpected'])} "
              f"skipped_ridge={rep['skipped_ridge']} skipped_prior={rep['skipped_prior']}")
        if rep["unexpected"] or rep["missing"]:
            print(f"  WARNING missing={rep['missing'][:4]}... unexpected={rep['unexpected'][:4]}...")
    else:
        print("[pretrained] skipped (--no-pretrained): backbone randomly initialized")
    model.to(device)

    # --- optimizer + schedule (MindEye2 recipe, kept as-is) ---
    no_decay = lambda n: "clip_proj" in n          # retrieval head trained without weight decay
    opt_groups = [
        {"params": list(model.ridge.parameters()), "weight_decay": 1e-2},
        {"params": [p for n, p in model.backbone.named_parameters() if not no_decay(n)],
         "weight_decay": 1e-2},
        {"params": [p for n, p in model.backbone.named_parameters() if no_decay(n)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(opt_groups, lr=args.max_lr)
    steps_per_epoch = len(train_dl)
    total_steps = args.num_epochs * steps_per_epoch
    pct_start = min(0.3, 2.0 / args.num_epochs)    # = 2/num_epochs for real runs; safe for smoke
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.max_lr, total_steps=total_steps,
        final_div_factor=1000, last_epoch=-1, pct_start=pct_start)

    mixup_until = int(args.mixup_pct * args.num_epochs)
    soft_temps = utils.cosine_anneal(0.004, 0.0075, max(1, args.num_epochs - mixup_until))

    use_amp = device.type == "cuda"
    autocast_ctx = (lambda: torch.autocast(device_type="cuda", dtype=torch.float16)) if use_amp \
        else (lambda: torch.autocast(device_type="cpu", enabled=False))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.wandb_name, config={
            "num_epochs": args.num_epochs, "batch_size": args.batch_size,
            "max_lr": args.max_lr, "mixup_pct": args.mixup_pct, "seed": args.seed,
            "pretrained": not args.no_pretrained, "feature_dim": F,
            "hidden_dim": HIDDEN_DIM, "n_blocks": N_BLOCKS, "clip_emb_dim": CLIP_EMB_DIM,
            "clip_seq_dim": CLIP_SEQ_DIM, "clip_scale": CLIP_SCALE,
            "n_train": len(train_ds), "n_test": len(test_ds), "device": str(device),
        })
        print(f"[wandb] logging to project={args.wandb_project!r} run={run.name}")

    history = {"epoch": [], "loss": [], "lr": [], "train_top1": [], "test_top1": [],
               "test_fwd": [], "test_bwd": []}
    best_test_top1, best_path = -1.0, out_dir / "mindeye_ieeg_best.pt"

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, use_mixco = 0.0, epoch < mixup_until
        for voxel, clip_target in train_dl:
            voxel = voxel.to(device)
            clip_target = clip_target.to(device).float()
            optimizer.zero_grad()
            if use_mixco:
                voxel, perm, betas, select = utils.mixco(voxel)
            with autocast_ctx():
                _, clip_voxels, _ = model.backbone(model.ridge(voxel, 0))
                preds = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                targs = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                if use_mixco:
                    loss = utils.mixco_nce(preds, targs, temp=0.006,
                                           perm=perm, betas=betas, select=select)
                else:
                    temp = soft_temps[epoch - mixup_until]
                    loss = utils.soft_clip_loss(preds, targs, temp=temp)
                loss = loss * CLIP_SCALE
            utils.check_loss(loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= steps_per_epoch

        # --- per-epoch retrieval readout (train subset + full test) ---
        trP, trT = collect_embeddings(model, train_eval_dl, device, autocast_ctx)
        tr_fwd, _ = retrieval_topk(trP, trT, TOP_KS, device=device)
        P, T = collect_embeddings(model, test_dl, device, autocast_ctx)
        te_fwd, te_bwd = retrieval_topk(P, T, TOP_KS, device=device)

        lr_now = scheduler.get_last_lr()[0]
        mode = "mixco" if use_mixco else "softclip"
        print(f"epoch {epoch:>3d}/{args.num_epochs} [{mode:>8}] loss={epoch_loss:.4f} lr={lr_now:.2e} "
              f"| train top1={tr_fwd[1]:.2%}  test top1={te_fwd[1]:.2%} "
              f"top5={te_fwd[5]:.2%} top100={te_fwd[100]:.2%} (chance@1={1/len(test_ds):.2%})")

        history["epoch"].append(epoch); history["loss"].append(epoch_loss)
        history["lr"].append(lr_now); history["train_top1"].append(tr_fwd[1])
        history["test_top1"].append(te_fwd[1]); history["test_fwd"].append(te_fwd)
        history["test_bwd"].append(te_bwd)

        if run is not None:
            run.log({"epoch": epoch, "loss": epoch_loss, "lr": lr_now, "mode": mode,
                     "train/top1": tr_fwd[1], "test/top1": te_fwd[1],
                     **{f"test/fwd_top{k}": te_fwd[k] for k in TOP_KS},
                     **{f"test/bwd_top{k}": te_bwd[k] for k in TOP_KS}}, step=epoch)

        if te_fwd[1] >= best_test_top1:
            best_test_top1 = te_fwd[1]
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "test_fwd": te_fwd, "test_bwd": te_bwd, "feature_dim": F,
                        "ieeg_mean": train_ds.mean, "ieeg_std": train_ds.std},
                       best_path)

    # --- final summary table (last epoch) + artifacts ---
    print(f"\nFinal test retrieval (n_test={len(test_ds)}):")
    print_table("test", te_fwd, te_bwd, len(test_ds), TOP_KS)
    print(f"best test top-1 over training: {best_test_top1:.2%} -> {best_path}")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(TOP_KS, [te_fwd[k] for k in TOP_KS], "o-", label="iEEG MindEye2 (test, fwd)")
    ax.plot(TOP_KS, [k / len(test_ds) for k in TOP_KS], "k:", label="chance")
    ax.set_xscale("log"); ax.set_xlabel("K"); ax.set_ylabel("Top-K CLIP retrieval (held-out test)")
    ax.set_title(f"MindEye2 iEEG->bigG retrieval (pretrained={not args.no_pretrained})")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); fig.tight_layout()
    fig.savefig(out_dir / "mindeye_ieeg_retrieval.png", dpi=120)
    print(f"saved {out_dir}/history.json and mindeye_ieeg_retrieval.png")

    if run is not None:
        import wandb
        run.summary["best_test_top1"] = best_test_top1
        run.log({"retrieval_curve": wandb.Image(fig)})
        run.finish()


if __name__ == "__main__":
    main()
