#!/usr/bin/env python3
"""train.py — real training entry point.

DefectDataset (data/dataset.py) -> A-PhysDeg or static PhysDeg triplet
augmentation -> ConvNeXtTinySWFA (models/backbone.py) -> AdamW with three
differential-LR parameter groups -> linear warmup then cosine LR ->
CE(clean, label_smoothing) + jsd_lambda * JSD(clean, deg1, deg2).

Every default below (AdamW, lr_backbone=5e-5, lr_head=1e-3, lr_wfa=5e-4,
weight_decay=0.05, label_smoothing=0.1, warmup_epochs=5, eta_min=1e-7,
epochs=100, batch_size=16, patience=30, jsd_lambda=6.0) mirrors the real
training config this project's design is based on — read directly (read-only)
from that codebase, not guessed.

This script has never been run against the real private dataset (not on this
machine) or a GPU — see README. Point --data_root at a real directory in the
data/dataset.py layout and run this on a GPU, and that's the only thing
missing, not the wiring. tests/test_train_smoke.py runs this exact code path
end-to-end on a synthetic on-disk dataset to prove it, on CPU, in a few
seconds.

Robustness evaluation, calibration, and ONNX export are deliberately kept
separate rather than folded into this script — eval/cube_protocol.py,
calibration/selective.py, and deploy/export.py are all model-agnostic
(take a predict_fn/model, not a hardcoded one) and already work against a
ConvNeXtTinySWFA checkpoint saved here; --export_onnx below is the one
convenience wired in directly, since it's a one-line call.
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from augmentation.a_physdeg import AdaptivePhysDegController, APhysDegTransform
from augmentation.physdeg import PhysDegTransform
from data.dataset import DefectDataset, build_transforms, split_dataset
from eval.metrics import macro_f1
from losses.jsd import jsd_loss
from models.backbone import ConvNeXtTinySWFA


def build_param_groups(model: nn.Module, lr_backbone: float, lr_wfa: float, lr_head: float):
    """Three differential-LR groups: backbone < wfa < head, matching the real
    training config (a freshly-initialized attention module and head need to
    move faster than an ImageNet-pretrained backbone)."""
    backbone_params, wfa_params, head_params = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("wfa_modules."):
            wfa_params.append(p)
        elif name.startswith("classifier."):
            head_params.append(p)
        else:
            backbone_params.append(p)
    groups = [{"params": backbone_params, "lr": lr_backbone}]
    if wfa_params:
        groups.append({"params": wfa_params, "lr": lr_wfa})
    groups.append({"params": head_params, "lr": lr_head})
    return groups


def set_warmup_lr(optimizer, base_lrs, epoch: int, warmup_epochs: int) -> None:
    scale = (epoch + 1) / warmup_epochs
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * scale


def difficulty_from_model(model: nn.Module, images: torch.Tensor, labels: torch.Tensor) -> float:
    """A-PhysDeg's difficulty signal: squashed CE loss on a single-degradation
    diagnostic batch. Temporarily flips to eval() so diagnostics never affect
    BatchNorm running stats mid-epoch, then restores the prior mode."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        loss = nn.functional.cross_entropy(model(images), labels)
    if was_training:
        model.train()
    return float(torch.tanh(loss / 2.0))


def train_one_epoch(model, loader, optimizer, device, jsd_lambda: float, label_smoothing: float) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss, n_batches = 0.0, 0
    for images, labels in loader:
        # images: (B, 3, C, H, W) — the (clean, deg1, deg2) triplet PhysDegTransform produces.
        batch_size = images.shape[0]
        clean, deg1, deg2 = images[:, 0], images[:, 1], images[:, 2]
        images_all = torch.cat([clean, deg1, deg2], dim=0).to(device)
        labels = labels.to(device)

        logits_all = model(images_all)
        logits_clean, logits_deg1, logits_deg2 = torch.split(logits_all, batch_size)

        loss = ce(logits_clean, labels) + jsd_lambda * jsd_loss(logits_clean, logits_deg1, logits_deg2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_macro_f1(model, loader, device, num_classes: int) -> float:
    model.eval()
    preds, labels_out = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        preds.extend(logits.argmax(dim=1).cpu().tolist())
        labels_out.extend(labels.tolist())
    return macro_f1(preds, labels_out, num_classes)


def run_training(args: argparse.Namespace) -> str:
    """Runs the full training loop; returns the path to the best checkpoint."""
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    train_dir, val_dir, _test_dir = split_dataset(args.data_root, seed=args.seed)
    train_base_transform = build_transforms(args.img_size, train=True)
    val_transform = build_transforms(args.img_size, train=False)

    train_ds = DefectDataset(train_dir)  # scan once; attach the PhysDeg transform below
    num_classes = len(train_ds.classes)

    if args.aug_strategy == "a_physdeg":
        controller = AdaptivePhysDegController()
        train_ds.transform = APhysDegTransform(train_base_transform, controller)
    else:
        controller = None
        train_ds.transform = PhysDegTransform(train_base_transform)

    val_ds = DefectDataset(val_dir, transform=val_transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ConvNeXtTinySWFA(
        num_classes=num_classes, wfa_positions=args.wfa_positions,
        wfa_reduction=args.wfa_reduction, wfa_hf_kernel=args.wfa_hf_kernel,
        pretrained=args.pretrained,
    ).to(device)

    param_groups = build_param_groups(model, args.lr_backbone, args.lr_wfa, args.lr_head)
    base_lrs = [g["lr"] for g in param_groups]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    cosine_epochs = max(1, args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.eta_min)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    best_f1, epochs_no_improve = -1.0, 0

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            set_warmup_lr(optimizer, base_lrs, epoch, args.warmup_epochs)

        if controller is not None:
            controller.set_progress(epoch / max(1, args.epochs - 1))
            controller.run_diagnostics(
                train_ds.samples, train_base_transform,
                difficulty_fn=lambda imgs, lbls: difficulty_from_model(model, imgs.to(device), lbls.to(device)),
                batch_size=args.batch_size, split="train",
            )

        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.jsd_lambda, args.label_smoothing)
        val_f1 = evaluate_macro_f1(model, val_loader, device, num_classes)

        if epoch >= args.warmup_epochs:
            scheduler.step()

        print(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, epochs_no_improve = val_f1, 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "num_classes": num_classes,
                "epoch": epoch,
                "val_macro_f1": val_f1,
                "config": vars(args),
            }, best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"early stopping at epoch {epoch + 1} (no val improvement for {args.patience} epochs)")
                break

    if args.export_onnx and os.path.exists(best_path):
        from deploy.export import export_to_onnx
        dummy = torch.randn(1, 3, args.img_size, args.img_size)
        onnx_path = os.path.join(args.checkpoint_dir, "best.onnx")
        export_to_onnx(model.cpu(), dummy, onnx_path)
        print(f"exported ONNX model to {onnx_path}")

    return best_path


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", required=True,
                    help="root_dir/<class_name>/<image>, or root_dir/{train,val,test}/<class_name>/ "
                         "if already split (see data/dataset.py)")
    p.add_argument("--checkpoint_dir", default="./checkpoints")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_wfa", type=float, default=5e-4)
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--eta_min", type=float, default=1e-7)
    p.add_argument("--jsd_lambda", type=float, default=6.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--aug_strategy", choices=["physdeg", "a_physdeg"], default="a_physdeg")
    p.add_argument("--wfa_positions", type=int, nargs="+", default=[4, 8])
    p.add_argument("--wfa_reduction", type=int, default=32)
    p.add_argument("--wfa_hf_kernel", type=int, default=3)
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--export_onnx", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    best_path = run_training(args)
    print(f"best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
