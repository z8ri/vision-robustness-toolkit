"""tests/test_train_smoke.py — runs train.py's real code path end-to-end
against a synthetic on-disk ImageFolder dataset: the real DefectDataset, the
real split_dataset, the real ConvNeXtTinySWFA backbone (not a toy model),
real A-PhysDeg-driven augmentation, the real AdamW/warmup/cosine schedule,
and the real JSD loss — small epoch count and a reduced image size to keep
this fast on CPU, but every component is the production one train.py uses.

This is the concrete claim behind "point this at a GPU and a real dataset
directory and it would run" — proven by actually running it, just against
synthetic images instead of the (unavailable, confidential) real dataset.
"""
import argparse
import os

import pytest
import torch

from tests._synthetic import make_synthetic_image
from train import run_training


def _make_imagefolder(root, classes_to_counts):
    for class_name, n in classes_to_counts.items():
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n):
            make_synthetic_image(size=(32, 32), seed=hash((class_name, i)) % (2 ** 31)).save(
                os.path.join(class_dir, f"img_{i}.png"))


def _base_args(data_root, checkpoint_dir, **overrides):
    args = argparse.Namespace(
        data_root=str(data_root),
        checkpoint_dir=str(checkpoint_dir),
        device="cpu",
        seed=0,
        img_size=64,
        epochs=2,
        batch_size=2,
        workers=0,
        lr_backbone=5e-5,
        lr_wfa=5e-4,
        lr_head=1e-3,
        weight_decay=0.05,
        label_smoothing=0.1,
        warmup_epochs=1,
        eta_min=1e-7,
        jsd_lambda=6.0,
        patience=100,  # large enough that this short smoke run never early-stops
        aug_strategy="a_physdeg",
        wfa_positions=[4, 8],
        wfa_reduction=32,
        wfa_hf_kernel=3,
        pretrained=False,
        export_onnx=True,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def test_full_training_pipeline_runs_on_synthetic_data(tmp_path):
    torch.manual_seed(0)
    data_root = tmp_path / "data"
    _make_imagefolder(data_root, {"scratch": 10, "dent": 10, "clean_surface": 10})

    checkpoint_dir = tmp_path / "checkpoints"
    args = _base_args(data_root, checkpoint_dir)

    best_path = run_training(args)

    assert os.path.exists(best_path)
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    assert set(ckpt.keys()) >= {"model_state_dict", "classes", "num_classes", "epoch", "val_macro_f1", "config"}
    assert ckpt["num_classes"] == 3
    assert ckpt["classes"] == sorted(["scratch", "dent", "clean_surface"])
    assert 0.0 <= ckpt["val_macro_f1"] <= 1.0

    onnx_path = os.path.join(checkpoint_dir, "best.onnx")
    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_static_physdeg_strategy_also_runs(tmp_path):
    """aug_strategy='physdeg' (no A-PhysDeg controller) is the other supported
    path through train_one_epoch — confirm it doesn't silently rely on
    controller-only state."""
    torch.manual_seed(0)
    data_root = tmp_path / "data"
    _make_imagefolder(data_root, {"a": 8, "b": 8})

    args = _base_args(data_root, tmp_path / "checkpoints", aug_strategy="physdeg", export_onnx=False)
    best_path = run_training(args)
    assert os.path.exists(best_path)


def test_rejects_a_data_root_with_no_classes(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    args = _base_args(empty_root, tmp_path / "checkpoints")
    with pytest.raises(ValueError):
        run_training(args)
