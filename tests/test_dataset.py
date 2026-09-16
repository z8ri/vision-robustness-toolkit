import os

import pytest
import torch

from data.dataset import DefectDataset, build_transforms, split_dataset
from tests._synthetic import make_synthetic_image


def _make_imagefolder(root, classes_to_counts):
    for class_name, n in classes_to_counts.items():
        class_dir = os.path.join(root, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(n):
            make_synthetic_image(size=(32, 32), seed=hash((class_name, i)) % (2 ** 31)).save(
                os.path.join(class_dir, f"img_{i}.png"))


def test_defect_dataset_loads_and_labels_by_sorted_class_name(tmp_path):
    _make_imagefolder(tmp_path, {"scratch": 3, "dent": 2})
    ds = DefectDataset(str(tmp_path))
    assert ds.classes == ["dent", "scratch"]  # alphabetical, not insertion order
    assert ds.class_to_idx == {"dent": 0, "scratch": 1}
    assert len(ds) == 5
    labels = [label for _, label in ds.samples]
    assert labels.count(0) == 2 and labels.count(1) == 3


def test_defect_dataset_applies_transform_and_returns_expected_shape(tmp_path):
    _make_imagefolder(tmp_path, {"a": 2})
    transform = build_transforms(img_size=48, train=False)
    ds = DefectDataset(str(tmp_path), transform=transform)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 48, 48)  # grayscale replicated to 3 channels
    assert isinstance(label, int)


def test_missing_root_dir_raises():
    with pytest.raises(FileNotFoundError):
        DefectDataset("/no/such/path/at/all")


def test_root_with_no_class_subdirs_raises(tmp_path):
    (tmp_path / "not_a_class.txt").write_text("hello")
    with pytest.raises(ValueError):
        DefectDataset(str(tmp_path))


def test_split_dataset_respects_ratios_per_class(tmp_path):
    _make_imagefolder(tmp_path, {"a": 20, "b": 20})
    train_dir, val_dir, test_dir = split_dataset(str(tmp_path), seed=0,
                                                  train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    for split_dir, expected in ((train_dir, 14), (val_dir, 3), (test_dir, 3)):
        for class_name in ("a", "b"):
            n = len(os.listdir(os.path.join(split_dir, class_name)))
            assert n == expected, f"{split_dir}/{class_name} had {n} files, expected {expected}"


def test_split_dataset_is_reproducible_with_the_same_seed(tmp_path):
    _make_imagefolder(tmp_path, {"a": 20})
    train_dir, _, _ = split_dataset(str(tmp_path), seed=42)
    first = sorted(os.listdir(os.path.join(train_dir, "a")))

    # A fresh root with identical source images, same seed -> identical split.
    root2 = tmp_path.parent / (tmp_path.name + "_2")
    _make_imagefolder(root2, {"a": 20})
    train_dir2, _, _ = split_dataset(str(root2), seed=42)
    second = sorted(os.listdir(os.path.join(train_dir2, "a")))
    assert first == second


def test_split_dataset_reuses_existing_train_val_test_dirs(tmp_path):
    for split_name in ("train", "val", "test"):
        d = tmp_path / split_name / "a"
        d.mkdir(parents=True)
        (d / "sentinel.png").write_bytes(b"not touched")

    train_dir, val_dir, test_dir = split_dataset(str(tmp_path), seed=0)
    assert os.path.exists(os.path.join(train_dir, "a", "sentinel.png"))
    assert os.path.exists(os.path.join(val_dir, "a", "sentinel.png"))
    assert os.path.exists(os.path.join(test_dir, "a", "sentinel.png"))


def test_split_dataset_rejects_ratios_that_dont_sum_to_one(tmp_path):
    _make_imagefolder(tmp_path, {"a": 10})
    with pytest.raises(ValueError):
        split_dataset(str(tmp_path), train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
