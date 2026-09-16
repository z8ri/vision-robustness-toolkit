"""data/dataset.py — ImageFolder-style defect dataset.

Directory convention: `root_dir/<class_name>/<image_file>`, classes sorted
alphabetically into an integer label (no separate manifest file), grayscale
source images. This matches the directory/labeling convention the real
private industrial-defect dataset uses — this repo has never had that dataset
available to test against (see README), so this loader is written to that
documented convention rather than verified against real files. Point it at a
directory in this layout and it should work as-is; if the real data's layout
ever differs from this, this file is the one place that would need to change.
"""
import os
import random
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

IMAGE_EXTENSIONS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DefectDataset(Dataset):
    """One split's worth of samples: `root_dir/<class_name>/<image>`, loaded
    grayscale (`convert("L")`) and handed to `transform` (see
    `build_transforms`, which replicates to 3 channels for an
    ImageNet-pretrained backbone before any resize/crop/normalize)."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"root_dir does not exist: {root_dir}")
        self.root_dir = root_dir
        self.transform = transform
        self.classes: List[str] = sorted(
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        )
        if not self.classes:
            raise ValueError(f"no class subdirectories found under {root_dir!r}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: List[Tuple[str, int]] = []
        for c in self.classes:
            class_dir = os.path.join(root_dir, c)
            for fname in sorted(os.listdir(class_dir)):
                if fname.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[c]))
        if not self.samples:
            raise ValueError(f"no images (extensions {IMAGE_EXTENSIONS}) found under {root_dir!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(img_size: int = 224, train: bool = True) -> Callable:
    """train=True: RandomResizedCrop + RandomHorizontalFlip (fed to PhysDeg's
    base_transform, so clean/deg1/deg2 each get an independently sampled
    crop/flip — see augmentation/physdeg.py). train=False: a plain resize, for
    val/test/calibration, where the evaluation protocol's own degradation
    (or nothing, for clean) is the only source of variation."""
    stages = [T.Grayscale(num_output_channels=3)]
    if train:
        stages += [T.RandomResizedCrop(img_size, scale=(0.8, 1.0)), T.RandomHorizontalFlip()]
    else:
        stages += [T.Resize((img_size, img_size))]
    stages += [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(stages)


def split_dataset(root_dir: str, seed: int = 0, train_ratio: float = 0.70,
                   val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[str, str, str]:
    """Returns (train_dir, val_dir, test_dir) under `root_dir`.

    If `root_dir/{train,val,test}` already exist (each with class
    subdirectories), they're used as-is — the split, once created, doesn't
    get silently redone. Otherwise this performs a seeded, per-class
    stratified split and materializes it as symlinks (not copies, so this
    doesn't duplicate image bytes on disk) under `root_dir/{train,val,test}/
    <class_name>/`, reproducible from the same seed.
    """
    split_dirs = {s: os.path.join(root_dir, s) for s in ("train", "val", "test")}
    if all(os.path.isdir(p) and os.listdir(p) for p in split_dirs.values()):
        return split_dirs["train"], split_dirs["val"], split_dirs["test"]

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio}")

    classes = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d not in split_dirs
    )
    if not classes:
        raise ValueError(f"no class subdirectories found under {root_dir!r} to split")

    rng = random.Random(seed)
    for split_dir in split_dirs.values():
        os.makedirs(split_dir, exist_ok=True)

    for c in classes:
        class_dir = os.path.join(root_dir, c)
        images = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTENSIONS))
        rng.shuffle(images)
        n_train = int(round(len(images) * train_ratio))
        n_val = int(round(len(images) * val_ratio))
        parts = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:],
        }
        for split_name, files in parts.items():
            split_class_dir = os.path.join(split_dirs[split_name], c)
            os.makedirs(split_class_dir, exist_ok=True)
            for fname in files:
                src = os.path.abspath(os.path.join(class_dir, fname))
                dst = os.path.join(split_class_dir, fname)
                if not os.path.exists(dst):
                    os.symlink(src, dst)

    return split_dirs["train"], split_dirs["val"], split_dirs["test"]
