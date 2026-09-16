"""eval/protocol.py — the 7-ID / 8-OOD corruption-robustness evaluation protocol.

Deliberately model-agnostic: this module only wires up "which images get which
degradation at which severity" and aggregates whatever macro-F1 numbers an
external `evaluate_fn` reports. It does not import a model, a training loop, or a
specific framework's Dataset base beyond torch's, so it can be exercised with a
synthetic in-memory sample list and a stub `evaluate_fn` (see tests/) without a
trained checkpoint or a real dataset — the two things this project doesn't have on
this machine.

Result shape mirrors the paper repo's eval.py/eval_ood.py convention on purpose
(same key names: `s1/s2/s3/mean`, `mPC`) so a reader already familiar with that
convention can follow this one immediately; the code underneath is independent.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from degradation.ops import apply_id_degradation, apply_ood_degradation
from degradation.params import ID_TYPES, N_SEVERITIES, OOD_FAMILIES, OOD_TYPES

Sample = Tuple[Union[str, Image.Image], int]  # (path-or-PIL-image, label)
EvaluateFn = Callable[[DataLoader], float]      # loader -> macro_f1


def load_gray(item: Union[str, Image.Image]) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("L")
    return Image.open(item).convert("L")


class CorruptionDataset(Dataset):
    """Wraps a sample list; applies one (deg_type, severity) corruption, then `transform`.

    `seed_base` lets the same underlying image produce different-but-reproducible
    noise realizations across (deg_type, severity) cells while staying deterministic
    across repeated evaluation runs of the same cell (seed = seed_base + sample idx).
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], Any],
        deg_type: str,
        severity: int,
        regime: str,
        seed_base: int = 0,
    ):
        if regime not in ("id", "ood"):
            raise ValueError(f"regime must be 'id' or 'ood', got {regime!r}")
        self.samples = samples
        self.transform = transform
        self.deg_type = deg_type
        self.severity = severity
        self.regime = regime
        self.seed_base = seed_base

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item, label = self.samples[idx]
        img = load_gray(item)
        apply_fn = apply_id_degradation if self.regime == "id" else apply_ood_degradation
        img = apply_fn(img, self.deg_type, self.severity, seed=self.seed_base + idx)
        return self.transform(img), label


@dataclass
class ProtocolResult:
    per_type: Dict[str, Dict[str, float]]   # {deg_type: {"s1":.., "s2":.., "s3":.., "mean":..}}
    mPC: float
    family_means: Dict[str, float]          # {} for the ID regime (no family grouping defined)


def _run(
    samples: Sequence[Sample],
    transform: Callable[[Image.Image], Any],
    evaluate_fn: EvaluateFn,
    regime: str,
    deg_types: Sequence[str],
    families: Dict[str, List[str]],
    batch_size: int,
    workers: int,
    loader_kwargs: dict,
) -> ProtocolResult:
    per_type: Dict[str, Dict[str, float]] = {}
    all_f1: List[float] = []
    for deg_type in deg_types:
        s_results: Dict[str, float] = {}
        for s in range(N_SEVERITIES):
            ds = CorruptionDataset(samples, transform, deg_type, s, regime)
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                 num_workers=workers, **loader_kwargs)
            f1 = float(evaluate_fn(loader))
            s_results[f"s{s + 1}"] = round(f1, 4)
            all_f1.append(f1)
        s_results["mean"] = round(sum(s_results[f"s{i + 1}"] for i in range(N_SEVERITIES)) / N_SEVERITIES, 4)
        per_type[deg_type] = s_results

    mPC = round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0
    family_means = {
        name: round(sum(per_type[t]["mean"] for t in members) / len(members), 4)
        for name, members in families.items()
    }
    return ProtocolResult(per_type=per_type, mPC=mPC, family_means=family_means)


def run_id_protocol(
    samples: Sequence[Sample],
    transform: Callable[[Image.Image], Any],
    evaluate_fn: EvaluateFn,
    batch_size: int = 32,
    workers: int = 0,
    **loader_kwargs: Any,
) -> ProtocolResult:
    """7 in-distribution types x 3 severities -> mPC (the ID robustness number)."""
    return _run(samples, transform, evaluate_fn, "id", ID_TYPES, {}, batch_size, workers, loader_kwargs)


def run_ood_protocol(
    samples: Sequence[Sample],
    transform: Callable[[Image.Image], Any],
    evaluate_fn: EvaluateFn,
    batch_size: int = 32,
    workers: int = 0,
    **loader_kwargs: Any,
) -> ProtocolResult:
    """8 held-out types x 3 severities -> mPC_OOD, plus per-family means."""
    return _run(samples, transform, evaluate_fn, "ood", OOD_TYPES, OOD_FAMILIES, batch_size, workers, loader_kwargs)
