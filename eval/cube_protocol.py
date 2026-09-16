"""eval/cube_protocol.py — wires RobustnessCube to the 7-ID / 8-OOD protocol.

Same model-agnostic contract as eval/protocol.py: `predict_fn` is caller-supplied
(a real model's inference loop, later; a stub in tests today) and this module
only owns "which images get which degradation, in what order, tagged with what
group id".
"""
from typing import Callable, List, Optional, Sequence, Tuple, Union

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from degradation.params import ID_TYPES, N_SEVERITIES, OOD_TYPES

from .cube import RobustnessCube
from .protocol import CorruptionDataset, Sample, load_gray
from .records import PredictionRecord

# (pred_labels, confidences, true_labels), aligned to the loader's (shuffle=False) iteration order.
PredictFn = Callable[[DataLoader], Tuple[List[int], List[float], List[int]]]
GroupIdFn = Callable[[Union[str, Image.Image]], str]


class _CleanDataset(Dataset):
    """The one condition CorruptionDataset can't express: no degradation at all."""

    def __init__(self, samples: Sequence[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item, label = self.samples[idx]
        return self.transform(load_gray(item)), label


def _build_records(sample_ids: List[str], group_ids: List[str],
                    preds: List[int], confidences: List[float], labels: List[int]) -> List[PredictionRecord]:
    lengths = {len(sample_ids), len(group_ids), len(preds), len(confidences), len(labels)}
    if len(lengths) != 1:
        raise ValueError(f"predict_fn output length mismatch with sample count: {lengths}")
    return [
        PredictionRecord(sid, gid, true, pred, conf)
        for sid, gid, true, pred, conf in zip(sample_ids, group_ids, labels, preds, confidences)
    ]


def run_cube_protocol(
    samples: Sequence[Sample],
    transform,
    predict_fn: PredictFn,
    regime: str,
    class_names: Sequence[str],
    group_id_fn: Optional[GroupIdFn] = None,
    min_samples_per_class: int = 5,
    batch_size: int = 32,
    workers: int = 0,
    **loader_kwargs,
) -> RobustnessCube:
    if regime not in ("id", "ood"):
        raise ValueError(f"regime must be 'id' or 'ood', got {regime!r}")
    deg_types = ID_TYPES if regime == "id" else OOD_TYPES

    sample_ids = [str(i) for i in range(len(samples))]
    if group_id_fn is None:
        group_ids = list(sample_ids)  # default: one group per sample (private/NEU-style datasets)
    else:
        group_ids = [group_id_fn(item) for item, _ in samples]

    cube = RobustnessCube(class_names, min_samples_per_class=min_samples_per_class)

    def _run(dataset) -> List[PredictionRecord]:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, **loader_kwargs)
        preds, confidences, labels = predict_fn(loader)
        return _build_records(sample_ids, group_ids, preds, confidences, labels)

    cube.add_condition(None, None, _run(_CleanDataset(samples, transform)))
    for deg_type in deg_types:
        for severity in range(N_SEVERITIES):
            cube.add_condition(deg_type, severity, _run(CorruptionDataset(samples, transform, deg_type, severity, regime)))

    return cube
