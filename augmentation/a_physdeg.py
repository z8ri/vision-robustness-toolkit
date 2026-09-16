"""augmentation/a_physdeg.py — A-PhysDeg: degradation-aware curriculum sampling.

Implements U8 from INTERVIEW_VALUE_UPGRADES.md:

  - "先轻后重地开放 severity"              -> SeverityCurriculum (curriculum.py)
  - "单退化诊断批次" 归因困难度            -> DiagnosticBatchRunner (below)
  - "退化族 x severity 维护困难度 EMA"     -> DifficultyEMA (difficulty.py)
  - "均匀采样与困难度采样的混合分布 + 概率上限" -> mixed_distribution / cap_and_renormalize
  - "困难度只作为停止梯度的采样统计"        -> difficulty values are plain floats,
                                              never touched by autograd
  - "绝不读取 test/OOD 反馈"               -> run_diagnostics requires split="train"

Everything else about PhysDeg is explicitly unchanged (data split, the 7-type
generator, the clean/deg1/deg2 triplet, CE(clean)+lambda*JSD): APhysDegTransform
subclasses PhysDegTransform and overrides only `_sample_chain`, exactly the
extension point component 1 was built to leave open — so a fair A vs. static
comparison only ever differs in this one method.
"""
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from degradation.params import ID_TYPES, N_SEVERITIES, PHYSICAL_ORDER

from .curriculum import SeverityCurriculum
from .difficulty import Cell, DifficultyEMA, mixed_distribution, weighted_sample_without_replacement
from .physdeg import Chain, PhysDegTransform

DifficultyFn = Callable[[torch.Tensor, torch.Tensor], float]  # (images, labels) -> difficulty in [0,1]
Sample = Tuple[object, int]  # (path-or-PIL-image, label) — mirrors eval/protocol.py's Sample


class DiagnosticBatchRunner:
    """For each (deg_type, severity) cell, draws a fresh *single-degradation*
    batch from the training split and asks the caller-supplied `difficulty_fn`
    how hard it currently is. Single-degradation (not a 2-3 type chain) so the
    resulting difficulty can be attributed to one cell — the design doc's fix for
    "混合退化的总错误无法准确归因给单个算子"."""

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
        deg_types: Sequence[str] = ID_TYPES,
        n_severities: int = N_SEVERITIES,
        batch_size: int = 16,
        rng: Optional[random.Random] = None,
    ):
        if not samples:
            raise ValueError("DiagnosticBatchRunner needs a non-empty training-split sample list")
        self.samples = samples
        self.transform = transform
        self.deg_types = tuple(deg_types)
        self.n_severities = n_severities
        self.batch_size = batch_size
        self.rng = rng or random.Random()

    def _draw_batch(self):
        n = min(self.batch_size, len(self.samples))
        return self.rng.sample(list(self.samples), n) if n < len(self.samples) else list(self.samples)

    def run(self, difficulty_fn: DifficultyFn, split: str) -> Dict[Cell, float]:
        if split != "train":
            raise ValueError(
                f"DiagnosticBatchRunner.run(split={split!r}) refused: difficulty EMA must only ever "
                "see the training split ('绝不读取 test/OOD 反馈'). Pass split='train' explicitly."
            )
        # Local import to keep this module importable without eval/protocol's torch DataLoader
        # dependency chain when only curriculum/difficulty logic is being unit-tested.
        from degradation.ops import apply_id_degradation

        results: Dict[Cell, float] = {}
        for deg_type in self.deg_types:
            for severity in range(self.n_severities):
                batch = self._draw_batch()
                images, labels = [], []
                for idx, (item, label) in enumerate(batch):
                    img = item.convert("L") if isinstance(item, Image.Image) else Image.open(item).convert("L")
                    img = apply_id_degradation(img, deg_type, severity, seed=self.rng.randrange(2 ** 31))
                    images.append(self.transform(img))
                    labels.append(label)
                image_batch = torch.stack(images, dim=0)
                label_batch = torch.tensor(labels, dtype=torch.long)
                results[(deg_type, severity)] = float(difficulty_fn(image_batch, label_batch))
        return results


def weighted_sample_chain(rng: random.Random, cell_dist: Dict[Cell, float],
                           chain_range: Tuple[int, int], deg_types: Sequence[str]) -> Chain:
    """Two-stage draw from a 21-cell (type, severity) distribution:
    1) marginalize over severity -> per-type weight; sample `n` types without replacement.
    2) for each chosen type, sample its severity from the conditional distribution
       (restricted to whichever severities are present in `cell_dist`, i.e. unlocked).
    Returns the chain sorted into physical acquisition order, matching static PhysDeg.
    """
    type_weight: Dict[str, float] = {t: 0.0 for t in deg_types}
    by_type: Dict[str, List[Tuple[int, float]]] = {t: [] for t in deg_types}
    for (t, s), p in cell_dist.items():
        type_weight[t] += p
        by_type[t].append((s, p))

    n = min(rng.randint(*chain_range), sum(1 for t in deg_types if by_type[t]))
    pool_types = [t for t in deg_types if by_type[t]]
    pool_weights = [type_weight[t] for t in pool_types]
    selected_types = weighted_sample_without_replacement(rng, pool_types, pool_weights, n)
    selected_types.sort(key=lambda t: PHYSICAL_ORDER.index(t))

    chain: Chain = []
    for t in selected_types:
        severities, probs = zip(*by_type[t])
        total = sum(probs)
        if total <= 0:
            s_chosen = rng.choice(severities)
        else:
            r = rng.random() * total
            upto = 0.0
            s_chosen = severities[-1]
            for s, p in zip(severities, probs):
                upto += p
                if upto >= r:
                    s_chosen = s
                    break
        chain.append((t, s_chosen))
    return chain


class AdaptivePhysDegController:
    """Owns the curriculum + difficulty state shared by every APhysDegTransform
    call; the training loop advances it via `set_progress` (once per epoch, say)
    and `run_diagnostics` (periodically, e.g. once per epoch on a fresh
    diagnostic batch)."""

    def __init__(
        self,
        deg_types: Sequence[str] = ID_TYPES,
        n_severities: int = N_SEVERITIES,
        unlock_fractions: Sequence[float] = (0.0, 1 / 3, 2 / 3),
        ema_decay: float = 0.9,
        mix_alpha: float = 0.5,
        prob_cap: float = 0.15,
        init_difficulty: float = 0.5,
    ):
        self.deg_types = tuple(deg_types)
        self.n_severities = n_severities
        self.curriculum = SeverityCurriculum(n_severities, unlock_fractions)
        self.difficulty = DifficultyEMA(self.deg_types, n_severities, decay=ema_decay, init_value=init_difficulty)
        self.mix_alpha = mix_alpha
        self.prob_cap = prob_cap
        self.progress = 0.0

    def set_progress(self, progress: float) -> None:
        self.progress = max(0.0, min(1.0, progress))

    def run_diagnostics(
        self,
        samples: Sequence[Sample],
        transform: Callable[[Image.Image], torch.Tensor],
        difficulty_fn: DifficultyFn,
        batch_size: int = 16,
        rng: Optional[random.Random] = None,
        split: str = "train",
    ) -> Dict[Cell, float]:
        runner = DiagnosticBatchRunner(samples, transform, self.deg_types, self.n_severities, batch_size, rng)
        raw = runner.run(difficulty_fn, split=split)
        for (t, s), value in raw.items():
            self.difficulty.update(t, s, value)
        return raw

    def cell_distribution(self) -> Dict[Cell, float]:
        unlocked = set(self.curriculum.unlocked_severities(self.progress))
        cells = [(t, s) for t in self.deg_types for s in range(self.n_severities) if s in unlocked]
        return mixed_distribution(cells, self.difficulty, self.mix_alpha, self.prob_cap)

    def sample_chain(self, rng: random.Random, chain_range: Tuple[int, int]) -> Chain:
        return weighted_sample_chain(rng, self.cell_distribution(), chain_range, self.deg_types)


class APhysDegTransform(PhysDegTransform):
    """Drop-in replacement for PhysDegTransform: identical (clean, deg1, deg2)
    triplet construction, only `_sample_chain` differs."""

    def __init__(
        self,
        base_transform: Callable[[Image.Image], torch.Tensor],
        controller: AdaptivePhysDegController,
        chain_range: Tuple[int, int] = (2, 3),
        rng: Optional[random.Random] = None,
    ):
        super().__init__(base_transform, deg_types=controller.deg_types, chain_range=chain_range, rng=rng)
        self.controller = controller

    def _sample_chain(self) -> Chain:
        return self.controller.sample_chain(self.rng, self.chain_range)
