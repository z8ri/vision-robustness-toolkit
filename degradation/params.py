"""degradation/params.py — single source of truth for severity parameters.

Design note (deviation from the frozen paper repo, see README "退化协议的口径改动"):
the paper repo keeps two separate parameter tables for the same degradation type —
one used at training time (`augmentation/physdeg.py`, 5 severity levels) and a
different one used at evaluation time (`eval.py`, 3 severity levels, independently
chosen numbers). This repo uses exactly one 3-severity table per degradation type,
shared by both training-time augmentation and evaluation-time corruption, so a
"severity 2" chain applied during training is physically the same perturbation as
"severity 2" reported in an evaluation table.

Seven in-distribution (ID) types — PhysDeg trains on these, and the ID evaluation
regime re-applies them:
    motion_blur, defocus_blur, gaussian_noise, jpeg, downsample, brightness, contrast

Eight out-of-distribution (OOD) types — never seen during training, evaluation-only:
    gaussian_blur, glass_blur, zoom_blur, gamma, low_light,
    speckle_noise, impulse_noise, pixelate

Severity indices are 0-based (0=light, 1=medium, 2=heavy); reported severities in
result dicts use the 1-based "s1/s2/s3" convention (matching the paper repo's
reporting convention, so it reads the same way in tables and plots).
"""
from typing import Dict, List, Tuple

N_SEVERITIES = 3

# ============================================================
# In-distribution (ID) severity tables
# ============================================================

ID_PARAMS: Dict[str, dict] = {
    "motion_blur": {"kernel_sizes": [10, 15, 20]},
    "defocus_blur": {"radii": [3, 6, 10]},
    "gaussian_noise": {"sigmas": [0.08, 0.18, 0.38]},          # on [0, 1] pixel scale
    "jpeg": {"qualities": [25, 15, 7]},
    "downsample": {"scales": [0.60, 0.40, 0.25]},
    # Photometric — additive illumination shift / mean-centered contrast gain.
    "brightness": {"offsets": [0.12, 0.24, 0.36]},             # fraction of 255, additive
    "contrast": {"factors": [0.60, 0.45, 0.30]},                # mean-centered gain, <1 = flatter
}

# Physical acquisition order used when PhysDeg composes a chain of 2-3 degradations:
# illumination/photometric -> blur -> noise -> compression -> resolution loss.
PHYSICAL_ORDER: List[str] = [
    "brightness", "contrast", "defocus_blur", "motion_blur",
    "gaussian_noise", "jpeg", "downsample",
]

ID_TYPES: Tuple[str, ...] = tuple(PHYSICAL_ORDER)
assert set(ID_TYPES) == set(ID_PARAMS), "PHYSICAL_ORDER and ID_PARAMS must cover the same 7 types"

# ============================================================
# Out-of-distribution (OOD) severity tables — held out from training entirely.
# ============================================================

OOD_PARAMS: Dict[str, dict] = {
    "gaussian_blur": {"radius": [1.0, 2.0, 3.0]},
    "glass_blur": {"sigma_delta": [(0.7, 1), (0.9, 2), (1.1, 3)]},
    "zoom_blur": {"zmax": [1.06, 1.11, 1.16]},
    "gamma": {"g": [1.5, 2.0, 2.5]},
    "low_light": {"gain": [0.6, 0.45, 0.3]},
    "speckle_noise": {"c": [0.15, 0.25, 0.35]},
    "impulse_noise": {"amount": [0.02, 0.04, 0.07]},
    "pixelate": {"scale": [0.6, 0.45, 0.3]},
}

OOD_TYPES: Tuple[str, ...] = tuple(OOD_PARAMS.keys())

OOD_FAMILIES: Dict[str, List[str]] = {
    "blur": ["gaussian_blur", "glass_blur", "zoom_blur"],
    "photometric": ["gamma", "low_light"],
    "noise": ["speckle_noise", "impulse_noise"],
    "digital": ["pixelate"],
}
assert sorted(sum(OOD_FAMILIES.values(), [])) == sorted(OOD_TYPES)
