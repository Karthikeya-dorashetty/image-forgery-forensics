"""Evidence fusion: weighted agreement, not a naive average of all maps."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import normalize01


@dataclass
class FusionResult:
    fused: np.ndarray
    corroboration: np.ndarray
    weights: dict[str, float]
    notes: list[str]


def fuse(
    maps: dict[str, np.ndarray],
    copy_move_verified: bool,
    copy_move_inliers: int,
) -> FusionResult:
    """
    Copy-move (when geometrically verified) dominates.
    Other maps only add mass where they overlap or when copy-move is weak/absent.
    """
    present = {k: v.astype(np.float32) for k, v in maps.items() if v is not None}
    h, w = next(iter(present.values())).shape[:2]
    fused = np.zeros((h, w), dtype=np.float32)
    corroboration = np.zeros((h, w), dtype=np.float32)
    notes: list[str] = []

    base_weights = {
        "copy_move": 0.62 if copy_move_verified else 0.12,
        "noise": 0.14,
        "color": 0.10,
        "edge": 0.16,
    }
    if copy_move_verified and copy_move_inliers >= 20:
        base_weights["copy_move"] = 0.72
        base_weights["noise"] = 0.08
        base_weights["color"] = 0.05
        base_weights["edge"] = 0.15
        notes.append("Strong verified copy-move: supporting maps are down-weighted.")
    elif "copy_move" not in present:
        s = base_weights["noise"] + base_weights["color"] + base_weights["edge"]
        for k in ("noise", "color", "edge"):
            base_weights[k] /= s
        notes.append("Copy-move unavailable; fusion uses residual/color/edge only.")

    used = {k: base_weights[k] for k in present}
    total = sum(used.values()) or 1.0
    used = {k: v / total for k, v in used.items()}

    binary_thr = 0.35
    for name, m in present.items():
        n = normalize01(m)
        fused += used[name] * n
        corroboration += (n > binary_thr).astype(np.float32)

    # Agreement bonus: independent methods overlapping in space
    if corroboration.max() >= 2:
        bonus = np.clip((corroboration - 1.0) / 3.0, 0, 1) * 0.18
        fused = np.clip(fused + bonus, 0, 1)
        notes.append("Multi-method overlap received an agreement bonus.")
    else:
        notes.append("Most suspicion is single-method; treat it as weak evidence.")

    fused = cv2.GaussianBlur(fused, (0, 0), 2.2)
    fused = normalize01(fused)
    notes.append("Fusion distinguishes geometrically verified copy-move from uncorroborated heatmaps.")
    return FusionResult(fused=fused, corroboration=corroboration, weights=used, notes=notes)
