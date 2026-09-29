"""Threshold, morphology, connected components, boxes. Keep dilation modest."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import RegionEvidence


@dataclass
class RefineResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    contours: list
    regions: list[RegionEvidence]


def refine_mask(
    fused: np.ndarray,
    corroboration: np.ndarray,
    method_maps: dict[str, np.ndarray],
    copy_move_mask: np.ndarray | None,
    copy_move_verified: bool,
    min_area_ratio: float = 0.0015,
) -> RefineResult:
    h, w = fused.shape[:2]
    area_img = h * w
    min_area = max(80, int(area_img * min_area_ratio))

    # Adaptive-ish threshold from score distribution
    flat = fused.reshape(-1)
    thr = max(0.42, float(np.quantile(flat, 0.92)))
    if copy_move_verified and copy_move_mask is not None and np.any(copy_move_mask):
        binary = ((fused >= 0.28) | (copy_move_mask > 0)).astype(np.uint8) * 255
    else:
        binary = (fused >= thr).astype(np.uint8) * 255

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    regions: list[RegionEvidence] = []

    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        # Drop extremely thin slivers that are usually edge noise
        if min(bw, bh) < 6:
            continue
        component = (labels == i).astype(np.uint8)
        mask[component > 0] = 255
        boxes.append((int(x), int(y), int(bw), int(bh)))

        methods = []
        for name, m in method_maps.items():
            if m is None:
                continue
            if float(m[component > 0].mean()) >= 0.22:
                methods.append(name)
        if copy_move_verified and copy_move_mask is not None:
            overlap = float(np.mean((copy_move_mask > 0) & (component > 0)))
            if overlap > 0 and "copy_move" not in methods:
                methods.append("copy_move")

        strength = float(fused[component > 0].mean())
        geo = bool(copy_move_verified and "copy_move" in methods)
        support = max(int(corroboration[component > 0].max()), len(methods))
        kind = "multi-method corroborated" if support >= 2 or geo else "single-method suspicion"
        regions.append(
            RegionEvidence(
                bbox=(int(x), int(y), int(bw), int(bh)),
                area=int(area),
                methods=methods or ["fusion"],
                strength=strength,
                verified_geometry=geo,
                supporting_methods=support,
                note=kind,
            )
        )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions.sort(key=lambda r: r.strength, reverse=True)
    return RefineResult(mask=mask, boxes=boxes, contours=contours, regions=regions)
