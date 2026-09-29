"""Unnatural edge-density / morphological-gradient cues around possible splice seams."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import normalize01, to_gray


@dataclass
class EdgeResult:
    score_map: np.ndarray
    notes: list[str]


def analyze_edges(image_bgr: np.ndarray) -> EdgeResult:
    gray = to_gray(image_bgr)
    edges = cv2.Canny(gray, 60, 140)
    density = cv2.blur(edges.astype(np.float32) / 255.0, (15, 15))
    density_wide = cv2.blur(edges.astype(np.float32) / 255.0, (45, 45))
    unusual = np.clip(density - density_wide, 0, None)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel).astype(np.float32)
    grad_n = normalize01(grad)
    # Paste boundaries often have a thin high-gradient rim without a matching object edge scale
    score = normalize01(0.55 * unusual + 0.45 * grad_n * density)

    notes = [
        "Edge map highlights local edge-density spikes and morphological gradients.",
        "Normal object outlines also produce edges; this is not used as a standalone verdict.",
    ]
    return EdgeResult(score_map=score.astype(np.float32), notes=notes)
