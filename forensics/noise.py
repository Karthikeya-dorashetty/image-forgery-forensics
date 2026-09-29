"""Local noise residual analysis. Differences are not proof of tampering by themselves."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import normalize01, to_gray


@dataclass
class NoiseResult:
    score_map: np.ndarray
    notes: list[str]


def analyze_noise(image_bgr: np.ndarray, patch: int = 16) -> NoiseResult:
    gray = to_gray(image_bgr).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    residual = gray - blur
    residual = residual - cv2.blur(residual, (31, 31))

    sq = residual * residual
    local_var = cv2.blur(sq, (patch, patch))
    global_med = float(np.median(local_var) + 1e-6)
    rel = np.abs(local_var - global_med) / global_med
    score = normalize01(rel)
    # Suppress extremely flat skies/walls that always look "different"
    texture = cv2.Laplacian(gray, cv2.CV_32F)
    texture = cv2.blur(texture * texture, (15, 15))
    tex_mask = normalize01(texture)
    score = score * np.clip(tex_mask + 0.15, 0, 1)

    notes = [
        "Noise map compares local residual variance to the image median.",
        "Lighting, texture, and camera processing also change noise, so this is supporting evidence only.",
    ]
    return NoiseResult(score_map=score.astype(np.float32), notes=notes)
