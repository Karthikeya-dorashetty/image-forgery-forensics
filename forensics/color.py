"""Local color consistency in LAB. Illumination/shadows can look similar."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import normalize01


@dataclass
class ColorResult:
    score_map: np.ndarray
    notes: list[str]


def analyze_color(image_bgr: np.ndarray) -> ColorResult:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:3]
    mean_ab = cv2.blur(ab, (21, 21))
    diff = np.linalg.norm(ab - mean_ab, axis=2)
    # Compare a pixel neighborhood to a larger neighborhood (chroma jump)
    local = cv2.blur(ab, (7, 7))
    wide = cv2.blur(ab, (41, 41))
    jump = np.linalg.norm(local - wide, axis=2)
    score = normalize01(0.4 * diff + 0.6 * jump)

    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    chroma = np.linalg.norm(ycrcb[:, :, 1:3] - cv2.blur(ycrcb[:, :, 1:3], (25, 25)), axis=2)
    score = normalize01(0.7 * score + 0.3 * normalize01(chroma))

    notes = [
        "Color map uses LAB/YCrCb local chroma deviation.",
        "Shadows, sunset lighting, and white-balance shifts can raise this score without editing.",
    ]
    return ColorResult(score_map=score.astype(np.float32), notes=notes)
