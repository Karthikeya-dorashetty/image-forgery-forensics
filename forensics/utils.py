"""Shared image helpers. No enhancement that would wipe forensic traces."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np


@dataclass
class Timed:
    name: str
    seconds: float = 0.0


class Timer:
    def __init__(self) -> None:
        self.records: dict[str, float] = {}

    def run(self, name: str, fn: Callable, *args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        self.records[name] = time.perf_counter() - t0
        return result


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    if image_bgr.ndim == 2:
        return image_bgr
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def normalize01(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-8:
        return np.zeros_like(a, dtype=np.float32)
    return (a - lo) / (hi - lo)


def resize_max_side(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image, 1.0
    scale = max_side / float(long_side)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def heatmap_bgr(score: np.ndarray) -> np.ndarray:
    u8 = np.clip(normalize01(score) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color=(0, 0, 255), alpha=0.35) -> np.ndarray:
    out = image_bgr.copy()
    binary = (mask > 0).astype(np.uint8)
    if not np.any(binary):
        return out
    tint = np.zeros_like(out)
    tint[binary > 0] = color
    cv2.addWeighted(tint, alpha, out, 1.0, 0, out)
    return out


def draw_regions(image_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]], contours=None) -> np.ndarray:
    out = image_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, -1, (0, 220, 255), 2)
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return out


@dataclass
class RegionEvidence:
    bbox: tuple[int, int, int, int]
    area: int
    methods: list[str]
    strength: float
    verified_geometry: bool
    supporting_methods: int
    note: str = ""
    extra: dict = field(default_factory=dict)
