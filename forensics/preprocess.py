"""Stage 1: load, validate, and optionally downscale. No denoising."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import ensure_bgr, resize_max_side


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ImageInfo:
    filename: str
    fmt: str
    original_shape: tuple[int, int, int]
    processed_shape: tuple[int, int, int]
    channels: int
    scale: float
    notes: list[str]


def decode_bytes(data: bytes, filename: str = "upload") -> tuple[np.ndarray, str]:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("Could not decode the file. Use JPEG or PNG photographs.")
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    return image, ext or ".jpg"


def preprocess(image: np.ndarray, filename: str = "image", max_side: int = 960) -> tuple[np.ndarray, np.ndarray, ImageInfo]:
    notes: list[str] = []
    original = ensure_bgr(image)
    if original.dtype != np.uint8:
        original = np.clip(original, 0, 255).astype(np.uint8)
        notes.append("Converted pixel depth to 8-bit for analysis.")

    h, w = original.shape[:2]
    if h < 32 or w < 32:
        raise ValueError("Image is too small for forensic screening (minimum 32x32).")

    processed, scale = resize_max_side(original, max_side)
    if scale < 1.0:
        notes.append(
            f"Working copy resized from {w}x{h} to {processed.shape[1]}x{processed.shape[0]} "
            "for speed. The original is kept untouched."
        )

    fmt = "PNG" if filename.lower().endswith(".png") else "JPEG"
    if filename.lower().endswith((".tif", ".tiff")):
        fmt = "TIFF"
    elif filename.lower().endswith(".bmp"):
        fmt = "BMP"
    elif filename.lower().endswith(".webp"):
        fmt = "WEBP"

    info = ImageInfo(
        filename=filename,
        fmt=fmt,
        original_shape=(h, w, original.shape[2]),
        processed_shape=(processed.shape[0], processed.shape[1], processed.shape[2]),
        channels=int(original.shape[2]),
        scale=scale,
        notes=notes,
    )
    return original, processed, info
