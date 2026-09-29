"""Stage 2: cheap measurements that decide which expensive methods to run."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import to_gray


@dataclass
class ScreenStats:
    width: int
    height: int
    texture: float
    edge_density: float
    local_variation: float
    fast_keypoints: int
    high_freq_energy: float
    is_color: bool


@dataclass
class MethodPlan:
    run_copy_move: bool
    use_sift: bool
    run_noise: bool
    run_color: bool
    run_edge: bool
    reasons: list[str] = field(default_factory=list)
    stats: ScreenStats | None = None


def measure(image_bgr: np.ndarray) -> ScreenStats:
    gray = to_gray(image_bgr)
    h, w = gray.shape
    texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(np.mean(edges > 0))
    local_variation = float(cv2.blur(gray.astype(np.float32), (9, 9)).std())
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
    high_freq_energy = float(np.mean(residual))
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kps = fast.detect(gray, None)
    return ScreenStats(
        width=w,
        height=h,
        texture=texture,
        edge_density=edge_density,
        local_variation=local_variation,
        fast_keypoints=len(kps),
        high_freq_energy=high_freq_energy,
        is_color=image_bgr.ndim == 3 and image_bgr.shape[2] == 3,
    )


def plan_methods(stats: ScreenStats, prefer_speed: bool = True) -> MethodPlan:
    """Copy-move first. Supporting maps only when they are likely useful."""
    reasons: list[str] = []
    area = stats.width * stats.height
    enough_features = stats.fast_keypoints >= 60 and stats.texture >= 15.0
    textured = stats.texture >= 40.0 and stats.edge_density >= 0.02
    compressed_like = stats.high_freq_energy < 4.5

    run_copy_move = enough_features and area >= 80 * 80
    if not run_copy_move:
        reasons.append(
            "Copy-move matching skipped or deprioritized: too few keypoints/texture for stable matches."
        )
    else:
        reasons.append(
            f"Copy-move enabled: {stats.fast_keypoints} FAST corners, texture={stats.texture:.1f}."
        )

    use_sift = run_copy_move and textured and stats.fast_keypoints >= 120 and not (
        prefer_speed and area > 900 * 900
    )
    if run_copy_move:
        if use_sift:
            reasons.append("SIFT selected: enough texture for distinctive descriptors.")
        else:
            reasons.append("ORB selected: faster matching for this texture/resolution.")

    # Edges are cheap and help localization of paste boundaries.
    run_edge = stats.edge_density >= 0.008
    if run_edge:
        reasons.append("Edge analysis enabled (cheap boundary check).")
    else:
        reasons.append("Edge analysis skipped: image is almost edge-free.")

    # Noise/color are supporting evidence for localized edits, not default for every photo.
    likely_localized_edit = (not run_copy_move) or compressed_like or stats.texture < 80
    run_noise = likely_localized_edit or not prefer_speed
    run_color = stats.is_color and (likely_localized_edit or not prefer_speed)

    if run_copy_move and textured and prefer_speed:
        # Strong copy-move case: keep edge, drop redundant color; keep a light noise pass
        # only if compression may have altered residuals.
        run_color = False
        run_noise = compressed_like
        reasons.append(
            "Primary copy-move case: color map skipped as likely redundant; "
            "noise map runs only if compression-like residuals are present."
        )
    else:
        if run_noise:
            reasons.append("Noise residual analysis enabled as supporting evidence.")
        if run_color:
            reasons.append("Color inconsistency analysis enabled as supporting evidence.")

    if not run_noise:
        reasons.append("Noise analysis skipped to save time.")
    if not run_color:
        reasons.append("Color analysis skipped (not expected to add independent evidence).")

    return MethodPlan(
        run_copy_move=run_copy_move,
        use_sift=use_sift,
        run_noise=run_noise,
        run_color=run_color,
        run_edge=run_edge,
        reasons=reasons,
        stats=stats,
    )
