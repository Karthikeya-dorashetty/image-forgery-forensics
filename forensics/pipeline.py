"""Orchestrates: preprocess → screen → selective forensics → fuse → localize."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .color import ColorResult, analyze_color
from .copy_move import CopyMoveResult, detect_copy_move
from .edges import EdgeResult, analyze_edges
from .fusion import FusionResult, fuse
from .noise import NoiseResult, analyze_noise
from .preprocess import ImageInfo, preprocess
from .refine import RefineResult, refine_mask
from .screening import MethodPlan, measure, plan_methods
from .utils import Timer, draw_regions, heatmap_bgr, overlay_mask


LIMITATIONS = [
    "This is a preliminary forensic screening tool, not a legal proof of forgery.",
    "A marked region does not automatically mean intentional manipulation.",
    "Natural repetition (windows, fabric, foliage) can create feature matches.",
    "Heavy JPEG compression, strong blur, and tiny edits can hide traces.",
    "Multiple successive edits and AI-generated images need other methods.",
    "Interpret results together with source, context, and other evidence.",
]


@dataclass
class ForensicResult:
    original: np.ndarray
    working: np.ndarray
    info: ImageInfo
    plan: MethodPlan
    times: dict[str, float]
    copy_move: CopyMoveResult | None
    noise: NoiseResult | None
    color: ColorResult | None
    edges: EdgeResult | None
    fusion: FusionResult
    refine: RefineResult
    heatmaps: dict[str, np.ndarray]
    highlighted: np.ndarray
    overlay: np.ndarray
    skipped: list[str] = field(default_factory=list)
    interpretation: str = ""
    total_time: float = 0.0
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS))


class ForensicPipeline:
    def __init__(self, max_side: int = 960, prefer_speed: bool = True) -> None:
        self.max_side = max_side
        self.prefer_speed = prefer_speed

    def run(self, image_bgr: np.ndarray, filename: str = "image") -> ForensicResult:
        timer = Timer()
        original, working, info = timer.run("preprocess", preprocess, image_bgr, filename, self.max_side)
        stats = timer.run("screening", measure, working)
        plan = plan_methods(stats, prefer_speed=self.prefer_speed)

        skipped: list[str] = []
        copy_move = None
        noise = None
        color = None
        edges = None
        h, w = working.shape[:2]
        zero = np.zeros((h, w), dtype=np.float32)
        maps: dict[str, np.ndarray] = {}

        if plan.run_copy_move:
            copy_move = timer.run(
                "copy_move",
                detect_copy_move,
                working,
                plan.use_sift,
            )
            maps["copy_move"] = copy_move.score_map
        else:
            skipped.append("copy_move")
            timer.records["copy_move"] = 0.0

        # If copy-move is weak, pull in supporting maps even if screening dropped them.
        verified = bool(copy_move and copy_move.verified)
        if copy_move is not None and not verified:
            if not plan.run_noise:
                plan.run_noise = True
                plan.reasons.append("Copy-move not verified: enabling noise as supporting evidence.")
            if not plan.run_color and stats.is_color:
                plan.run_color = True
                plan.reasons.append("Copy-move not verified: enabling color as supporting evidence.")
            if not plan.run_edge:
                plan.run_edge = True
                plan.reasons.append("Copy-move not verified: enabling edge as supporting evidence.")

        if plan.run_noise:
            noise = timer.run("noise", analyze_noise, working)
            maps["noise"] = noise.score_map
        else:
            skipped.append("noise")
            timer.records["noise"] = 0.0

        if plan.run_color:
            color = timer.run("color", analyze_color, working)
            maps["color"] = color.score_map
        else:
            skipped.append("color")
            timer.records["color"] = 0.0

        if plan.run_edge:
            edges = timer.run("edges", analyze_edges, working)
            maps["edge"] = edges.score_map
        else:
            skipped.append("edge")
            timer.records["edges"] = 0.0

        if not maps:
            maps["edge"] = zero
            skipped.append("all_primary")

        inliers = 0 if copy_move is None else copy_move.n_verified
        fusion = timer.run("fusion", fuse, maps, verified, inliers)
        cm_mask = None if copy_move is None else copy_move.mask
        refine = timer.run(
            "refine",
            refine_mask,
            fusion.fused,
            fusion.corroboration,
            maps,
            cm_mask,
            verified,
        )

        heatmaps = {
            "copy_move": heatmap_bgr(maps.get("copy_move", zero)),
            "noise": heatmap_bgr(maps.get("noise", zero)),
            "color": heatmap_bgr(maps.get("color", zero)),
            "edge": heatmap_bgr(maps.get("edge", zero)),
            "fusion": heatmap_bgr(fusion.fused),
        }
        overlay = overlay_mask(working, refine.mask)
        highlighted = draw_regions(overlay, refine.boxes, refine.contours)

        interpretation = _interpret(copy_move, refine, skipped)
        total = sum(timer.records.values())
        return ForensicResult(
            original=original,
            working=working,
            info=info,
            plan=plan,
            times=timer.records,
            copy_move=copy_move,
            noise=noise,
            color=color,
            edges=edges,
            fusion=fusion,
            refine=refine,
            heatmaps=heatmaps,
            highlighted=highlighted,
            overlay=overlay,
            skipped=skipped,
            interpretation=interpretation,
            total_time=total,
        )


def _interpret(copy_move, refine, skipped) -> str:
    if not refine.regions:
        return (
            "The analysis did not find a geometrically consistent copy-move pair or a "
            "stable multi-method region. This does not prove the photograph is authentic."
        )
    n = len(refine.regions)
    geo = any(r.verified_geometry for r in refine.regions)
    multi = sum(1 for r in refine.regions if r.supporting_methods >= 2)
    if geo:
        return (
            f"The analysis identified {n} suspicious region(s). At least one is supported by "
            "geometrically verified copy-move matches (source/target consistent under a rigid/homography model). "
            "This is screening evidence of possible copy-move, not a legal conclusion."
        )
    if multi:
        return (
            f"The analysis identified {n} region(s) where more than one forensic signal overlaps. "
            "Treat this as multi-method suspicion that still needs human review."
        )
    skip = ", ".join(skipped) if skipped else "none"
    return (
        f"The analysis marked {n} region(s) from single-method evidence only (skipped: {skip}). "
        "Single-method heatmaps are weak indicators and often include natural image variation."
    )
