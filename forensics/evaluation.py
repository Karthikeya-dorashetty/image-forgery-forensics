"""Controlled synthetic edits + pixel metrics. Numbers come from these tests only."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import ForensicPipeline, ForensicResult


@dataclass
class MetricRow:
    name: str
    precision: float
    recall: float
    f1: float
    iou: float
    fpr: float
    detected: bool
    seconds: float
    extra: dict


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred > 0
    g = gt > 0
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, np.logical_not(g)).sum())
    fn = float(np.logical_and(np.logical_not(p), g).sum())
    tn = float(np.logical_and(np.logical_not(p), np.logical_not(g)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "fpr": fpr,
        "detected": iou >= 0.15 or (tp > 0 and recall >= 0.2),
    }


def make_textured_base(h: int = 480, w: int = 640, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :, 0] = np.uint8(40 + 80 * (y / h))
        img[y, :, 1] = np.uint8(60 + 40 * np.sin(y / 18.0))
        img[y, :, 2] = np.uint8(50 + 50 * np.cos(y / 22.0))
    noise = rng.integers(0, 55, size=(h, w, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    for i in range(18):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x2, y2 = int(rng.integers(0, w)), int(rng.integers(0, h))
        color = tuple(int(c) for c in rng.integers(20, 255, size=3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=2)
        cv2.circle(img, (x1, y1), int(rng.integers(8, 28)), color, -1)
    # Distinct patch with rich corners (good SIFT/ORB target)
    cv2.putText(img, "SRC", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 220, 255), 3)
    cv2.putText(img, "MARKER", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 80, 40), 3)
    for i in range(12):
        cv2.drawMarker(img, (50 + i * 18, 160), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 12, 2)
    return img


def copy_move_forge(
    image_bgr: np.ndarray,
    src_box=(36, 40, 150, 150),
    dst_xy=(360, 220),
) -> tuple[np.ndarray, np.ndarray]:
    img = image_bgr.copy()
    x, y, bw, bh = src_box
    h, w = img.shape[:2]
    dx, dy = dst_xy
    dx = min(dx, w - bw - 1)
    dy = min(dy, h - bh - 1)
    patch = img[y : y + bh, x : x + bw].copy()
    img[dy : dy + bh, dx : dx + bw] = patch
    gt = np.zeros((h, w), dtype=np.uint8)
    gt[y : y + bh, x : x + bw] = 255
    gt[dy : dy + bh, dx : dx + bw] = 255
    return img, gt


def degrade(image_bgr: np.ndarray, kind: str) -> np.ndarray:
    img = image_bgr.copy()
    if kind == "original":
        return img
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img
    if kind == "resize":
        h, w = img.shape[:2]
        small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if kind == "blur":
        return cv2.GaussianBlur(img, (7, 7), 1.6)
    if kind == "noise":
        rng = np.random.default_rng(3)
        n = rng.normal(0, 12, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
    if kind == "small_forgery":
        return img
    return img


def _row(name: str, result: ForensicResult, gt: np.ndarray, extra=None) -> MetricRow:
    pred = result.refine.mask
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = _metrics(pred, gt)
    return MetricRow(
        name=name,
        precision=m["precision"],
        recall=m["recall"],
        f1=m["f1"],
        iou=m["iou"],
        fpr=m["fpr"],
        detected=bool(m["detected"]),
        seconds=result.total_time,
        extra=extra or {},
    )


def run_robustness(image_bgr: np.ndarray | None = None, max_side: int = 640) -> list[MetricRow]:
    base = image_bgr if image_bgr is not None else make_textured_base()
    forged, gt = copy_move_forge(base)
    pipe = ForensicPipeline(max_side=max_side, prefer_speed=True)
    rows = []
    for kind in ("original", "jpeg", "resize", "blur", "noise"):
        img = degrade(forged, kind)
        g = gt
        if img.shape[:2] != gt.shape[:2]:
            g = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        result = pipe.run(img, filename=f"{kind}.png")
        rows.append(_row(kind, result, g, extra={"condition": kind}))
    # smaller paste
    small, gt_s = copy_move_forge(base, src_box=(40, 50, 56, 56), dst_xy=(400, 280))
    result = pipe.run(small, filename="small.png")
    rows.append(_row("small_forgery", result, gt_s, extra={"condition": "small"}))
    return rows


def run_ablation(image_bgr: np.ndarray | None = None, max_side: int = 640) -> list[MetricRow]:
    base = image_bgr if image_bgr is not None else make_textured_base()
    forged, gt = copy_move_forge(base)
    configs = [
        ("A_copy_move_only", True, False, False, False),
        ("B_copy_move_geometry", True, False, False, False),  # geometry is always on in detector
        ("C_copy_move_noise", True, True, False, False),
        ("D_copy_move_color", True, False, True, False),
        ("E_copy_move_edge", True, False, False, True),
        ("F_all_useful", True, True, True, True),
    ]
    rows = []
    for name, cm, nz, col, ed in configs:
        result = _forced_run(forged, max_side, cm, nz, col, ed)
        rows.append(_row(name, result, gt))
    return rows


def _forced_run(image_bgr, max_side, run_cm, run_nz, run_col, run_ed) -> ForensicResult:
    """Bypass screening to measure each component; geometry stays inside copy-move."""
    from .copy_move import detect_copy_move
    from .color import analyze_color
    from .edges import analyze_edges
    from .fusion import fuse
    from .noise import analyze_noise
    from .preprocess import preprocess
    from .refine import refine_mask
    from .screening import measure, plan_methods
    from .utils import Timer, draw_regions, heatmap_bgr, overlay_mask
    from .pipeline import ForensicResult, LIMITATIONS, _interpret

    timer = Timer()
    original, working, info = timer.run("preprocess", preprocess, image_bgr, "eval.png", max_side)
    stats = timer.run("screening", measure, working)
    plan = plan_methods(stats, prefer_speed=False)
    plan.run_copy_move, plan.run_noise, plan.run_color, plan.run_edge = run_cm, run_nz, run_col, run_ed
    plan.use_sift = True
    maps = {}
    copy_move = noise = color = edges = None
    skipped = []
    if run_cm:
        copy_move = timer.run("copy_move", detect_copy_move, working, True)
        maps["copy_move"] = copy_move.score_map
    else:
        skipped.append("copy_move")
        timer.records["copy_move"] = 0.0
    if run_nz:
        noise = timer.run("noise", analyze_noise, working)
        maps["noise"] = noise.score_map
    else:
        skipped.append("noise")
        timer.records["noise"] = 0.0
    if run_col:
        color = timer.run("color", analyze_color, working)
        maps["color"] = color.score_map
    else:
        skipped.append("color")
        timer.records["color"] = 0.0
    if run_ed:
        edges = timer.run("edges", analyze_edges, working)
        maps["edge"] = edges.score_map
    else:
        skipped.append("edge")
        timer.records["edges"] = 0.0
    verified = bool(copy_move and copy_move.verified)
    inliers = 0 if copy_move is None else copy_move.n_verified
    h, w = working.shape[:2]
    if not maps:
        maps["edge"] = np.zeros((h, w), np.float32)
    fusion = timer.run("fusion", fuse, maps, verified, inliers)
    cm_mask = None if copy_move is None else copy_move.mask
    refine = timer.run("refine", refine_mask, fusion.fused, fusion.corroboration, maps, cm_mask, verified)
    heatmaps = {k: heatmap_bgr(v) for k, v in maps.items()}
    heatmaps["fusion"] = heatmap_bgr(fusion.fused)
    overlay = overlay_mask(working, refine.mask)
    highlighted = draw_regions(overlay, refine.boxes, refine.contours)
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
        interpretation=_interpret(copy_move, refine, skipped),
        total_time=sum(timer.records.values()),
        limitations=list(LIMITATIONS),
    )


def authentic_false_positive(image_bgr: np.ndarray | None = None, max_side: int = 640) -> MetricRow:
    base = image_bgr if image_bgr is not None else make_textured_base()
    gt = np.zeros(base.shape[:2], dtype=np.uint8)
    result = ForensicPipeline(max_side=max_side, prefer_speed=True).run(base, "auth.png")
    return _row("authentic_image", result, gt)


def print_tables() -> None:
    def show(title: str, rows: list[MetricRow]) -> None:
        print(f"\n{title}")
        print(f"{'name':<24} {'P':>6} {'R':>6} {'F1':>6} {'IoU':>6} {'FPR':>8} {'det':>5} {'sec':>7}")
        for r in rows:
            print(
                f"{r.name:<24} {r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f} "
                f"{r.iou:6.3f} {r.fpr:8.4f} {str(r.detected):>5} {r.seconds:7.3f}"
            )

    show("Robustness", run_robustness())
    show("Ablation", run_ablation())
    show("Authentic image", [authentic_false_positive()])


if __name__ == "__main__":
    print_tables()
