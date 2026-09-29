"""Copy-move detection: features, ratio test, spatial filter, geometric verification."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .utils import to_gray


@dataclass
class CopyMoveResult:
    method: str
    n_keypoints: int
    n_raw_matches: int
    n_ratio_matches: int
    n_spatial_matches: int
    n_verified: int
    inlier_ratio: float
    verified: bool
    score_map: np.ndarray
    mask: np.ndarray
    src_pts: np.ndarray
    dst_pts: np.ndarray
    verified_src: np.ndarray
    verified_dst: np.ndarray
    match_image: np.ndarray
    notes: list[str] = field(default_factory=list)


def _detector(use_sift: bool, n_features: int):
    if use_sift:
        try:
            return cv2.SIFT_create(nfeatures=n_features), "SIFT"
        except Exception:
            pass
    return cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2, nlevels=8), "ORB"


def _knn_match(desc, method: str):
    if desc is None or len(desc) < 8:
        return []
    if method == "SIFT":
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # k=3 because the nearest neighbor of a descriptor in the same set is itself.
    try:
        return matcher.knnMatch(desc, desc, k=3)
    except cv2.error:
        return []


def detect_copy_move(
    image_bgr: np.ndarray,
    use_sift: bool = True,
    n_features: int = 1800,
    ratio: float = 0.72,
    min_inliers: int = 8,
) -> CopyMoveResult:
    gray = to_gray(image_bgr)
    h, w = gray.shape
    det, method = _detector(use_sift, n_features)
    kps, desc = det.detectAndCompute(gray, None)
    n_kps = 0 if kps is None else len(kps)
    empty = np.zeros((h, w), dtype=np.float32)
    blank = image_bgr.copy()
    notes: list[str] = []

    def _pack(
        n_raw=0,
        n_ratio=0,
        n_spatial=0,
        n_verified=0,
        inlier_ratio=0.0,
        verified=False,
        score=None,
        mask=None,
        src=None,
        dst=None,
        vsrc=None,
        vdst=None,
        vis=None,
    ) -> CopyMoveResult:
        return CopyMoveResult(
            method=method,
            n_keypoints=n_kps,
            n_raw_matches=n_raw,
            n_ratio_matches=n_ratio,
            n_spatial_matches=n_spatial,
            n_verified=n_verified,
            inlier_ratio=inlier_ratio,
            verified=verified,
            score_map=empty if score is None else score,
            mask=np.zeros((h, w), dtype=np.uint8) if mask is None else mask,
            src_pts=np.zeros((0, 2), np.float32) if src is None else src,
            dst_pts=np.zeros((0, 2), np.float32) if dst is None else dst,
            verified_src=np.zeros((0, 2), np.float32) if vsrc is None else vsrc,
            verified_dst=np.zeros((0, 2), np.float32) if vdst is None else vdst,
            match_image=blank if vis is None else vis,
            notes=notes,
        )

    if n_kps < 12 or desc is None:
        notes.append("Not enough keypoints for copy-move matching.")
        return _pack()

    pairs = _knn_match(desc, method)
    n_raw = len(pairs)
    good = []
    for pair in pairs:
        if len(pair) < 3:
            continue
        _self, m, n = pair
        if m.queryIdx == m.trainIdx:
            continue
        if m.queryIdx >= m.trainIdx:
            continue  # unique unordered pairs
        if m.distance < ratio * n.distance:
            good.append(m)

    min_dist = 0.045 * float(np.hypot(w, h))
    spatial = []
    src_list = []
    dst_list = []
    for m in good:
        p1 = np.array(kps[m.queryIdx].pt, dtype=np.float32)
        p2 = np.array(kps[m.trainIdx].pt, dtype=np.float32)
        if np.linalg.norm(p1 - p2) < min_dist:
            continue
        spatial.append(m)
        src_list.append(p1)
        dst_list.append(p2)

    src = np.array(src_list, dtype=np.float32)
    dst = np.array(dst_list, dtype=np.float32)
    notes.append(
        f"{method}: {n_kps} keypoints, {len(good)} ratio matches, {len(spatial)} spatially separated."
    )

    vis = _draw_matches(image_bgr, src, dst, color=(180, 180, 180))
    if len(spatial) < min_inliers:
        notes.append("Too few spatially separated matches to attempt geometric verification.")
        score = _match_density(h, w, src, dst, weak=True)
        return _pack(n_raw, len(good), len(spatial), 0, 0.0, False, score, None, src, dst, None, None, vis)

    vsrc, vdst, model_name = _verify_geometry(src, dst, min_inliers)
    n_verified = 0 if vsrc is None else len(vsrc)
    inlier_ratio = n_verified / max(len(spatial), 1)
    verified = n_verified >= min_inliers and inlier_ratio >= 0.22

    score = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    if verified:
        notes.append(f"Geometric verification passed ({model_name}): {n_verified} inliers.")
        score = _match_density(h, w, vsrc, vdst, weak=False)
        mask = _hull_mask(h, w, vsrc, vdst)
        vis = _draw_matches(image_bgr, vsrc, vdst, color=(0, 255, 255))
    else:
        notes.append(
            "Matches failed geometric consistency (repeated texture / random matches). "
            "They are not treated as copy-move evidence."
        )
        score = _match_density(h, w, src, dst, weak=True) * 0.15
        vsrc = np.zeros((0, 2), np.float32)
        vdst = np.zeros((0, 2), np.float32)

    return _pack(
        n_raw,
        len(good),
        len(spatial),
        n_verified,
        inlier_ratio,
        verified,
        score,
        mask,
        src,
        dst,
        vsrc,
        vdst,
        vis,
    )


def _cluster_displacements(src: np.ndarray, dst: np.ndarray, bin_size: float = 12.0):
    disp = dst - src
    keys = np.round(disp / bin_size).astype(np.int32)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(keys):
        k = (int(key[0]), int(key[1]))
        buckets.setdefault(k, []).append(i)
    return sorted(buckets.values(), key=len, reverse=True)


def _verify_geometry(src: np.ndarray, dst: np.ndarray, min_inliers: int):
    clusters = _cluster_displacements(src, dst)
    best_src = None
    best_dst = None
    best_n = 0
    best_name = "none"

    for idxs in clusters[:6]:
        if len(idxs) < min_inliers:
            continue
        s = src[idxs]
        d = dst[idxs]
        in_s, in_d, name = _ransac_pair(s, d, min_inliers)
        n = 0 if in_s is None else len(in_s)
        if n > best_n:
            best_n, best_src, best_dst, best_name = n, in_s, in_d, name

    # Global fallback if clustering split a valid set
    if best_n < min_inliers:
        in_s, in_d, name = _ransac_pair(src, dst, min_inliers)
        return in_s, in_d, name
    return best_src, best_dst, best_name


def _ransac_pair(src: np.ndarray, dst: np.ndarray, min_inliers: int):
    if len(src) < min_inliers:
        return None, None, "none"
    s = src.reshape(-1, 1, 2)
    d = dst.reshape(-1, 1, 2)

    affine, inliers = cv2.estimateAffinePartial2D(
        s, d, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.995
    )
    if affine is not None and inliers is not None and int(inliers.sum()) >= min_inliers:
        mask = inliers.ravel().astype(bool)
        return src[mask], dst[mask], "affine-partial+RANSAC"

    H, inliers = cv2.findHomography(s, d, cv2.RANSAC, 4.0, maxIters=2000, confidence=0.995)
    if H is not None and inliers is not None and int(inliers.sum()) >= min_inliers:
        mask = inliers.ravel().astype(bool)
        return src[mask], dst[mask], "homography+RANSAC"
    return None, None, "none"


def _match_density(h, w, src, dst, weak: bool) -> np.ndarray:
    acc = np.zeros((h, w), dtype=np.float32)
    if src is None or len(src) == 0:
        return acc
    pts = np.vstack([src, dst])
    radius = 18 if not weak else 12
    strength = 1.0 if not weak else 0.25
    for x, y in pts:
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < w and 0 <= iy < h:
            cv2.circle(acc, (ix, iy), radius, strength, -1)
    acc = cv2.GaussianBlur(acc, (0, 0), 6)
    m = float(acc.max())
    if m > 0:
        acc /= m
    return acc


def _hull_mask(h, w, src, dst) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in (src, dst):
        if pts is None or len(pts) < 3:
            continue
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2).astype(np.int32)
        cv2.fillConvexPoly(mask, hull, 255)
    if np.any(mask):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def _draw_matches(image_bgr, src, dst, color):
    vis = image_bgr.copy()
    if src is None:
        return vis
    n = min(len(src), 80)
    for i in range(n):
        p1 = tuple(np.round(src[i]).astype(int))
        p2 = tuple(np.round(dst[i]).astype(int))
        cv2.circle(vis, p1, 3, color, -1)
        cv2.circle(vis, p2, 3, color, -1)
        cv2.line(vis, p1, p2, color, 1)
    return vis
