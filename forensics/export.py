"""Save masks, heatmaps, overlay, and reports to a folder."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

from .pipeline import ForensicResult
from .report import to_markdown


def _encode_png(image_bgr: np.ndarray) -> bytes:
    if image_bgr.ndim == 2:
        ok, buf = cv2.imencode(".png", image_bgr)
    else:
        ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("Could not encode PNG.")
    return buf.tobytes()


def save_result(result: ForensicResult, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out / "01_original.png"), result.original)
    cv2.imwrite(str(out / "02_working.png"), result.working)
    if result.copy_move is not None:
        cv2.imwrite(str(out / "03_copy_move_matches.png"), result.copy_move.match_image)
    cv2.imwrite(str(out / "04_heatmap_copy_move.png"), result.heatmaps["copy_move"])
    cv2.imwrite(str(out / "05_heatmap_noise.png"), result.heatmaps["noise"])
    cv2.imwrite(str(out / "06_heatmap_color.png"), result.heatmaps["color"])
    cv2.imwrite(str(out / "07_heatmap_edge.png"), result.heatmaps["edge"])
    cv2.imwrite(str(out / "08_heatmap_fusion.png"), result.heatmaps["fusion"])
    cv2.imwrite(str(out / "09_mask.png"), result.refine.mask)
    cv2.imwrite(str(out / "10_overlay.png"), result.overlay)
    cv2.imwrite(str(out / "11_highlighted.png"), result.highlighted)

    md = to_markdown(result)
    (out / "report.md").write_text(md, encoding="utf-8")
    (out / "report.html").write_text(to_html(result), encoding="utf-8")
    return out


def to_html(result: ForensicResult) -> str:
    def img_tag(title: str, bgr: np.ndarray) -> str:
        b64 = base64.b64encode(_encode_png(bgr)).decode("ascii")
        return (
            f"<figure><figcaption>{title}</figcaption>"
            f'<img src="data:image/png;base64,{b64}" alt="{title}"/></figure>'
        )

    md = to_markdown(result).replace("&", "&amp;").replace("<", "&lt;")
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Forensic screening report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;background:#111;color:#eee}",
        "img{max-width:100%;height:auto;border-radius:8px} figure{margin:16px 0}",
        "pre{white-space:pre-wrap;background:#1b1b1b;padding:16px;border-radius:8px}</style></head><body>",
        "<h1>Image forensic screening report</h1>",
        "<p>Preliminary analysis only. This does not prove that an image is forged.</p>",
        img_tag("Original", result.original),
        img_tag("Copy-move matches", result.copy_move.match_image if result.copy_move is not None else result.working),
        img_tag("Fusion heatmap", result.heatmaps["fusion"]),
        img_tag("Final mask", result.refine.mask),
        img_tag("Highlighted regions", result.highlighted),
        "<h2>Text report</h2><pre>",
        md,
        "</pre></body></html>",
    ]
    return "\n".join(parts)
