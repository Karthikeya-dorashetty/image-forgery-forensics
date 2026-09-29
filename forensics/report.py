"""Plain-language HTML/Markdown report. No absolute 'this is fake' claims."""

from __future__ import annotations

from .pipeline import ForensicResult


def to_markdown(result: ForensicResult) -> str:
    info = result.info
    cm = result.copy_move
    lines = [
        "# Image Forensic Screening Report",
        "",
        "This report is a **preliminary** analysis. It does not prove that an image is forged.",
        "",
        "## Image information",
        f"- Filename: `{info.filename}`",
        f"- Format: {info.fmt}",
        f"- Original resolution: {info.original_shape[1]} x {info.original_shape[0]} x {info.channels}",
        f"- Working resolution: {info.processed_shape[1]} x {info.processed_shape[0]}",
        f"- Scale used for analysis: {info.scale:.3f}",
        "",
        "## Screening decision",
    ]
    if result.plan.stats:
        s = result.plan.stats
        lines += [
            f"- Texture (Laplacian variance): {s.texture:.1f}",
            f"- Edge density: {s.edge_density:.4f}",
            f"- FAST keypoints: {s.fast_keypoints}",
            f"- High-frequency energy: {s.high_freq_energy:.2f}",
            "",
        ]
    lines.append("Method plan:")
    for r in result.plan.reasons:
        lines.append(f"- {r}")
    if result.skipped:
        lines.append(f"- Skipped methods: {', '.join(result.skipped)}")

    lines += ["", "## Detection results"]
    if cm is None:
        lines.append("- Copy-move: not run")
    else:
        lines += [
            f"- Copy-move detector: {cm.method}",
            f"- Keypoints: {cm.n_keypoints}",
            f"- Ratio matches: {cm.n_ratio_matches} (raw knn pairs: {cm.n_raw_matches})",
            f"- Spatially separated matches: {cm.n_spatial_matches}",
            f"- Geometrically verified inliers: {cm.n_verified}",
            f"- Inlier ratio: {cm.inlier_ratio:.3f}",
            f"- Geometric verification: {'yes' if cm.verified else 'no'}",
        ]
        for n in cm.notes:
            lines.append(f"  - {n}")

    lines += [
        f"- Noise analysis: {'run' if result.noise else 'skipped'}",
        f"- Color analysis: {'run' if result.color else 'skipped'}",
        f"- Edge analysis: {'run' if result.edges else 'skipped'}",
        "",
        "## Fusion weights",
    ]
    for k, v in result.fusion.weights.items():
        lines.append(f"- {k}: {v:.3f}")
    for n in result.fusion.notes:
        lines.append(f"- {n}")

    lines += ["", "## Suspicious regions"]
    if not result.refine.regions:
        lines.append("- None localized after refinement.")
    for i, r in enumerate(result.refine.regions, 1):
        x, y, w, h = r.bbox
        lines += [
            f"### Region {i}",
            f"- Location: x={x}, y={y}, w={w}, h={h}",
            f"- Area (working pixels): {r.area}",
            f"- Methods: {', '.join(r.methods)}",
            f"- Supporting method count: {r.supporting_methods}",
            f"- Evidence strength (mean fused score): {r.strength:.3f}",
            f"- Geometrically verified copy-move: {'yes' if r.verified_geometry else 'no'}",
            f"- Evidence class: {r.note}",
        ]

    lines += [
        "",
        "## Interpretation",
        result.interpretation,
        "",
        "## Performance",
        f"- Total time: {result.total_time:.3f} s",
    ]
    for k, v in result.times.items():
        lines.append(f"- {k}: {v:.3f} s")

    lines += ["", "## Limitations"]
    for lim in result.limitations:
        lines.append(f"- {lim}")
    return "\n".join(lines) + "\n"
