"""Streamlit forensic screening app."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import streamlit as st

from forensics.evaluation import (
    authentic_false_positive,
    copy_move_forge,
    make_textured_base,
    run_ablation,
    run_robustness,
)
from forensics.pipeline import ForensicPipeline
from forensics.preprocess import decode_bytes
from forensics.report import to_markdown


st.set_page_config(
    page_title="Image Forensic Screening",
    page_icon="🔍",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; max-width: 1400px;}
      div[data-testid="stMetric"] {background: #1B232C; padding: 0.6rem 0.8rem; border-radius: 10px;}
      .hint {color: #9AA8B5; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def metric_table(rows) -> None:
    st.dataframe(
        [
            {
                "experiment": r.name,
                "precision": round(r.precision, 3),
                "recall": round(r.recall, 3),
                "F1": round(r.f1, 3),
                "IoU": round(r.iou, 3),
                "FPR": round(r.fpr, 4),
                "detected": r.detected,
                "seconds": round(r.seconds, 3),
            }
            for r in rows
        ],
        use_container_width=True,
        hide_index=True,
    )


def show_result(result) -> None:
    cm = result.copy_move
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total time", f"{result.total_time:.2f}s")
    c2.metric("Working size", f"{result.info.processed_shape[1]}×{result.info.processed_shape[0]}")
    c3.metric("Regions", str(len(result.refine.regions)))
    c4.metric("Verified inliers", "—" if cm is None else str(cm.n_verified))
    c5.metric("Keypoints", "—" if cm is None else str(cm.n_keypoints))

    st.info(result.interpretation)

    tabs = st.tabs(
        [
            "Overview",
            "Copy-Move",
            "Noise",
            "Color",
            "Edge",
            "Fusion",
            "Final result",
            "Evaluation",
        ]
    )

    with tabs[0]:
        a, b = st.columns(2)
        with a:
            st.caption("Original (untouched)")
            st.image(bgr_to_rgb(result.original), use_container_width=True)
        with b:
            st.caption("Highlighted working image")
            st.image(bgr_to_rgb(result.highlighted), use_container_width=True)
        st.subheader("Screening plan")
        for reason in result.plan.reasons:
            st.write("• " + reason)
        st.subheader("Stage times")
        st.json({k: round(v, 4) for k, v in result.times.items()})

    with tabs[1]:
        if cm is None:
            st.warning("Copy-move matching was skipped by the screening stage.")
        else:
            x, y = st.columns(2)
            x.image(bgr_to_rgb(cm.match_image), caption="Matches (gray = raw / yellow = verified)", use_container_width=True)
            y.image(bgr_to_rgb(result.heatmaps["copy_move"]), caption="Copy-move evidence map", use_container_width=True)
            st.write(
                f"{cm.method} | keypoints={cm.n_keypoints} | ratio matches={cm.n_ratio_matches} | "
                f"spatial={cm.n_spatial_matches} | verified={cm.n_verified} | "
                f"geometry={'passed' if cm.verified else 'failed'}"
            )
            for n in cm.notes:
                st.caption(n)

    with tabs[2]:
        if result.noise is None:
            st.info("Noise analysis was not required for this image.")
        else:
            st.image(bgr_to_rgb(result.heatmaps["noise"]), caption="Noise anomaly map", use_container_width=True)
            for n in result.noise.notes:
                st.caption(n)

    with tabs[3]:
        if result.color is None:
            st.info("Color analysis was not required for this image.")
        else:
            st.image(bgr_to_rgb(result.heatmaps["color"]), caption="Color inconsistency map", use_container_width=True)
            for n in result.color.notes:
                st.caption(n)

    with tabs[4]:
        if result.edges is None:
            st.info("Edge analysis was skipped.")
        else:
            st.image(bgr_to_rgb(result.heatmaps["edge"]), caption="Edge anomaly map", use_container_width=True)
            for n in result.edges.notes:
                st.caption(n)

    with tabs[5]:
        st.image(bgr_to_rgb(result.heatmaps["fusion"]), caption="Fused evidence heatmap", use_container_width=True)
        st.write("Weights used (renormalized over methods that actually ran):")
        st.json({k: round(v, 3) for k, v in result.fusion.weights.items()})
        for n in result.fusion.notes:
            st.caption(n)
        st.caption("Single-method heat is weak. Overlap of independent signals is stronger.")

    with tabs[6]:
        a, b, c = st.columns(3)
        a.image(bgr_to_rgb(result.refine.mask), caption="Refined tampering mask", use_container_width=True)
        b.image(bgr_to_rgb(result.overlay), caption="Overlay", use_container_width=True)
        c.image(bgr_to_rgb(result.highlighted), caption="Contours / boxes", use_container_width=True)
        st.subheader("Region evidence")
        if not result.refine.regions:
            st.write("No region survived morphological cleanup.")
        for i, r in enumerate(result.refine.regions, 1):
            x, y, w, h = r.bbox
            st.markdown(
                f"**Region {i}** — ({x}, {y}, {w}×{h}) · strength {r.strength:.3f} · "
                f"{r.note} · methods: {', '.join(r.methods)} · "
                f"geometry verified: {'yes' if r.verified_geometry else 'no'}"
            )
        with st.expander("Limitations (read before using these results)"):
            for lim in result.limitations:
                st.write("• " + lim)

    with tabs[7]:
        st.write(
            "Metrics are computed on **controlled synthetic copy-move** with a known ground-truth mask. "
            "They are not a claimed real-world accuracy rate."
        )
        source = st.radio(
            "Evaluation source",
            ["Built-in textured sample", "Use the current working image as the host"],
            horizontal=True,
        )
        host = None
        if source.endswith("host") and "result" in st.session_state:
            host = st.session_state.result.working
        if st.button("Run robustness + ablation experiments", type="primary"):
            with st.spinner("Running controlled forgeries (this is slower than a single screening pass)…"):
                robust = run_robustness(host, max_side=640)
                ablation = run_ablation(host, max_side=640)
                auth = authentic_false_positive(host, max_side=640)
            st.session_state.eval_robust = robust
            st.session_state.eval_ablation = ablation
            st.session_state.eval_auth = auth
        if "eval_robust" in st.session_state:
            st.subheader("Robustness (same copy-move, different degradations)")
            metric_table(st.session_state.eval_robust)
            st.subheader("Ablation")
            metric_table(st.session_state.eval_ablation)
            st.subheader("Authentic image (false-positive check)")
            metric_table([st.session_state.eval_auth])
            st.caption(
                "If F1/IoU drop under JPEG/blur, that is the measured robustness — not a claim of invariance. "
                "Experiment B keeps geometric verification inside copy-move; raw matching is never used as the mask."
            )


def main() -> None:
    st.title("Multi-stage image forensic screening")
    st.markdown(
        '<p class="hint">Copy-move first, geometric verification, then supporting maps only when they help. '
        "This is a preliminary screen — not a forensic verdict.</p>",
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Input")
        uploaded = st.file_uploader("Upload JPEG or PNG", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"])
        demo = st.button("Load synthetic copy-move demo")
        max_side = st.slider("Working max side (speed vs detail)", 512, 1280, 960, 64)
        prefer_speed = st.toggle("Intelligent method selection (recommended)", value=True)
        run = st.button("Run forensic analysis", type="primary", use_container_width=True)

    image = None
    filename = "image"
    if demo:
        image, _gt = copy_move_forge(make_textured_base())
        filename = "synthetic_copy_move.png"
        st.session_state.input_image = image
        st.session_state.input_name = filename
    if uploaded is not None:
        image, _ext = decode_bytes(uploaded.getvalue(), uploaded.name)
        filename = uploaded.name
        st.session_state.input_image = image
        st.session_state.input_name = filename
    if image is None:
        image = st.session_state.get("input_image")
        filename = st.session_state.get("input_name", filename)

    if image is None:
        st.write("Upload a photograph or load the synthetic demo, then run analysis.")
        st.markdown(
            """
            **Pipeline:** validate → fast screen → SIFT/ORB + RANSAC → optional noise/color/edge
            → evidence fusion → morphological localization → measured evaluation.

            **On purpose we do not** run every expensive method on every image.
            """
        )
        return

    h, w = image.shape[:2]
    st.caption(f"Loaded `{filename}` — {w}×{h} px, {image.shape[2] if image.ndim == 3 else 1} channel(s)")

    if run or demo:
        pipe = ForensicPipeline(max_side=max_side, prefer_speed=prefer_speed)
        with st.status("Running multi-stage forensic pipeline…", expanded=True) as status:
            st.write("Preprocess and screen")
            result = pipe.run(image, filename)
            st.write(
                f"Selected: copy-move={result.plan.run_copy_move} "
                f"({'SIFT' if result.plan.use_sift else 'ORB'}) · "
                f"noise={result.plan.run_noise} · color={result.plan.run_color} · edge={result.plan.run_edge}"
            )
            status.update(label=f"Done in {result.total_time:.2f}s", state="complete")
        st.session_state.result = result

    if "result" in st.session_state:
        show_result(st.session_state.result)
        md = to_markdown(st.session_state.result)
        st.download_button("Download report (Markdown)", md, file_name="forensic_report.md", mime="text/markdown")


if __name__ == "__main__":
    main()
