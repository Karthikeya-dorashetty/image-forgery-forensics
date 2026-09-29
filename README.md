# Multi-Stage Image Forgery Detection (Classical CV)

Software-only **preliminary** screening for copy-move and localized editing using **Python, NumPy, and OpenCV**.

It does **not** legally prove that an image is forged.

## Install and run

```text
cd image-forgery-forensics
python -m pip install -r requirements.txt

python -m streamlit run app.py
python run.py --demo
python run.py path\to\photo.jpg --out outputs
python run.py --eval
```

## Complete source files

| File | Role |
|---|---|
| `app.py` | Streamlit UI (upload, tabs, report download, evaluation) |
| `run.py` | Command-line entry |
| `requirements.txt` | numpy, opencv-python, streamlit, Pillow |
| `forensics/__main__.py` | `python -m forensics` |
| `forensics/pipeline.py` | Screen → selective methods → fuse → localize |
| `forensics/preprocess.py` | Load/validate/optional resize (no denoise) |
| `forensics/screening.py` | FAST/texture/edges decide what to run |
| `forensics/copy_move.py` | SIFT/ORB, ratio test, RANSAC geometry |
| `forensics/noise.py` | Residual variance map |
| `forensics/color.py` | LAB / YCrCb inconsistency map |
| `forensics/edges.py` | Canny + morphological gradient map |
| `forensics/fusion.py` | Weighted agreement fusion |
| `forensics/refine.py` | Morphology, components, boxes |
| `forensics/evaluation.py` | Synthetic GT, robustness, ablation |
| `forensics/report.py` | Markdown report |
| `forensics/export.py` | Save PNG + HTML report |
| `forensics/utils.py` | Shared OpenCV/NumPy helpers |

There are no empty stubs. The detection logic is in those files.

## Pipeline

validate → fast screen → SIFT/ORB + RANSAC → optional noise/color/edge → fusion → morphology → metrics.

## Design choices (jury / PS)

| Question | Choice |
|---|---|
| What runs first? | Validation + cheap screening, then copy-move. |
| What is expensive? | SIFT matching. ORB is the fast path. |
| Primary scenario? | Copy-move: features + geometric verification. |
| Complementary? | Noise/color/edge for localized edits or unverified matching. |
| Redundant? | Color is skipped when copy-move is already a strong, textured case. |
| When to skip? | Too few corners → skip matching. Strong verified copy-move → skip color. |
| Safe preprocess? | 8-bit BGR, optional downscale. No CLAHE, no denoise. |
| Working resolution? | Default max side 960 (slider 512–1280). |
| Weights? | Verified copy-move ~0.62–0.72; others supporting. |
| False positives? | Ratio test, min displacement, RANSAC, morphology, single vs multi-method labels. |

## Limitations

Very small edits, heavy compression, strong blur, stacked edits, and AI-generated images are outside reliable range for this classical pipeline.
