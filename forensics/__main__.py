"""Command-line entry: python -m forensics --demo | python -m forensics photo.jpg"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .evaluation import (
    authentic_false_positive,
    copy_move_forge,
    make_textured_base,
    run_ablation,
    run_robustness,
)
from .export import save_result
from .pipeline import ForensicPipeline


def _print_rows(title: str, rows) -> None:
    print(f"\n{title}")
    print(f"{'name':<24} {'P':>6} {'R':>6} {'F1':>6} {'IoU':>6} {'FPR':>8} {'det':>5} {'sec':>7}")
    for r in rows:
        print(
            f"{r.name:<24} {r.precision:6.3f} {r.recall:6.3f} {r.f1:6.3f} "
            f"{r.iou:6.3f} {r.fpr:8.4f} {str(r.detected):>5} {r.seconds:7.3f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-stage classical image forensic screening (NumPy/OpenCV)."
    )
    parser.add_argument("image", nargs="?", help="Path to JPEG/PNG photograph")
    parser.add_argument("--demo", action="store_true", help="Analyze a built-in synthetic copy-move")
    parser.add_argument("--eval", action="store_true", help="Run robustness + ablation experiments")
    parser.add_argument("--out", default="outputs", help="Folder for saved images and report")
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--all-methods", action="store_true", help="Disable intelligent skipping")
    args = parser.parse_args(argv)

    if args.eval:
        print("Running controlled synthetic experiments (not a real-world accuracy claim)...")
        _print_rows("Robustness", run_robustness(max_side=640))
        _print_rows("Ablation", run_ablation(max_side=640))
        _print_rows("Authentic image", [authentic_false_positive(max_side=640)])
        if not args.demo and args.image is None:
            return 0

    if args.demo:
        image, _gt = copy_move_forge(make_textured_base())
        filename = "synthetic_copy_move.png"
    elif args.image:
        path = Path(args.image)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"Could not read: {path}")
        filename = path.name
    else:
        parser.print_help()
        print("\nExamples:\n  python -m forensics --demo\n  python -m forensics photo.jpg --out outputs")
        return 2

    pipe = ForensicPipeline(max_side=args.max_side, prefer_speed=not args.all_methods)
    result = pipe.run(image, filename)
    out = save_result(result, args.out)
    print(result.interpretation)
    print(f"Total time: {result.total_time:.3f}s")
    print(f"Saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
