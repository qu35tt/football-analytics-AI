from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


DEFAULT_SOURCE = Path("video.mp4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference (and optional tracking) with trained YOLO weights."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/train/football-yolo11s/weights/best.pt"),
        help="Path to model weights (.pt)",
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Image/video/folder source")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device, e.g. 0 (GPU), 0,1 (multi-GPU), or cpu",
    )
    parser.add_argument("--project", type=Path, default=Path("runs/predict"), help="Output project directory")
    parser.add_argument("--name", type=str, default="football-infer", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwrite of existing run folder")
    parser.add_argument("--save-txt", action="store_true", help="Save detections as YOLO txt labels")
    parser.add_argument("--save-conf", action="store_true", help="Save confidence in label files")
    parser.add_argument("--track", action="store_true", help="Use ByteTrack during inference")
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config when --track is enabled",
    )
    return parser.parse_args()


def run_inference() -> None:
    args = parse_args()

    weights = args.weights.resolve()
    source = args.source.resolve()
    project = args.project.resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    project.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))

    common_kwargs = dict(
        source=str(source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str(project),
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        verbose=False,
    )

    if args.track:
        results = model.track(
            **common_kwargs,
            tracker=args.tracker,
            persist=True,
        )
        mode = "tracking"
    else:
        results = model.predict(**common_kwargs)
        mode = "prediction"

    print(f"[done] {mode} completed")
    print(f"[info] frames/items processed: {len(results)}")
    print(f"[info] output directory: {project / args.name}")


if __name__ == "__main__":
    run_inference()
