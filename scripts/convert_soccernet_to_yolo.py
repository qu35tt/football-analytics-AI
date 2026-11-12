from __future__ import annotations

import argparse
import configparser
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from datasets import load_dataset


CLASS_NAMES = ["player", "ball", "referee"]
DEFAULT_DATASET_NAME = "SoccerNet/SN-Tracking-2023"
DEFAULT_OUTPUT_DIR = Path("data/soccernet/converted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load SoccerNet tracking data from Hugging Face and export YOLO detection labels plus tracking metadata."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="soccernet",
        choices=["soccernet", "snmot"],
        help="Dataset source format to convert: soccernet or local SNMOT tracking sequences.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Hugging Face dataset name to load, e.g. SoccerNet/SN-Tracking-2023",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of samples to export for a quick smoke test.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("datasets/raw"),
        help="Directory containing local SNMOT sequences when --dataset-type snmot is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where converted YOLO dataset files will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio to apply after export if you want a quick split.",
    )
    return parser.parse_args()


def _normalize_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(width)))
    y1 = max(0.0, min(float(y1), float(height)))
    x2 = max(0.0, min(float(x2), float(width)))
    y2 = max(0.0, min(float(y2), float(height)))
    if x2 <= x1 or y2 <= y1:
        return (0.0, 0.0, 0.0, 0.0)

    x_center = (x1 + x2) / 2.0 / width
    y_center = (y1 + y2) / 2.0 / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    if box_w <= 0.0 or box_h <= 0.0:
        return (0.0, 0.0, 0.0, 0.0)
    return (x_center, y_center, box_w, box_h)


def _safe_get(sample: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    return None


def _extract_boxes_and_labels(sample: dict[str, Any], image_width: int, image_height: int) -> list[tuple[int, float, float, float, float]]:
    candidates: list[tuple[int, float, float, float, float]] = []

    annotations = _safe_get(sample, "annotations", "boxes", "objects", "detections")
    if annotations is None:
        return candidates

    if isinstance(annotations, dict):
        box_list = _safe_get(annotations, "boxes", "bboxes", "bbox", "coordinates")
        label_list = _safe_get(annotations, "labels", "class_ids", "classes", "category_ids")
        track_list = _safe_get(annotations, "track_ids", "ids")
        if box_list is not None:
            box_entries = box_list if isinstance(box_list, list) else [box_list]
            label_entries = label_list if isinstance(label_list, list) else ([] if label_list is None else [label_list])
            for idx, box in enumerate(box_entries):
                if isinstance(box, dict):
                    x1 = float(box.get("x", box.get("x1", 0.0)))
                    y1 = float(box.get("y", box.get("y1", 0.0)))
                    x2 = float(box.get("x2", box.get("x_max", x1)))
                    y2 = float(box.get("y2", box.get("y_max", y1)))
                    label_id = int(box.get("class_id", box.get("label", 0)))
                elif isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    label_id = int(label_entries[idx]) if idx < len(label_entries) else 0
                else:
                    continue

                cx, cy, bw, bh = _normalize_box(x1, y1, x2, y2, image_width, image_height)
                if bw <= 0.0 or bh <= 0.0:
                    continue
                candidates.append((label_id, cx, cy, bw, bh))
            return candidates

    if isinstance(annotations, list):
        for item in annotations:
            if isinstance(item, dict):
                box = _safe_get(item, "bbox", "box", "coordinates")
                label_id = _safe_get(item, "class_id", "label", "category_id")
                if box is None:
                    continue
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = [float(v) for v in box[:4]]
                    class_id = int(label_id) if label_id is not None else 0
                    cx, cy, bw, bh = _normalize_box(x1, y1, x2, y2, image_width, image_height)
                    if bw > 0.0 and bh > 0.0:
                        candidates.append((class_id, cx, cy, bw, bh))
            elif isinstance(item, (list, tuple)) and len(item) >= 5:
                box = item[:4]
                class_id = int(item[4]) if item[4] is not None else 0
                cx, cy, bw, bh = _normalize_box(*[float(v) for v in box], image_width, image_height)
                if bw > 0.0 and bh > 0.0:
                    candidates.append((class_id, cx, cy, bw, bh))

    return candidates


def _write_yolo_label(label_path: Path, detections: list[tuple[int, float, float, float, float]]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w", encoding="utf-8") as handle:
        for class_id, cx, cy, bw, bh in detections:
            if bw <= 0.0 or bh <= 0.0:
                continue
            handle.write(f"{int(class_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def _write_track_meta(meta_path: Path, payload: dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _extract_image(sample: dict[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    for key in ("image", "frame", "img", "images"):
        value = sample.get(key)
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            return value, {"source_key": key}
        if isinstance(value, dict):
            array = value.get("array")
            if isinstance(array, np.ndarray):
                return array, {"source_key": key}
    return None, {}


def _ensure_dataset_split_directories(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "train_images": output_dir / "images" / "train",
        "val_images": output_dir / "images" / "val",
        "test_images": output_dir / "images" / "test",
        "train_labels": output_dir / "labels" / "train",
        "val_labels": output_dir / "labels" / "val",
        "test_labels": output_dir / "labels" / "test",
        "tracks": output_dir / "tracks",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _snmot_class_id(tracklet_name: str) -> int:
    name = tracklet_name.lower()
    if "ball" in name:
        return 1
    if "referee" in name:
        return 2
    if "goalkeeper" in name or "player" in name:
        return 0
    return 0


def _parse_snmot_gameinfo(seq_dir: Path) -> dict[int, int]:
    ini_path = seq_dir / "gameinfo.ini"
    if not ini_path.exists():
        raise FileNotFoundError(f"Missing gameinfo.ini in {seq_dir}")

    parser = configparser.ConfigParser()
    parser.read(ini_path, encoding="utf-8")
    mapping: dict[int, int] = {}
    section = parser["Sequence"]
    for key, value in section.items():
        if not key.startswith("trackletID_"):
            continue
        body = value.strip()
        if ";" not in body:
            continue
        label_name, track_id_raw = [part.strip() for part in body.split(";", 1)]
        try:
            track_id = int(track_id_raw)
        except ValueError:
            continue
        mapping[track_id] = _snmot_class_id(label_name)
    return mapping


def _export_snmot_to_yolo(source_dir: Path, output_dir: Path, train_ratio: float) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"SNMOT source directory not found: {source_dir}")

    if source_dir.name.startswith("SNMOT-") and (source_dir / "img1").exists() and (source_dir / "gt").exists():
        seq_dirs = [source_dir]
    else:
        seq_dirs = [
            p for p in sorted(source_dir.iterdir())
            if p.is_dir() and p.name.startswith("SNMOT-")
        ]

    if not seq_dirs:
        raise FileNotFoundError(f"No SNMOT sequence folders found under {source_dir}")

    image_paths: list[Path] = []
    gt_by_frame: dict[int, list[tuple[int, float, float, float, float]]] = {}
    width = height = None

    for seq_dir in seq_dirs:
        seq_info = seq_dir / "seqinfo.ini"
        if seq_info.exists():
            parser = configparser.ConfigParser()
            parser.read(seq_info, encoding="utf-8")
            if "Sequence" in parser:
                section = parser["Sequence"]
                if "imWidth" in section and "imHeight" in section:
                    width = int(section["imWidth"])
                    height = int(section["imHeight"])

        img_dir = seq_dir / "img1"
        gt_path = seq_dir / "gt" / "gt.txt"
        if not img_dir.exists() or not gt_path.exists():
            continue

        tracklet_map = _parse_snmot_gameinfo(seq_dir)
        with gt_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if not row or not row[0].strip() or row[0].startswith("#"):
                    continue
                try:
                    frame_id = int(row[0])
                    track_id = int(row[1])
                    x = float(row[2])
                    y = float(row[3])
                    w = float(row[4])
                    h = float(row[5])
                except (ValueError, IndexError):
                    continue

                class_id = tracklet_map.get(track_id, 0)
                if width is None or height is None:
                    width = max(int(x + w), 1)
                    height = max(int(y + h), 1)

                cx, cy, bw, bh = _normalize_box(x, y, x + w, y + h, width, height)
                if bw <= 0.0 or bh <= 0.0:
                    continue
                gt_by_frame.setdefault(frame_id, []).append((class_id, cx, cy, bw, bh))

        for image_path in sorted(img_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(image_path)

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dirs = _ensure_dataset_split_directories(output_dir)
    image_paths = sorted(image_paths, key=lambda p: int(p.stem))
    train_count = max(1, int(len(image_paths) * train_ratio))
    val_count = max(1, int(len(image_paths) * (1.0 - train_ratio) / 2.0))
    test_count = max(1, len(image_paths) - train_count - val_count)

    for index, image_path in enumerate(image_paths):
        if index < train_count:
            split_name = "train"
        elif index < train_count + val_count:
            split_name = "val"
        else:
            split_name = "test"

        image_target = dirs[f"{split_name}_images"] / image_path.name
        shutil.copy2(image_path, image_target)

        frame_id = int(image_path.stem)
        detections = gt_by_frame.get(frame_id, [])
        label_target = dirs[f"{split_name}_labels"] / image_path.with_suffix(".txt").name
        _write_yolo_label(label_target, detections)

    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: player\n  1: ball\n  2: referee\n",
        encoding="utf-8",
    )
    print(f"[done] converted {len(image_paths)} SNMOT frames to {output_dir}")
    print(f"[done] YOLO config written to {yaml_path}")


def _copy_to_yolo_format(dataset_name: str, split_name: str, output_dir: Path, limit: int | None, train_ratio: float) -> None:
    ds = load_dataset(dataset_name, split=split_name)
    records = ds if limit is None else ds.select(range(min(limit, len(ds))))

    dirs = _ensure_dataset_split_directories(output_dir)
    train_items: list[Path] = []
    val_items: list[Path] = []

    for sample_index, sample in enumerate(records):
        image, _ = _extract_image(sample)
        if image is None:
            print(f"[warn] sample {sample_index} has no image/frame field. Available keys: {list(sample.keys())[:20]}")
            continue

        image = np.asarray(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        height, width = image.shape[:2]
        file_stem = f"{split_name}_{sample_index:05d}"
        image_path = dirs["train_images"] if sample_index < max(1, int(len(records) * train_ratio)) else dirs["val_images"]
        image_path = image_path / f"{file_stem}.jpg"
        cv2.imwrite(str(image_path), image)

        detections = _extract_boxes_and_labels(sample, width, height)
        label_path = (dirs["train_labels"] if sample_index < max(1, int(len(records) * train_ratio)) else dirs["val_labels"]) / f"{file_stem}.txt"
        _write_yolo_label(label_path, detections)

        track_meta = {
            "dataset": dataset_name,
            "split": split_name,
            "sample_index": sample_index,
            "source_keys": sorted(sample.keys()),
            "detections": [
                {"class_id": int(class_id), "cx": float(cx), "cy": float(cy), "w": float(w), "h": float(h)}
                for class_id, cx, cy, w, h in detections
            ],
        }
        _write_track_meta(dirs["tracks"] / f"{file_stem}.json", track_meta)

        if sample_index < max(1, int(len(records) * train_ratio)):
            train_items.append(image_path)
        else:
            val_items.append(image_path)

    print(f"[done] exported {len(train_items)} train samples and {len(val_items)} val samples")
    print(f"[done] output directory: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_type == "snmot":
        _export_snmot_to_yolo(
            source_dir=args.source_dir.resolve(),
            output_dir=output_dir,
            train_ratio=args.train_ratio,
        )
        return

    try:
        _copy_to_yolo_format(
            dataset_name=args.dataset_name,
            split_name=args.split,
            output_dir=output_dir,
            limit=args.limit,
            train_ratio=args.train_ratio,
        )
    except ValueError as exc:
        print(f"[error] {exc}")
        raise SystemExit(1) from exc

    yaml_path = output_dir.parent / "data.yaml"
    yaml_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: player\n  1: ball\n  2: referee\n",
        encoding="utf-8",
    )
    print(f"[done] YOLO config written to {yaml_path}")


if __name__ == "__main__":
    main()
