from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_SOURCE_DIR = Path("data/soccernet/converted")
DEFAULT_TARGET_DIR = Path("data")
CLASS_NAMES = ["player", "ball", "referee"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a converted YOLO dataset into train/val/test folders and write data.yaml."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing converted images and labels.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=DEFAULT_TARGET_DIR,
        help="Output directory for train/val/test folders.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio (0.0 to 1.0)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio (0.0 to 1.0)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio (0.0 to 1.0)")
    return parser.parse_args()


def _iter_image_files(source_dir: Path) -> list[Path]:
    return sorted(
        p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def _build_label_path(image_path: Path, source_dir: Path) -> Path:
    rel = image_path.relative_to(source_dir)
    label_path = source_dir.parent / "labels" / rel.with_suffix(".txt")
    return label_path


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _write_yaml(target_dir: Path) -> None:
    yaml_path = target_dir / "data.yaml"
    content = """# YOLO dataset config for football analytics\npath: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: player\n  1: ball\n  2: referee\n"""
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[done] dataset config written to {yaml_path}")


def main() -> None:
    args = parse_args()
    if not (args.train_ratio + args.val_ratio + args.test_ratio) > 0.99:
        raise ValueError("Train/val/test ratios must sum to approximately 1.0")

    source_dir = args.source_dir.resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir = args.target_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_image_files(source_dir)
    if not images:
        raise FileNotFoundError(f"No images found under {source_dir}")

    sorted_images = sorted(images, key=lambda p: p.name)
    n = len(sorted_images)
    train_count = max(1, int(n * args.train_ratio))
    val_count = max(1, int(n * args.val_ratio))
    test_count = max(1, n - train_count - val_count)

    split_map: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for idx, image_path in enumerate(sorted_images):
        if idx < train_count:
            split_map["train"].append(image_path)
        elif idx < train_count + val_count:
            split_map["val"].append(image_path)
        else:
            split_map["test"].append(image_path)

    for split_name in ("train", "val", "test"):
        split_image_dir = output_dir / "images" / split_name
        split_label_dir = output_dir / "labels" / split_name
        split_image_dir.mkdir(parents=True, exist_ok=True)
        split_label_dir.mkdir(parents=True, exist_ok=True)

        for image_path in split_map[split_name]:
            target_image = split_image_dir / image_path.name
            label_path = _build_label_path(image_path, source_dir)
            if label_path.exists():
                target_label = split_label_dir / label_path.name
                _copy_file(label_path, target_label)
            _copy_file(image_path, target_image)

    _write_yaml(output_dir)
    print("[done] dataset split complete")


if __name__ == "__main__":
    main()
