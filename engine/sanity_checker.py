from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _safe_import_yaml() -> Any:
    try:
        import yaml

        return yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required for sanity checks. Install it with: pip install pyyaml"
        ) from exc


def _resolve_split_path(split_value: str, dataset_root: Path) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path.resolve()


def _to_labels_dir(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def _load_data_yaml(data_yaml_path: Path) -> tuple[dict[str, Any], Path, list[str]]:
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    yaml = _safe_import_yaml()
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Invalid data.yaml: expected mapping")

    required_keys = ("train", "val", "names")
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Invalid data.yaml: missing keys {missing}")

    base_dir = data_yaml_path.parent.resolve()
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (base_dir / root).resolve()

    names = cfg["names"]
    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    else:
        raise ValueError("Invalid names in data.yaml: expected list or dict")

    if not class_names:
        raise ValueError("Invalid names in data.yaml: no classes configured")

    return cfg, root, class_names


def _validate_label_line(
    line: str,
    label_path: Path,
    line_number: int,
    class_count: int,
) -> list[str]:
    errors: list[str] = []
    parts = line.split()
    if len(parts) != 5:
        errors.append(
            f"{label_path}:{line_number} invalid format (expected 5 values, got {len(parts)})"
        )
        return errors

    class_raw, *coords_raw = parts
    try:
        class_id = int(class_raw)
    except ValueError:
        errors.append(f"{label_path}:{line_number} class_id is not an integer: {class_raw}")
        return errors

    if class_id < 0 or class_id >= class_count:
        errors.append(
            f"{label_path}:{line_number} class_id {class_id} out of range [0, {class_count - 1}]"
        )

    coord_names = ("x_center", "y_center", "width", "height")
    coords: list[float] = []
    for value in coords_raw:
        try:
            coords.append(float(value))
        except ValueError:
            errors.append(f"{label_path}:{line_number} non-float coordinate: {value}")
            return errors

    for coord_name, coord in zip(coord_names, coords):
        if coord < 0.0 or coord > 1.0:
            errors.append(
                f"{label_path}:{line_number} {coord_name}={coord} outside normalized range [0, 1]"
            )

    if coords[2] <= 0.0 or coords[3] <= 0.0:
        errors.append(
            f"{label_path}:{line_number} width/height must be > 0 (got {coords[2]}, {coords[3]})"
        )

    return errors


def _scan_split(
    split_name: str,
    images_dir: Path,
    labels_dir: Path,
    class_count: int,
) -> tuple[list[str], list[str], int, int, int]:
    warnings: list[str] = []
    errors: list[str] = []

    if not images_dir.exists():
        errors.append(f"[{split_name}] images directory not found: {images_dir}")
        return warnings, errors, 0, 0, 0
    if not labels_dir.exists():
        errors.append(f"[{split_name}] labels directory not found: {labels_dir}")
        return warnings, errors, 0, 0, 0

    image_files = sorted(
        file_path
        for file_path in images_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_SUFFIXES
    )
    label_files = sorted(
        file_path
        for file_path in labels_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".txt"
    )

    image_stems = {path.stem for path in image_files}
    label_stems = {path.stem for path in label_files}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    if missing_labels:
        errors.append(
            f"[{split_name}] {len(missing_labels)} images are missing label files (example: {missing_labels[0]})"
        )
    if orphan_labels:
        warnings.append(
            f"[{split_name}] {len(orphan_labels)} label files have no matching image (example: {orphan_labels[0]})"
        )

    annotation_count = 0
    non_empty_label_files = 0

    for label_path in label_files:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        active_lines = [line.strip() for line in lines if line.strip()]
        if active_lines:
            non_empty_label_files += 1

        for index, line in enumerate(active_lines, start=1):
            line_errors = _validate_label_line(
                line=line,
                label_path=label_path,
                line_number=index,
                class_count=class_count,
            )
            errors.extend(line_errors)
            if not line_errors:
                annotation_count += 1

    return warnings, errors, len(image_files), len(label_files), annotation_count


def run_sanity_check(data_yaml_path: Path, strict: bool = False) -> None:
    cfg, root, class_names = _load_data_yaml(data_yaml_path.resolve())

    all_warnings: list[str] = []
    all_errors: list[str] = []
    split_summary: list[str] = []

    for split_name in ("train", "val", "test"):
        split_value = cfg.get(split_name)
        if split_value is None:
            continue

        images_dir = _resolve_split_path(str(split_value), root)
        labels_dir = _to_labels_dir(images_dir)

        warnings, errors, image_count, label_count, annotation_count = _scan_split(
            split_name=split_name,
            images_dir=images_dir,
            labels_dir=labels_dir,
            class_count=len(class_names),
        )
        all_warnings.extend(warnings)
        all_errors.extend(errors)
        split_summary.append(
            f"[{split_name}] images={image_count}, labels={label_count}, annotations={annotation_count}"
        )

    print("[sanity] dataset summary")
    print(f"[sanity] classes ({len(class_names)}): {class_names}")
    for line in split_summary:
        print(f"[sanity] {line}")

    for warning in all_warnings:
        print(f"[warn] {warning}")

    if all_errors:
        preview = "\n".join(all_errors[:20])
        suffix = "" if len(all_errors) <= 20 else f"\n... and {len(all_errors) - 20} more errors"
        raise ValueError(f"Sanity check failed with {len(all_errors)} error(s):\n{preview}{suffix}")

    if strict and all_warnings:
        preview = "\n".join(all_warnings[:20])
        suffix = "" if len(all_warnings) <= 20 else f"\n... and {len(all_warnings) - 20} more warnings"
        raise ValueError(f"Sanity check strict mode failed with warnings:\n{preview}{suffix}")

    print("[ok] sanity check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset sanity checks for YOLO training")
    parser.add_argument("--data", type=Path, default=Path("data/data.yaml"), help="Path to dataset YAML")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sanity_check(args.data, strict=args.strict)


if __name__ == "__main__":
    main()
