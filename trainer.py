from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from engine.sanity_checker import run_sanity_check
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT / "datasets" / "prepared" / "data" / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO model for football analytics detection tasks."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to dataset YAML",
    )
    parser.add_argument("--model", type=str, default="yolo11m.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default=0,
        help="Training device, e.g. 0 (GPU), 0,1 (multi-GPU), or cpu",
    )
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--project", type=Path, default=Path("runs/train"), help="Output project directory")
    parser.add_argument("--name", type=str, default="football-yolo11s", help="Run name")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwrite of existing run folder")
    parser.add_argument("--no-val", action="store_true", help="Skip final validation after training")
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip dataset sanity checks before training",
    )
    parser.add_argument(
        "--strict-sanity",
        action="store_true",
        help="Treat sanity-check warnings as errors",
    )
    return parser.parse_args()


def _safe_import_yaml() -> Any:
    try:
        import yaml

        return yaml
    except Exception as exc:
        raise ImportError(
            "PyYAML is required to validate data.yaml before training. "
            "Install it with: pip install pyyaml"
        ) from exc


def _resolve_split_path(split_value: str, dataset_root: Path) -> Path:
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = dataset_root / split_path
    return split_path.resolve()


def validate_data_yaml(data_yaml_path: Path) -> dict[str, Any]:
    if not data_yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

    yaml = _safe_import_yaml()
    cfg = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Invalid data.yaml: expected key-value mapping")

    required_keys = ("train", "val", "names")
    missing_keys = [key for key in required_keys if key not in cfg]
    if missing_keys:
        raise ValueError(f"Invalid data.yaml: missing keys {missing_keys}")

    base_dir = data_yaml_path.parent.resolve()
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (base_dir / root).resolve()

    for split in ("train", "val"):
        split_path = _resolve_split_path(str(cfg[split]), root)
        if not split_path.exists():
            raise FileNotFoundError(f"{split} path from data.yaml does not exist: {split_path}")

    test_value = cfg.get("test")
    if test_value is not None:
        test_path = _resolve_split_path(str(test_value), root)
        if not test_path.exists():
            print(f"[warn] test path does not exist yet: {test_path}")

    names = cfg["names"]
    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        class_names = [names[k] for k in sorted(names)]
    else:
        raise ValueError("Invalid names in data.yaml: expected list or dict")

    if len(class_names) == 0:
        raise ValueError("Invalid names in data.yaml: at least one class is required")

    print(f"[ok] data.yaml validated: {data_yaml_path}")
    print(f"[info] classes ({len(class_names)}): {class_names}")
    return cfg


def train() -> None:
    args = parse_args()

    data_yaml = args.data.resolve()
    validate_data_yaml(data_yaml)
    if not args.skip_sanity_check:
        run_sanity_check(data_yaml, strict=args.strict_sanity)

    project_dir = args.project.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"[info] starting training with model: {args.model}")
    print(f"[info] device: {args.device}")

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(project_dir),
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        resume=args.resume,
        cache=args.cache,
        exist_ok=args.exist_ok,
    )

    save_dir = Path(getattr(results, "save_dir", project_dir / args.name))
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print(f"[done] training outputs: {save_dir}")
    if best_weights.exists():
        print(f"[done] best weights: {best_weights}")
    if last_weights.exists():
        print(f"[done] last weights: {last_weights}")

    if not args.no_val:
        print("[info] running final validation...")
        model.val(
            data=str(data_yaml),
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project_dir),
            name=f"{args.name}-val",
            exist_ok=True,
        )
        print("[done] validation completed")


if __name__ == "__main__":
    train()
