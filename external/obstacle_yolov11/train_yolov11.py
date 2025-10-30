import argparse
from pathlib import Path
from typing import List

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv11 model on a prepared dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root folder containing images/{train,val,test} and labels/{train,val,test}.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pretrained YOLOv11 checkpoint to fine-tune (e.g., yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Square training resolution.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size used for training.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device (''=auto, '0'=GPU0, 'cpu'=force CPU, etc.).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Ultralytics project directory for training runs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="obstacle_yolov11",
        help="Run name used inside the project directory.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["cylindrical_cone", "wall_type_1", "wall_type_2"],
        help="Class names in order of their label indices.",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="dataset.yaml",
        help="Filename for the auto-generated dataset config.",
    )
    return parser.parse_args()


def ensure_dataset_structure(root: Path) -> None:
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "images" / "test",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "test",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        missing_str = "\n".join(str(p) for p in missing)
        raise FileNotFoundError(
            "Dataset structure incomplete. Expected folders:\n" f"{missing_str}"
        )


def write_dataset_yaml(root: Path, yaml_path: Path, class_names: List[str]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    ensure_dataset_structure(args.dataset_root)

    yaml_path = args.dataset_root / args.yaml_name
    write_dataset_yaml(args.dataset_root, yaml_path, args.class_names)

    # Try to load pretrained weights; if not present (offline),
    # fall back to training from a bundled YAML definition (YOLOv8).
    try:
        model = YOLO(args.model)
    except Exception as e:
        fallback = "yolov8.yaml"
        print(f"[WARN] Could not load '{args.model}' ({e}). Falling back to '{fallback}'.")
        model = YOLO(fallback)
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    print("Training complete. Checkpoints (including best.pt) saved under:", args.project)


if __name__ == "__main__":
    main()
