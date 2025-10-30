import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a YOLO-format dataset into train/val/test folders."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to directory containing augmented images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to directory containing augmented YOLO label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination root folder that will receive train/val/test splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of samples assigned to the training split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples assigned to the validation split.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling.",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for image_path in images_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] Label missing for {image_path.name}; skipping.")
            continue
        pairs.append((image_path, label_path))
    return pairs


def prepare_output_dirs(root: Path) -> Dict[str, Tuple[Path, Path]]:
    mapping: Dict[str, Tuple[Path, Path]] = {}
    for split in ("train", "val", "test"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        mapping[split] = (images_dir, labels_dir)
    return mapping


def distribute(
    pairs: List[Tuple[Path, Path]],
    ratios: Tuple[float, float, float],
) -> Dict[str, List[Tuple[Path, Path]]]:
    total = len(pairs)
    train_ratio, val_ratio, test_ratio = ratios
    if train_ratio + val_ratio + test_ratio <= 0:
        raise ValueError("Split ratios must sum to a positive number.")

    # Normalize to 1.0 in case of rounding mistakes by the caller.
    ratio_sum = train_ratio + val_ratio + test_ratio
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    train_cutoff = int(total * train_ratio)
    val_cutoff = train_cutoff + int(total * val_ratio)

    splits: Dict[str, List[Tuple[Path, Path]]] = {
        "train": pairs[:train_cutoff],
        "val": pairs[train_cutoff:val_cutoff],
        "test": pairs[val_cutoff:],
    }

    return splits


def copy_pairs(pairs: List[Tuple[Path, Path]], dest_images: Path, dest_labels: Path) -> None:
    for image_path, label_path in pairs:
        shutil.copy2(image_path, dest_images / image_path.name)
        shutil.copy2(label_path, dest_labels / label_path.name)


def main() -> None:
    args = parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Images directory not found: {args.images}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels directory not found: {args.labels}")

    pairs = collect_pairs(args.images, args.labels)
    if not pairs:
        raise RuntimeError("No image/label pairs found; verify your paths.")

    random.Random(args.seed).shuffle(pairs)

    split_map = distribute(
        pairs,
        ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    output_dirs = prepare_output_dirs(args.output)

    for split, split_pairs in split_map.items():
        images_dir, labels_dir = output_dirs[split]
        copy_pairs(split_pairs, images_dir, labels_dir)
        print(f"{split}: {len(split_pairs)} samples")

    print(f"Split complete. Output dataset root: {args.output}")


if __name__ == "__main__":
    main()
