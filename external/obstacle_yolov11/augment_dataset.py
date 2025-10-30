import argparse
import random
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read image at unicode path using OpenCV."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write image to unicode path using OpenCV."""
    # Ensure parent directory exists (robust to external deletions)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        raise RuntimeError(f"Failed to encode image for {path}")
    encoded.tofile(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heavy augmentations for YOLO-format datasets."
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Path to the folder containing the original images.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to the folder containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination root where augmented images/labels will be written.",
    )
    parser.add_argument(
        "--per-image",
        type=int,
        default=10,
        help="Number of augmented samples to create per source image.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Target square resolution (pixels) for augmented outputs.",
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        help="Image format/extension for augmented outputs (jpg, png, ...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility.",
    )
    parser.add_argument(
        "--min-box-visibility",
        type=float,
        default=0.1,
        help="Drop boxes whose visible area falls below this threshold (0-1).",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Copy original image/label into the output set before augmentations.",
    )
    return parser.parse_args()


def build_pipeline(min_box_visibility: float, img_size: int) -> A.Compose:
    """Create a rich augmentation pipeline covering geometry & photometric changes."""

    # Albumentations applies transforms sequentially; probabilities drive variety.
    smaller_side = max(32, int(img_size * 0.75))
    larger_side = max(32, int(img_size * 1.25))

    return A.Compose(
        [
            A.OneOrOther(
                first=A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.5, 1.0),
                    ratio=(0.75, 1.33),
                    p=0.7,
                ),
                second=A.PadIfNeeded(
                    min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_REFLECT_101
                ),
                p=1.0,
            ),
            A.OneOf(
                [
                    A.Resize(
                        height=smaller_side,
                        width=smaller_side,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,
                    ),
                    A.Resize(
                        height=larger_side,
                        width=larger_side,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.5,
                    ),
                    A.NoOp(),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.Affine(
                        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                        scale=(0.8, 1.15),
                        rotate=(-20, 20),
                        shear=(-12, 12),
                        fit_output=False,
                        p=0.8,
                    ),
                    A.Perspective(scale=(0.05, 0.12), p=0.3),
                ],
                p=0.9,
            ),
            A.OneOf(
                [
                    A.RandomBrightnessContrast(p=0.7),
                    A.CLAHE(clip_limit=(1, 4), p=0.3),
                    A.ColorJitter(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ],
                p=0.9,
            ),
            A.Downscale(scale_range=(0.6, 0.95), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.2),
                    A.GaussianBlur(blur_limit=5, p=0.3),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=0.3),
                    A.ISONoise(p=0.3),
                    A.MultiplicativeNoise(p=0.3),
                ],
                p=0.4,
            ),
            A.RandomGamma(p=0.3),
            A.RandomFog(p=0.15),
            A.RandomRain(p=0.15),
            A.RandomShadow(p=0.15),
            A.RandomSunFlare(p=0.1),
            A.Resize(height=img_size, width=img_size),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=min_box_visibility,
        ),
    )


def read_labels(label_path: Path) -> Tuple[List[Tuple[int, float, float, float, float]], List[str]]:
    """Read a YOLO-format .txt file into numeric boxes + class labels."""
    boxes: List[Tuple[int, float, float, float, float]] = []
    labels: List[str] = []
    if not label_path.exists():
        return boxes, labels

    with label_path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                continue
            boxes.append((cls, x, y, w, h))
            labels.append(str(cls))
    return boxes, labels


def write_labels(label_path: Path, boxes: List[Tuple[int, float, float, float, float]]) -> None:
    with label_path.open("w") as f:
        for cls, x, y, w, h in boxes:
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def ensure_output_dirs(root: Path) -> Tuple[Path, Path]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    if not args.images.exists():
        raise FileNotFoundError(f"Image directory not found: {args.images}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Label directory not found: {args.labels}")

    output_images_dir, output_labels_dir = ensure_output_dirs(args.output)
    pipeline = build_pipeline(args.min_box_visibility, args.img_size)

    image_paths = sorted(
        [p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not image_paths:
        raise RuntimeError(f"No image files discovered in {args.images}")

    for image_path in image_paths:
        label_path = args.labels / f"{image_path.stem}.txt"
        boxes_raw, class_labels = read_labels(label_path)

        # Copy clean original to output when requested.
        if args.keep_original:
            original_dst = output_images_dir / f"{image_path.stem}_orig.{args.image_format}"
            label_dst = output_labels_dir / f"{image_path.stem}_orig.txt"
            image = read_image(image_path)
            if image is None:
                print(f"[WARN] Could not read image: {image_path}")
            else:
                write_image(original_dst, image)
                write_labels(label_dst, boxes_raw)

        if not boxes_raw:
            print(f"[WARN] No labels for {image_path.name}; augmentations will be skipped.")
            continue

        image_np = read_image(image_path)
        if image_np is None:
            print(f"[WARN] Could not decode image: {image_path}")
            continue

        augments_saved = 0
        attempts = 0
        max_attempts = args.per_image * 4

        while augments_saved < args.per_image and attempts < max_attempts:
            attempts += 1

            transformed = pipeline(
                image=image_np,
                bboxes=[b[1:] for b in boxes_raw],
                class_labels=class_labels,
            )
            aug_image = transformed["image"]
            aug_boxes_xywh = transformed["bboxes"]
            aug_labels = transformed["class_labels"]

            if not aug_boxes_xywh:
                # Skip if all boxes disappeared after augmentation.
                continue

            aug_boxes = [
                (
                    int(float(cls)),
                    float(x),
                    float(y),
                    float(w),
                    float(h),
                )
                for cls, (x, y, w, h) in zip(aug_labels, aug_boxes_xywh)
            ]

            image_name = f"{image_path.stem}_aug_{augments_saved:03d}.{args.image_format}"
            label_name = f"{image_path.stem}_aug_{augments_saved:03d}.txt"

            output_image_path = output_images_dir / image_name
            output_label_path = output_labels_dir / label_name

            write_image(output_image_path, aug_image)
            write_labels(output_label_path, aug_boxes)
            augments_saved += 1

        if augments_saved < args.per_image:
            print(
                f"[WARN] Generated {augments_saved} augmentations for {image_path.name}. "
                "Consider relaxing augmentation parameters."
            )

    print(f"Augmentations complete. Outputs saved in {args.output}")


if __name__ == "__main__":
    main()
