from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import GENERATED_DATASET_DIR, IMAGE_EXTENSIONS, RAW_DATASET_DIR


def _first_image(folder: Path) -> Path | None:
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            return path
    return None


def _make_background(width: int, height: int) -> np.ndarray:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    base[:] = random.choice([(70, 70, 70), (95, 95, 90), (55, 65, 70), (115, 105, 95)])
    for _ in range(random.randint(8, 24)):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        x2 = random.randint(0, width - 1)
        y2 = random.randint(0, height - 1)
        color = tuple(int(v) for v in np.random.randint(40, 150, size=3))
        cv2.line(base, (x1, y1), (x2, y2), color, random.randint(1, 3), cv2.LINE_AA)

    belt_y = random.randint(height // 3, height - height // 5)
    cv2.rectangle(base, (0, belt_y - 70), (width, belt_y + 90), (45, 48, 48), -1)
    for x in range(-80, width + 80, 80):
        cv2.line(base, (x, belt_y - 70), (x + 55, belt_y + 90), (65, 68, 68), 2)
    noise = np.random.normal(0, random.uniform(3, 10), base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _augment_object(image: np.ndarray, canvas_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    canvas_w, canvas_h = canvas_size
    h, w = image.shape[:2]

    scale = random.uniform(0.35, 1.05)
    target_w = max(20, int(canvas_w * random.uniform(0.18, 0.42) * scale))
    target_h = max(20, int(target_w * h / w))
    if target_h > canvas_h * 0.75:
        target_h = int(canvas_h * 0.75)
        target_w = int(target_h * w / h)

    obj = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        obj = cv2.flip(obj, 1)
    if random.random() < 0.25:
        obj = cv2.flip(obj, 0)

    alpha = np.full(obj.shape[:2], 255, dtype=np.uint8)
    angle = random.uniform(0, 360)
    center = (target_w / 2, target_h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, random.uniform(0.85, 1.2))
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    rotated_w = int((target_h * sin) + (target_w * cos))
    rotated_h = int((target_h * cos) + (target_w * sin))
    matrix[0, 2] += (rotated_w / 2) - center[0]
    matrix[1, 2] += (rotated_h / 2) - center[1]
    obj = cv2.warpAffine(obj, matrix, (rotated_w, rotated_h), borderValue=(0, 0, 0))
    alpha = cv2.warpAffine(alpha, matrix, (rotated_w, rotated_h), borderValue=0)

    src = np.float32([[0, 0], [rotated_w - 1, 0], [rotated_w - 1, rotated_h - 1], [0, rotated_h - 1]])
    shift = min(rotated_w, rotated_h) * random.uniform(0.02, 0.12)
    dst = src + np.float32(np.random.uniform(-shift, shift, size=(4, 2)))
    perspective = cv2.getPerspectiveTransform(src, dst)
    obj = cv2.warpPerspective(obj, perspective, (rotated_w, rotated_h), borderValue=(0, 0, 0))
    alpha = cv2.warpPerspective(alpha, perspective, (rotated_w, rotated_h), borderValue=0)

    beta = random.uniform(-45, 45)
    contrast = random.uniform(0.65, 1.45)
    obj = cv2.convertScaleAbs(obj, alpha=contrast, beta=beta)
    if random.random() < 0.45:
        obj = cv2.GaussianBlur(obj, (random.choice([3, 5]), random.choice([3, 5])), 0)
    if random.random() < 0.6:
        noise = np.random.normal(0, random.uniform(4, 18), obj.shape).astype(np.int16)
        obj = np.clip(obj.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return obj, alpha


def _paste_object(background: np.ndarray, obj: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    canvas_h, canvas_w = background.shape[:2]
    obj_h, obj_w = obj.shape[:2]
    if obj_w >= canvas_w or obj_h >= canvas_h:
        scale = min((canvas_w - 8) / obj_w, (canvas_h - 8) / obj_h)
        obj = cv2.resize(obj, (int(obj_w * scale), int(obj_h * scale)))
        alpha = cv2.resize(alpha, (obj.shape[1], obj.shape[0]))
        obj_h, obj_w = obj.shape[:2]

    x = random.randint(0, max(0, canvas_w - obj_w))
    y = random.randint(0, max(0, canvas_h - obj_h))
    roi = background[y : y + obj_h, x : x + obj_w]
    mask = (alpha.astype(np.float32) / 255.0)[..., None]
    blended = (obj.astype(np.float32) * mask + roi.astype(np.float32) * (1 - mask)).astype(np.uint8)
    background[y : y + obj_h, x : x + obj_w] = blended

    points = cv2.findNonZero((alpha > 20).astype(np.uint8))
    if points is None:
        return background, (x, y, x + obj_w, y + obj_h)
    bx, by, bw, bh = cv2.boundingRect(points)
    return background, (x + bx, y + by, x + bx + bw, y + by + bh)


def _write_yolo_label(path: Path, class_id: int, bbox: tuple[int, int, int, int], width: int, height: int) -> None:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    path.write_text(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n", encoding="utf-8")


def generate_dataset(
    raw_dir: Path = RAW_DATASET_DIR,
    output_dir: Path = GENERATED_DATASET_DIR,
    samples_per_class: int = 200,
    image_size: int = 640,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)

    class_dirs = [folder for folder in sorted(raw_dir.iterdir()) if folder.is_dir() and _first_image(folder)]
    if not class_dirs:
        raise FileNotFoundError(f"No class image folders found in {raw_dir}")

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    class_names = [folder.name for folder in class_dirs]
    for class_id, folder in enumerate(class_dirs):
        source_path = _first_image(folder)
        assert source_path is not None
        source = cv2.imread(str(source_path))
        if source is None:
            raise ValueError(f"Cannot read image: {source_path}")

        for idx in range(samples_per_class):
            split = "val" if random.random() < val_ratio else "train"
            background = _make_background(image_size, image_size)
            obj, alpha = _augment_object(source, (image_size, image_size))
            synthetic, bbox = _paste_object(background, obj, alpha)

            stem = f"{folder.name}_{idx:04d}"
            image_path = output_dir / "images" / split / f"{stem}.jpg"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(image_path), synthetic, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            _write_yolo_label(label_path, class_id, bbox, image_size, image_size)

    yaml_path = output_dir / "dataset.yaml"
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {idx: name for idx, name in enumerate(class_names)},
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic YOLO dataset from one image per class.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DATASET_DIR)
    parser.add_argument("--samples-per-class", type=int, default=200)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    yaml_path = generate_dataset(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Generated dataset config: {yaml_path}")


if __name__ == "__main__":
    main()
