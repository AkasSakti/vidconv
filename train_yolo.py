from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from src.config import GENERATED_DATASET_DIR, RUNS_DIR


def train(
    data_yaml: Path = GENERATED_DATASET_DIR / "dataset.yaml",
    epochs: int = 80,
    image_size: int = 640,
    batch: int = 8,
    model: str = "yolov8n.pt",
) -> Path:
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run: python -m src.dataset_generator --samples-per-class 200"
        )

    yolo = YOLO(model)
    result = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        project=str(RUNS_DIR),
        name="beverage_yolov8n",
        exist_ok=True,
    )
    return Path(result.save_dir) / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8n on the generated beverage dataset.")
    parser.add_argument("--data", type=Path, default=GENERATED_DATASET_DIR / "dataset.yaml")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--model", default="yolov8n.pt")
    args = parser.parse_args()

    best_model = train(args.data, args.epochs, args.imgsz, args.batch, args.model)
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
