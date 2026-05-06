from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class ProductDetector:
    def __init__(self, model_path: str | Path, class_names: list[str]) -> None:
        self.model = YOLO(str(model_path))
        self.class_names = class_names

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = 0.5,
        min_area: int = 1_500,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 4.0,
    ) -> tuple[list[dict], list[dict]]:
        results = self.model.predict(frame, conf=max(0.01, confidence_threshold * 0.5), verbose=False)
        detections: list[dict] = []
        unknowns: list[dict] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                area = width * height
                aspect_ratio = width / height
                reason = None

                if conf < confidence_threshold:
                    reason = "confidence_below_threshold"
                elif area < min_area:
                    reason = "bbox_area_too_small"
                elif not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
                    reason = "invalid_aspect_ratio"

                class_name = self.class_names[cls_id] if 0 <= cls_id < len(self.class_names) else "unknown"
                item = {
                    "bbox": (x1, y1, x2, y2),
                    "centroid": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "confidence": conf,
                    "class_name": class_name if reason is None else "unknown",
                    "reason": reason,
                    "area": area,
                    "aspect_ratio": aspect_ratio,
                }
                if reason is None:
                    detections.append(item)
                else:
                    unknowns.append(item)
                    detections.append(item)

        return detections, unknowns


def draw_overlay(frame: np.ndarray, detections: list[dict], tracks, line_y: int) -> np.ndarray:
    output = frame.copy()
    cv2.line(output, (0, line_y), (output.shape[1], line_y), (0, 255, 255), 2)

    track_by_centroid = {track.centroid: track for track in tracks}
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        color = (0, 0, 255) if detection["class_name"] == "unknown" else (0, 180, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        track = track_by_centroid.get(detection["centroid"])
        track_text = f"ID {track.track_id} " if track else ""
        label = f"{track_text}{detection['class_name']} {detection['confidence']:.2f}"
        cv2.putText(output, label, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.circle(output, detection["centroid"], 4, color, -1)

    return output
