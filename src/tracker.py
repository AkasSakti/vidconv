from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass
class Track:
    track_id: int
    centroid: tuple[int, int]
    class_name: str
    confidence: float
    missed: int = 0
    counted: bool = False
    history: list[tuple[int, int]] = field(default_factory=list)

    def update(self, centroid: tuple[int, int], class_name: str, confidence: float) -> None:
        self.history.append(self.centroid)
        self.centroid = centroid
        self.class_name = class_name
        self.confidence = confidence
        self.missed = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(self, detections: list[dict]) -> list[Track]:
        """Update tracks from detections containing bbox, class_name, and confidence."""
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_detection_ids = set(range(len(detections)))
        pairs: list[tuple[float, int, int]] = []

        for track_id, track in self.tracks.items():
            for det_id, detection in enumerate(detections):
                cx, cy = detection["centroid"]
                distance = hypot(track.centroid[0] - cx, track.centroid[1] - cy)
                if distance <= self.max_distance:
                    pairs.append((distance, track_id, det_id))

        for _, track_id, det_id in sorted(pairs, key=lambda item: item[0]):
            if track_id not in unmatched_track_ids or det_id not in unmatched_detection_ids:
                continue
            detection = detections[det_id]
            self.tracks[track_id].update(
                detection["centroid"],
                detection["class_name"],
                detection["confidence"],
            )
            unmatched_track_ids.remove(track_id)
            unmatched_detection_ids.remove(det_id)

        for track_id in list(unmatched_track_ids):
            self.tracks[track_id].missed += 1
            if self.tracks[track_id].missed > self.max_missed:
                del self.tracks[track_id]

        for det_id in unmatched_detection_ids:
            detection = detections[det_id]
            self.tracks[self.next_id] = Track(
                track_id=self.next_id,
                centroid=detection["centroid"],
                class_name=detection["class_name"],
                confidence=detection["confidence"],
            )
            self.next_id += 1

        return list(self.tracks.values())


class LineCounter:
    def __init__(self, class_names: list[str]) -> None:
        self.counts = {class_name: 0 for class_name in class_names}
        self.unknown_count = 0

    def reset(self) -> None:
        for key in self.counts:
            self.counts[key] = 0
        self.unknown_count = 0

    def apply(self, tracks: list[Track], line_y: int) -> None:
        for track in tracks:
            if track.counted or not track.history:
                continue
            previous_y = track.history[-1][1]
            current_y = track.centroid[1]
            crossed = (previous_y < line_y <= current_y) or (previous_y > line_y >= current_y)
            if not crossed:
                continue
            if track.class_name == "unknown":
                self.unknown_count += 1
            elif track.class_name in self.counts:
                self.counts[track.class_name] += 1
            track.counted = True
