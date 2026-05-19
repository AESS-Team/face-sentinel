from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import AgentConfig
from ..vision import VisionSetupError


@dataclass(frozen=True)
class PresenceDetection:
    label: str
    bbox: list[float]
    score: float


@dataclass(frozen=True)
class PresenceAnalysis:
    detections: list[PresenceDetection]

    @property
    def present(self) -> bool:
        return bool(self.detections)

    @property
    def status(self) -> str:
        if self.present:
            return "person_present"
        return "no_person"

    @property
    def primary_detection(self) -> PresenceDetection | None:
        if not self.detections:
            return None
        return max(self.detections, key=lambda detection: detection.score)


def boxes_to_detections(label: str, boxes: Iterable[Any], score: float) -> list[PresenceDetection]:
    detections = []
    for box in boxes:
        x, y, width, height = box
        detections.append(
            PresenceDetection(
                label=label,
                bbox=[round(float(x), 2), round(float(y), 2), round(float(width), 2), round(float(height), 2)],
                score=float(score),
            )
        )
    return detections


def _cascade_path(cv2_module: Any, filename: str) -> Path:
    return Path(cv2_module.data.haarcascades) / filename


def _load_cascade(cv2_module: Any, filename: str) -> Any | None:
    path = _cascade_path(cv2_module, filename)
    cascade = cv2_module.CascadeClassifier(str(path))
    if cascade.empty():
        return None
    return cascade


class HumanPresenceDetector:
    def __init__(self, config: AgentConfig):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise VisionSetupError("OpenCV is not installed") from exc

        self.cv2 = cv2
        self.config = config
        self.cascades = [
            ("face", _load_cascade(cv2, "haarcascade_frontalface_default.xml")),
            ("profile_face", _load_cascade(cv2, "haarcascade_profileface.xml")),
            ("upper_body", _load_cascade(cv2, "haarcascade_upperbody.xml")),
            ("full_body", _load_cascade(cv2, "haarcascade_fullbody.xml")),
        ]
        self.cascades = [(label, cascade) for label, cascade in self.cascades if cascade is not None]

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def analyze(self, frame: Any) -> PresenceAnalysis:
        gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
        detections: list[PresenceDetection] = []

        for label, cascade in self.cascades:
            boxes = cascade.detectMultiScale(
                gray,
                scaleFactor=self.config.presence_haar_scale_factor,
                minNeighbors=self.config.presence_haar_min_neighbors,
                minSize=(30, 30),
            )
            detections.extend(boxes_to_detections(label, boxes, 0.8))

        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        for box, weight in zip(boxes, weights):
            detections.extend(boxes_to_detections("person_hog", [box], max(0.0, min(float(weight), 1.0))))

        return PresenceAnalysis(detections)
