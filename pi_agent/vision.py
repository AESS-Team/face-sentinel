from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .config import AgentConfig


class VisionSetupError(RuntimeError):
    pass


def largest_face(faces: Iterable[Any] | None) -> list[float] | None:
    if faces is None:
        return None

    face_list = [list(face) for face in faces]
    if not face_list:
        return None
    return max(face_list, key=lambda face: float(face[2]) * float(face[3]))


def _extract_faces(detection_result: Any) -> Any:
    if isinstance(detection_result, tuple):
        return detection_result[1]
    return detection_result


def iter_image_paths(root: Path) -> Iterable[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


class FaceEngine:
    def __init__(self, config: AgentConfig):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise VisionSetupError("OpenCV is not installed. Install python3-opencv on the Pi.") from exc

        self.cv2 = cv2
        self.config = config
        self._require_file(config.detector_model_path)
        self._require_file(config.recognition_model_path)
        self.detector = cv2.FaceDetectorYN.create(
            str(config.detector_model_path),
            "",
            (config.camera_width, config.camera_height),
            config.detector_score_threshold,
            config.nms_threshold,
            config.top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.recognition_model_path), "")

    def _require_file(self, path: Path) -> None:
        if not path.exists():
            raise VisionSetupError(f"Missing model file: {path}. Run python -m pi_agent.download_models")

    def detect_faces(self, frame: Any) -> list[list[float]]:
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        faces = _extract_faces(self.detector.detect(frame))
        if faces is None:
            return []
        return [list(face) for face in faces]

    def embedding_for_face(self, frame: Any, face: list[float]) -> list[float]:
        aligned = self.recognizer.alignCrop(frame, face)
        feature = self.recognizer.feature(aligned)
        return [float(value) for value in feature.flatten()]

    def embedding_for_image(self, image_path: Path) -> tuple[list[float], list[float]]:
        image = self.cv2.imread(str(image_path))
        if image is None:
            raise VisionSetupError(f"Could not read image: {image_path}")

        face = largest_face(self.detect_faces(image))
        if face is None:
            raise VisionSetupError(f"No face detected in image: {image_path}")

        return self.embedding_for_face(image, face), face

    def jpeg_bytes(self, frame: Any) -> bytes:
        ok, encoded = self.cv2.imencode(".jpg", frame)
        if not ok:
            raise VisionSetupError("Could not encode frame as JPEG")
        return encoded.tobytes()
