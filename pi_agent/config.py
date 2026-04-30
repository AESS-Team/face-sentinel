from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    receiver_url: str
    match_threshold: float
    camera_width: int
    camera_height: int
    detector_score_threshold: float
    nms_threshold: float
    top_k: int
    unknown_cooldown_seconds: float
    known_faces_dir: Path
    events_dir: Path
    models_dir: Path
    embeddings_path: Path
    detector_model_path: Path
    recognition_model_path: Path


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def load_config(config_path: Path) -> AgentConfig:
    raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    base_dir = config_path.parent
    camera = raw.get("camera", {})
    paths = raw.get("paths", {})
    detector = raw.get("detector", {})

    known_faces_dir = _resolve_path(base_dir, paths.get("known_faces_dir", "known_faces"))
    events_dir = _resolve_path(base_dir, paths.get("events_dir", "events"))
    models_dir = _resolve_path(base_dir, paths.get("models_dir", "models"))

    return AgentConfig(
        receiver_url=raw.get("receiver_url", "http://127.0.0.1:8765/event"),
        match_threshold=float(raw.get("match_threshold", 0.55)),
        camera_width=int(camera.get("width", 640)),
        camera_height=int(camera.get("height", 480)),
        detector_score_threshold=float(detector.get("score_threshold", 0.9)),
        nms_threshold=float(detector.get("nms_threshold", 0.3)),
        top_k=int(detector.get("top_k", 5000)),
        unknown_cooldown_seconds=float(raw.get("unknown_cooldown_seconds", 10.0)),
        known_faces_dir=known_faces_dir,
        events_dir=events_dir,
        models_dir=models_dir,
        embeddings_path=known_faces_dir / "embeddings.json",
        detector_model_path=models_dir / "face_detection_yunet_2023mar.onnx",
        recognition_model_path=models_dir / "face_recognition_sface_2021dec.onnx",
    )
