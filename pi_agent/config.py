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
    gatekeeper_enabled: bool
    gatekeeper_questions_to_pass: int
    gatekeeper_max_rounds: int
    gatekeeper_record_seconds: float
    gatekeeper_silence_after_pass_seconds: float
    gatekeeper_audio_dir: Path
    gatekeeper_voice_name: str
    gatekeeper_gemini_model: str
    gatekeeper_tts_model: str
    presence_enabled: bool
    presence_cooldown_seconds: float
    presence_haar_scale_factor: float
    presence_haar_min_neighbors: int


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
    gatekeeper = raw.get("gatekeeper", {})
    presence = raw.get("presence", {})

    known_faces_dir = _resolve_path(base_dir, paths.get("known_faces_dir", "known_faces"))
    events_dir = _resolve_path(base_dir, paths.get("events_dir", "events"))
    models_dir = _resolve_path(base_dir, paths.get("models_dir", "models"))
    gatekeeper_audio_dir = _resolve_path(base_dir, gatekeeper.get("audio_dir", "gatekeeper_audio"))

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
        gatekeeper_enabled=bool(gatekeeper.get("enabled", False)),
        gatekeeper_questions_to_pass=int(gatekeeper.get("questions_to_pass", 2)),
        gatekeeper_max_rounds=int(gatekeeper.get("max_rounds", 4)),
        gatekeeper_record_seconds=float(gatekeeper.get("record_seconds", 5.0)),
        gatekeeper_silence_after_pass_seconds=float(gatekeeper.get("silence_after_pass_seconds", 60.0)),
        gatekeeper_audio_dir=gatekeeper_audio_dir,
        gatekeeper_voice_name=gatekeeper.get("voice_name", "Kore"),
        gatekeeper_gemini_model=gatekeeper.get("gemini_model", "gemini-3.1-flash-lite"),
        gatekeeper_tts_model=gatekeeper.get("tts_model", "gemini-3.1-flash-tts-preview"),
        presence_enabled=bool(presence.get("enabled", True)),
        presence_cooldown_seconds=float(presence.get("cooldown_seconds", 10.0)),
        presence_haar_scale_factor=float(presence.get("haar_scale_factor", 1.1)),
        presence_haar_min_neighbors=int(presence.get("haar_min_neighbors", 4)),
    )
