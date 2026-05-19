from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .config import AgentConfig, load_config
from .events import build_event_payload, build_event_record, post_event, save_unknown_event
from .face_db import load_embeddings
from .gatekeeper import build_gatekeeper_session
from .presence import HumanPresenceDetector, PresenceAnalysis
from .recognition import MatchDecision, best_identity_match
from .vision import FaceEngine, VisionSetupError


class PiCamera2Capture:
    def __init__(self, config: AgentConfig):
        try:
            import cv2  # type: ignore
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:
            raise VisionSetupError("Picamera2 is not available") from exc

        self.cv2 = cv2
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (config.camera_width, config.camera_height)}
        )
        self.picam2.configure(camera_config)
        self.picam2.start()
        time.sleep(1.0)

    def read(self) -> Any:
        frame_rgb = self.picam2.capture_array()
        return self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        self.picam2.stop()


class OpenCVCapture:
    def __init__(self, config: AgentConfig):
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise VisionSetupError("OpenCV is not installed") from exc

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
        if not self.capture.isOpened():
            raise VisionSetupError("Could not open OpenCV camera index 0")

    def read(self) -> Any:
        ok, frame = self.capture.read()
        if not ok:
            raise VisionSetupError("Could not read a camera frame")
        return frame

    def close(self) -> None:
        self.capture.release()


def open_camera(config: AgentConfig, camera_backend: str) -> PiCamera2Capture | OpenCVCapture:
    if camera_backend == "picamera2":
        return PiCamera2Capture(config)
    if camera_backend == "opencv":
        return OpenCVCapture(config)

    try:
        return PiCamera2Capture(config)
    except VisionSetupError as exc:
        print(f"Picamera2 unavailable, trying OpenCV camera: {exc}")
        return OpenCVCapture(config)


def describe_decision(decision: MatchDecision) -> str:
    if decision.is_known:
        return f"known identity={decision.identity} score={decision.score:.3f}"
    return f"unknown score={decision.score:.3f}"


@dataclass(frozen=True)
class FaceAnalysis:
    face: list[float]
    decision: MatchDecision


@dataclass(frozen=True)
class FrameAnalysis:
    faces: list[FaceAnalysis]

    @property
    def status(self) -> str:
        if not self.faces:
            return "no_face"
        if self.primary_face is not None and self.primary_face.decision.is_known:
            return "known"
        return "unknown"

    @property
    def primary_face(self) -> FaceAnalysis | None:
        if not self.faces:
            return None
        known_faces = [face for face in self.faces if face.decision.is_known]
        candidates = known_faces or self.faces
        return max(candidates, key=lambda face: face.decision.score)

    @property
    def identity(self) -> str | None:
        primary = self.primary_face
        if primary is not None and primary.decision.is_known:
            return primary.decision.identity
        return None

    @property
    def score(self) -> float:
        primary = self.primary_face
        if primary is None:
            return 0.0
        return primary.decision.score

    @property
    def bbox(self) -> list[float] | None:
        primary = self.primary_face
        if primary is None:
            return None
        return [round(float(value), 2) for value in primary.face[:4]]

    def first_unknown_face(self) -> FaceAnalysis | None:
        for face in self.faces:
            if not face.decision.is_known:
                return face
        return None


class UnknownGatekeeper(Protocol):
    def run(self) -> bool:
        raise NotImplementedError


class PresenceDetector(Protocol):
    def analyze(self, frame: Any) -> PresenceAnalysis:
        raise NotImplementedError


def analyze_frame(
    config: AgentConfig,
    engine: FaceEngine,
    frame: Any,
    known_embeddings: dict[str, list[list[float]]],
) -> FrameAnalysis:
    faces = engine.detect_faces(frame)
    analyzed_faces = []
    for face in faces:
        embedding = engine.embedding_for_face(frame, face)
        decision = best_identity_match(embedding, known_embeddings, config.match_threshold)
        analyzed_faces.append(FaceAnalysis(face, decision))
    return FrameAnalysis(analyzed_faces)


def monitor_metadata(analysis: FrameAnalysis, source: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "event_type": "monitor",
        "status": analysis.status,
        "source": source,
        "face_count": len(analysis.faces),
        "score": round(float(analysis.score), 6),
    }
    if analysis.identity is not None:
        metadata["identity"] = analysis.identity
    if analysis.bbox is not None:
        metadata["bbox"] = analysis.bbox
    return build_event_record(metadata)


def emit_monitor_event(
    config: AgentConfig,
    engine: FaceEngine,
    frame: Any,
    analysis: FrameAnalysis,
    dry_run: bool,
) -> None:
    image_bytes = engine.jpeg_bytes(frame)
    metadata = monitor_metadata(analysis, socket.gethostname())
    label = metadata.get("identity") or metadata["status"]
    print(f"monitor {label} status={metadata['status']} score={metadata['score']:.3f}")

    if dry_run:
        print("Dry run enabled; not sending monitor HTTP frame")
        return

    payload = build_event_payload(metadata, image_bytes)
    response = post_event(config.receiver_url, payload)
    print(f"Sent monitor frame: {response}")


def emit_unknown_event(
    config: AgentConfig,
    engine: FaceEngine,
    frame: Any,
    face_analysis: FaceAnalysis,
    dry_run: bool,
) -> None:
    image_bytes = engine.jpeg_bytes(frame)
    metadata = {
        "source": socket.gethostname(),
        "score": round(float(face_analysis.decision.score), 6),
        "bbox": [round(float(value), 2) for value in face_analysis.face[:4]],
        "receiver_url": config.receiver_url,
    }
    event = save_unknown_event(config.events_dir, image_bytes, metadata)
    print(f"Saved unknown event: {event['metadata_path']}")

    if dry_run:
        print("Dry run enabled; not sending HTTP event")
        return

    payload = build_event_payload(event, image_bytes)
    response = post_event(config.receiver_url, payload)
    print(f"Sent HTTP event: {response}")


def process_frame(
    config: AgentConfig,
    engine: FaceEngine,
    frame: Any,
    known_embeddings: dict[str, list[list[float]]],
    dry_run: bool,
    can_emit_unknown: bool,
) -> tuple[bool, FrameAnalysis]:
    analysis = analyze_frame(config, engine, frame, known_embeddings)
    if not analysis.faces:
        print("No faces")
        return False, analysis

    emitted = False
    for face_analysis in analysis.faces:
        print(describe_decision(face_analysis.decision))

        if not face_analysis.decision.is_known and can_emit_unknown and not emitted:
            emit_unknown_event(config, engine, frame, face_analysis, dry_run)
            emitted = True

    return emitted, analysis


def process_presence_gatekeeper(
    detector: PresenceDetector,
    frame: Any,
    gatekeeper_session: UnknownGatekeeper | None,
    can_run_gatekeeper: bool,
) -> tuple[bool, bool, PresenceAnalysis]:
    analysis = detector.analyze(frame)
    primary = analysis.primary_detection
    if primary is None:
        print("No person present")
        return False, False, analysis

    print(f"person_present detector={primary.label} score={primary.score:.3f}")
    if gatekeeper_session is None or not can_run_gatekeeper:
        return False, False, analysis

    return True, gatekeeper_session.run(), analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Face Sentinel on the Raspberry Pi.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to pi_agent/config.json",
    )
    parser.add_argument("--camera", choices=["auto", "picamera2", "opencv"], default="auto")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between frames")
    parser.add_argument("--once", action="store_true", help="Process one frame and exit")
    parser.add_argument("--dry-run", action="store_true", help="Save locally but do not POST to the PC")
    parser.add_argument("--monitor", action="store_true", help="Send labelled debug frames to the PC")
    parser.add_argument("--monitor-fps", type=float, default=2.0, help="Monitor frames per second")
    args = parser.parse_args()
    if args.monitor_fps <= 0:
        raise SystemExit("--monitor-fps must be greater than 0")

    config = load_config(Path(args.config))
    known_embeddings = load_embeddings(config.embeddings_path)
    if not known_embeddings and not (config.gatekeeper_enabled and config.presence_enabled):
        raise SystemExit(f"No enrolled faces found at {config.embeddings_path}. Run python -m pi_agent.enroll first.")
    if not known_embeddings:
        print(f"No enrolled faces found at {config.embeddings_path}; continuing with presence gatekeeper only.")

    engine = FaceEngine(config)
    camera = open_camera(config, args.camera)
    gatekeeper_session = build_gatekeeper_session(config) if config.gatekeeper_enabled else None
    presence_detector = HumanPresenceDetector(config) if config.presence_enabled and gatekeeper_session is not None else None
    last_unknown_at = 0.0
    last_monitor_at = 0.0
    last_gatekeeper_attempt_at = 0.0
    last_gatekeeper_passed_at = 0.0
    monitor_interval = 1.0 / args.monitor_fps

    try:
        while True:
            now = time.monotonic()
            can_emit_unknown = bool(known_embeddings) and now - last_unknown_at >= config.unknown_cooldown_seconds
            can_run_gatekeeper = (
                now - last_gatekeeper_attempt_at >= config.presence_cooldown_seconds
                and now - last_gatekeeper_passed_at >= config.gatekeeper_silence_after_pass_seconds
            )
            frame = camera.read()
            if presence_detector is not None:
                gatekeeper_ran, gatekeeper_passed, _presence_analysis = process_presence_gatekeeper(
                    presence_detector,
                    frame,
                    gatekeeper_session,
                    can_run_gatekeeper,
                )
                if gatekeeper_ran:
                    last_gatekeeper_attempt_at = now
                if gatekeeper_passed:
                    last_gatekeeper_passed_at = now

            emitted, analysis = process_frame(config, engine, frame, known_embeddings, args.dry_run, can_emit_unknown)
            if emitted:
                last_unknown_at = now
            if args.monitor and now - last_monitor_at >= monitor_interval:
                emit_monitor_event(config, engine, frame, analysis, args.dry_run)
                last_monitor_at = now
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
