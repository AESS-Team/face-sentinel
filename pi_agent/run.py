from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Any

from .config import AgentConfig, load_config
from .events import build_event_payload, post_event, save_unknown_event
from .face_db import load_embeddings
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


def emit_unknown_event(
    config: AgentConfig,
    engine: FaceEngine,
    frame: Any,
    face: list[float],
    decision: MatchDecision,
    dry_run: bool,
) -> None:
    image_bytes = engine.jpeg_bytes(frame)
    metadata = {
        "source": socket.gethostname(),
        "score": round(float(decision.score), 6),
        "bbox": [round(float(value), 2) for value in face[:4]],
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
) -> bool:
    faces = engine.detect_faces(frame)
    if not faces:
        print("No faces")
        return False

    emitted = False
    for face in faces:
        embedding = engine.embedding_for_face(frame, face)
        decision = best_identity_match(embedding, known_embeddings, config.match_threshold)
        print(describe_decision(decision))

        if not decision.is_known and can_emit_unknown and not emitted:
            emit_unknown_event(config, engine, frame, face, decision, dry_run)
            emitted = True

    return emitted


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
    args = parser.parse_args()

    config = load_config(Path(args.config))
    known_embeddings = load_embeddings(config.embeddings_path)
    if not known_embeddings:
        raise SystemExit(f"No enrolled faces found at {config.embeddings_path}. Run python -m pi_agent.enroll first.")

    engine = FaceEngine(config)
    camera = open_camera(config, args.camera)
    last_unknown_at = 0.0

    try:
        while True:
            now = time.monotonic()
            can_emit_unknown = now - last_unknown_at >= config.unknown_cooldown_seconds
            frame = camera.read()
            emitted = process_frame(config, engine, frame, known_embeddings, args.dry_run, can_emit_unknown)
            if emitted:
                last_unknown_at = now
            if args.once:
                break
            time.sleep(args.interval)
    finally:
        camera.close()


if __name__ == "__main__":
    main()
