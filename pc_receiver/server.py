from __future__ import annotations

import argparse
import base64
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def save_received_event(received_dir: Path, payload: dict[str, Any]) -> dict[str, str]:
    received_dir.mkdir(parents=True, exist_ok=True)

    event_id = str(payload["event_id"])
    image_bytes = base64.b64decode(payload["image_base64"])
    image_path = received_dir / f"{event_id}.jpg"
    metadata_path = received_dir / f"{event_id}.json"

    metadata = {key: value for key, value in payload.items() if key != "image_base64"}
    metadata["image_file"] = image_path.name

    image_path.write_bytes(image_bytes)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
    }


class EventRequestHandler(BaseHTTPRequestHandler):
    received_dir = Path("received")

    def do_POST(self) -> None:
        if self.path != "/event":
            self.send_error(HTTPStatus.NOT_FOUND, "Use POST /event")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            payload = json.loads(raw_body.decode("utf-8"))
            saved = save_received_event(self.received_dir, payload)
        except Exception as exc:  # noqa: BLE001 - keep server response explicit.
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        response = json.dumps({"ok": True, **saved}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


def run_server(host: str, port: int, received_dir: Path) -> None:
    handler = type(
        "ConfiguredEventRequestHandler",
        (EventRequestHandler,),
        {"received_dir": received_dir},
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Listening on http://{host}:{port}/event")
    print(f"Saving received events to {received_dir.resolve()}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive Face Sentinel events from the Raspberry Pi.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--received-dir", default=str(Path(__file__).with_name("received")))
    args = parser.parse_args()

    run_server(args.host, args.port, Path(args.received_dir))


if __name__ == "__main__":
    main()
