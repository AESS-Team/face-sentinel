from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _event_id(timestamp: str) -> str:
    safe_timestamp = timestamp.replace(":", "").replace("+", "Z")
    return f"{safe_timestamp}-{uuid.uuid4().hex[:8]}"


def save_unknown_event(
    events_dir: Path,
    image_bytes: bytes,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    events_dir.mkdir(parents=True, exist_ok=True)

    timestamp = str(metadata.get("timestamp") or _utc_timestamp())
    event_id = str(metadata.get("event_id") or _event_id(timestamp))
    image_path = events_dir / f"{event_id}.jpg"
    metadata_path = events_dir / f"{event_id}.json"

    record = {
        "event_id": event_id,
        "timestamp": timestamp,
        "identity": "unknown",
        **metadata,
    }
    record["identity"] = "unknown"
    record["image_file"] = image_path.name

    image_path.write_bytes(image_bytes)
    metadata_path.write_text(
        json.dumps(record, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return {
        **record,
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
    }


def build_event_payload(event: dict[str, Any], image_bytes: bytes) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in event.items()
        if key not in {"image_path", "metadata_path"}
    }
    payload["image_base64"] = base64.b64encode(image_bytes).decode("ascii")
    return payload


def post_event(receiver_url: str, payload: dict[str, Any], timeout: float = 3.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        receiver_url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(http_request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        if not body:
            return {"status": response.status}
        return json.loads(body)
