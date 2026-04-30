import base64
import json
import tempfile
import unittest
from pathlib import Path

from pi_agent.events import build_event_payload, save_unknown_event


class EventTests(unittest.TestCase):
    def test_save_unknown_event_writes_metadata_and_image(self):
        image_bytes = b"fake-jpeg"

        with tempfile.TemporaryDirectory() as temp_dir:
            event = save_unknown_event(
                Path(temp_dir),
                image_bytes,
                {
                    "source": "pi",
                    "score": 0.42,
                    "bbox": [1, 2, 3, 4],
                },
            )

            image_path = Path(event["image_path"])
            metadata_path = Path(event["metadata_path"])
            self.assertTrue(image_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(image_path.read_bytes(), image_bytes)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["identity"], "unknown")
            self.assertEqual(metadata["source"], "pi")
            self.assertEqual(metadata["score"], 0.42)

    def test_build_event_payload_includes_base64_image(self):
        payload = build_event_payload(
            {
                "event_id": "abc123",
                "timestamp": "2026-04-30T12:00:00Z",
                "identity": "unknown",
                "score": 0.1,
            },
            b"fake-jpeg",
        )

        self.assertEqual(payload["event_id"], "abc123")
        self.assertEqual(base64.b64decode(payload["image_base64"]), b"fake-jpeg")


if __name__ == "__main__":
    unittest.main()
