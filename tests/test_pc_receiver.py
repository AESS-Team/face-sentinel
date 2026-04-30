import json
import tempfile
import unittest
from pathlib import Path

from pc_receiver.server import save_received_event


class PcReceiverTests(unittest.TestCase):
    def test_save_received_event_stores_json_and_image(self):
        payload = {
            "event_id": "event-1",
            "timestamp": "2026-04-30T12:00:00Z",
            "identity": "unknown",
            "score": 0.2,
            "image_base64": "ZmFrZS1qcGVn",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved = save_received_event(Path(temp_dir), payload)

            image_path = Path(saved["image_path"])
            metadata_path = Path(saved["metadata_path"])
            self.assertTrue(image_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertEqual(image_path.read_bytes(), b"fake-jpeg")

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertNotIn("image_base64", metadata)
            self.assertEqual(metadata["event_id"], "event-1")

    def test_save_received_monitor_event_uses_monitor_folder(self):
        payload = {
            "event_id": "monitor-1",
            "timestamp": "2026-04-30T12:00:00Z",
            "event_type": "monitor",
            "status": "no_face",
            "image_base64": "ZmFrZS1qcGVn",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            saved = save_received_event(Path(temp_dir), payload)

            image_path = Path(saved["image_path"])
            metadata_path = Path(saved["metadata_path"])
            self.assertEqual(image_path.parent.name, "monitor")
            self.assertEqual(metadata_path.parent.name, "monitor")
            self.assertTrue(image_path.exists())


if __name__ == "__main__":
    unittest.main()
