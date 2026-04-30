import json
import tempfile
import unittest
from pathlib import Path

from pi_agent.config import load_config


class ConfigTests(unittest.TestCase):
    def test_load_config_resolves_relative_paths_from_config_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "receiver_url": "http://192.168.1.10:8765/event",
                        "match_threshold": 0.5,
                        "paths": {
                            "known_faces_dir": "faces",
                            "events_dir": "out",
                            "models_dir": "model-files",
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.receiver_url, "http://192.168.1.10:8765/event")
            self.assertEqual(config.match_threshold, 0.5)
            self.assertEqual(config.known_faces_dir, Path(temp_dir) / "faces")
            self.assertEqual(config.events_dir, Path(temp_dir) / "out")
            self.assertEqual(config.models_dir, Path(temp_dir) / "model-files")

    def test_load_config_uses_pi_friendly_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            config = load_config(config_path)

            self.assertEqual(config.camera_width, 640)
            self.assertEqual(config.camera_height, 480)
            self.assertEqual(config.unknown_cooldown_seconds, 10.0)


if __name__ == "__main__":
    unittest.main()
