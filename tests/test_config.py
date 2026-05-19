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
            self.assertFalse(config.gatekeeper_enabled)
            self.assertEqual(config.gatekeeper_gemini_model, "gemini-3.1-flash-lite")
            self.assertEqual(config.gatekeeper_tts_model, "gemini-3.1-flash-tts-preview")
            self.assertTrue(config.presence_enabled)
            self.assertEqual(config.presence_cooldown_seconds, 10.0)

    def test_load_config_reads_gatekeeper_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "gatekeeper": {
                            "enabled": True,
                            "questions_to_pass": 3,
                            "max_rounds": 5,
                            "record_seconds": 4.5,
                            "silence_after_pass_seconds": 120,
                            "audio_dir": "voice-cache",
                            "voice_name": "Kore",
                            "gemini_model": "gemini-3-flash-preview",
                            "tts_model": "gemini-3.1-flash-tts-preview",
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertTrue(config.gatekeeper_enabled)
            self.assertEqual(config.gatekeeper_questions_to_pass, 3)
            self.assertEqual(config.gatekeeper_max_rounds, 5)
            self.assertEqual(config.gatekeeper_record_seconds, 4.5)
            self.assertEqual(config.gatekeeper_silence_after_pass_seconds, 120)
            self.assertEqual(config.gatekeeper_audio_dir, Path(temp_dir) / "voice-cache")
            self.assertEqual(config.gatekeeper_voice_name, "Kore")
            self.assertEqual(config.gatekeeper_gemini_model, "gemini-3-flash-preview")
            self.assertEqual(config.gatekeeper_tts_model, "gemini-3.1-flash-tts-preview")

    def test_load_config_reads_presence_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "presence": {
                            "enabled": False,
                            "cooldown_seconds": 4.5,
                            "haar_scale_factor": 1.2,
                            "haar_min_neighbors": 6,
                        }
                    }
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertFalse(config.presence_enabled)
            self.assertEqual(config.presence_cooldown_seconds, 4.5)
            self.assertEqual(config.presence_haar_scale_factor, 1.2)
            self.assertEqual(config.presence_haar_min_neighbors, 6)


if __name__ == "__main__":
    unittest.main()
