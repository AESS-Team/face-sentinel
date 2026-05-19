import unittest
import tempfile
from pathlib import Path

from pi_agent.presence import PresenceAnalysis, PresenceDetection
from pi_agent.run import analyze_frame, monitor_metadata, process_frame, process_presence_gatekeeper


class FakeConfig:
    match_threshold = 0.85


class FakeEngine:
    def __init__(self, faces, embeddings):
        self.faces = faces
        self.embeddings = embeddings

    def detect_faces(self, frame):
        return self.faces

    def embedding_for_face(self, frame, face):
        return self.embeddings[tuple(face)]

    def jpeg_bytes(self, frame):
        return b"fake-jpeg"


class FakeGatekeeperSession:
    def __init__(self, passed):
        self.passed = passed
        self.run_count = 0

    def run(self):
        self.run_count += 1
        return self.passed


class FakePresenceDetector:
    def __init__(self, present):
        self.present = present

    def analyze(self, frame):
        if not self.present:
            return PresenceAnalysis([])
        return PresenceAnalysis([PresenceDetection("face", [1, 2, 3, 4], 0.9)])


class RunMonitorTests(unittest.TestCase):
    def test_analyze_frame_labels_no_face(self):
        analysis = analyze_frame(FakeConfig(), FakeEngine([], {}), "frame", {})

        self.assertEqual(analysis.status, "no_face")
        self.assertEqual(analysis.identity, None)
        self.assertEqual(analysis.score, 0.0)

    def test_analyze_frame_labels_known_face(self):
        face = [1, 2, 3, 4, 0.9]
        engine = FakeEngine([face], {tuple(face): [1.0, 0.0]})
        known = {"guillem": [[1.0, 0.0]]}

        analysis = analyze_frame(FakeConfig(), engine, "frame", known)

        self.assertEqual(analysis.status, "known")
        self.assertEqual(analysis.identity, "guillem")
        self.assertGreaterEqual(analysis.score, 0.85)

    def test_analyze_frame_labels_unknown_face(self):
        face = [1, 2, 3, 4, 0.9]
        engine = FakeEngine([face], {tuple(face): [0.0, 1.0]})
        known = {"guillem": [[1.0, 0.0]]}

        analysis = analyze_frame(FakeConfig(), engine, "frame", known)

        self.assertEqual(analysis.status, "unknown")
        self.assertEqual(analysis.identity, None)
        self.assertLess(analysis.score, 0.85)

    def test_monitor_metadata_marks_monitor_event(self):
        analysis = analyze_frame(FakeConfig(), FakeEngine([], {}), "frame", {})

        metadata = monitor_metadata(analysis, "aess")

        self.assertEqual(metadata["event_type"], "monitor")
        self.assertEqual(metadata["status"], "no_face")
        self.assertEqual(metadata["source"], "aess")

    def test_process_frame_emits_unknown_face_without_gatekeeper(self):
        face = [1, 2, 3, 4, 0.9]
        engine = FakeEngine([face], {tuple(face): [0.0, 1.0]})
        gatekeeper = FakeGatekeeperSession(passed=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            config = FakeConfig()
            config.events_dir = Path(temp_dir)
            config.receiver_url = "http://127.0.0.1:8765/event"

            emitted, analysis = process_frame(
                config,
                engine,
                "frame",
                {"guillem": [[1.0, 0.0]]},
                dry_run=True,
                can_emit_unknown=True,
            )

        self.assertTrue(emitted)
        self.assertEqual(analysis.status, "unknown")
        self.assertEqual(gatekeeper.run_count, 0)

    def test_process_presence_gatekeeper_runs_for_any_present_person(self):
        gatekeeper = FakeGatekeeperSession(passed=True)

        ran, gatekeeper_passed, analysis = process_presence_gatekeeper(
            FakePresenceDetector(present=True),
            "frame",
            gatekeeper,
            can_run_gatekeeper=True,
        )

        self.assertTrue(ran)
        self.assertTrue(gatekeeper_passed)
        self.assertTrue(analysis.present)
        self.assertEqual(gatekeeper.run_count, 1)


if __name__ == "__main__":
    unittest.main()
