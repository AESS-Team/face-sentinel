import unittest

from pi_agent.run import analyze_frame, monitor_metadata


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


if __name__ == "__main__":
    unittest.main()
