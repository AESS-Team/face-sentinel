import unittest

from pi_agent.presence import PresenceAnalysis, PresenceDetection, boxes_to_detections


class PresenceTests(unittest.TestCase):
    def test_presence_analysis_reports_no_person(self):
        analysis = PresenceAnalysis([])

        self.assertFalse(analysis.present)
        self.assertEqual(analysis.status, "no_person")
        self.assertIsNone(analysis.primary_detection)

    def test_presence_analysis_reports_person_present(self):
        analysis = PresenceAnalysis(
            [
                PresenceDetection("upper_body", [1, 2, 30, 40], 0.6),
                PresenceDetection("face", [5, 6, 10, 10], 0.95),
            ]
        )

        self.assertTrue(analysis.present)
        self.assertEqual(analysis.status, "person_present")
        self.assertEqual(analysis.primary_detection.label, "face")

    def test_boxes_to_detections_converts_cv_boxes(self):
        detections = boxes_to_detections("full_body", [(1, 2, 30, 40)], 0.7)

        self.assertEqual(detections, [PresenceDetection("full_body", [1, 2, 30, 40], 0.7)])


if __name__ == "__main__":
    unittest.main()
