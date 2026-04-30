import unittest

from pi_agent.recognition import best_identity_match, cosine_similarity


class RecognitionTests(unittest.TestCase):
    def test_cosine_similarity_matches_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_best_identity_match_accepts_known_face_above_threshold(self):
        known = {
            "alice": [[1.0, 0.0, 0.0], [0.98, 0.02, 0.0]],
            "bob": [[0.0, 1.0, 0.0]],
        }

        decision = best_identity_match([0.99, 0.01, 0.0], known, threshold=0.85)

        self.assertTrue(decision.is_known)
        self.assertEqual(decision.identity, "alice")
        self.assertGreaterEqual(decision.score, 0.85)

    def test_best_identity_match_marks_low_score_as_unknown(self):
        known = {"alice": [[1.0, 0.0, 0.0]]}

        decision = best_identity_match([0.0, 1.0, 0.0], known, threshold=0.85)

        self.assertFalse(decision.is_known)
        self.assertEqual(decision.identity, "unknown")

    def test_best_identity_match_handles_empty_database(self):
        decision = best_identity_match([1.0, 0.0], {}, threshold=0.85)

        self.assertFalse(decision.is_known)
        self.assertEqual(decision.identity, "unknown")
        self.assertEqual(decision.score, 0.0)


if __name__ == "__main__":
    unittest.main()
