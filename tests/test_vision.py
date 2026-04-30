import unittest

from pi_agent.vision import largest_face


class VisionTests(unittest.TestCase):
    def test_largest_face_returns_face_with_biggest_box_area(self):
        faces = [
            [0, 0, 20, 20, 0.9],
            [0, 0, 40, 15, 0.8],
            [0, 0, 10, 10, 0.99],
        ]

        self.assertEqual(largest_face(faces), faces[1])

    def test_largest_face_returns_none_when_no_faces_exist(self):
        self.assertIsNone(largest_face([]))


if __name__ == "__main__":
    unittest.main()
