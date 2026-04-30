import unittest

from pi_agent.vision import FaceEngine, largest_face


class FakeFeature:
    def flatten(self):
        return [0.1, 0.2, 0.3]


class FakeRecognizer:
    def __init__(self):
        self.face_box_type_name = None

    def alignCrop(self, frame, face_box):
        self.face_box_type_name = type(face_box).__name__
        if self.face_box_type_name != "ndarray":
            raise TypeError("face_box must be converted to numpy.ndarray")
        return "aligned-face"

    def feature(self, aligned):
        return FakeFeature()


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

    def test_embedding_for_face_converts_face_box_for_opencv(self):
        engine = object.__new__(FaceEngine)
        engine.recognizer = FakeRecognizer()

        embedding = engine.embedding_for_face(frame="fake-frame", face=[1, 2, 3, 4, 0.9])

        self.assertEqual(embedding, [0.1, 0.2, 0.3])
        self.assertEqual(engine.recognizer.face_box_type_name, "ndarray")


if __name__ == "__main__":
    unittest.main()
