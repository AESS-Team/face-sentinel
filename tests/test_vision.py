import unittest

from pi_agent.vision import (
    FaceEngine,
    create_face_detector,
    create_face_recognizer,
    largest_face,
)


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


class ModernDetectorClass:
    @staticmethod
    def create(*args):
        return ("modern-detector", args)


class ModernRecognizerClass:
    @staticmethod
    def create(*args):
        return ("modern-recognizer", args)


class ModernCv2:
    FaceDetectorYN = ModernDetectorClass
    FaceRecognizerSF = ModernRecognizerClass


class LegacyCv2:
    @staticmethod
    def FaceDetectorYN_create(*args):
        return ("legacy-detector", args)

    @staticmethod
    def FaceRecognizerSF_create(*args):
        return ("legacy-recognizer", args)


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

    def test_create_face_detector_supports_modern_opencv_api(self):
        detector = create_face_detector(ModernCv2, "detector.onnx", (640, 480), 0.9, 0.3, 5000)

        self.assertEqual(detector[0], "modern-detector")

    def test_create_face_detector_supports_legacy_opencv_api(self):
        detector = create_face_detector(LegacyCv2, "detector.onnx", (640, 480), 0.9, 0.3, 5000)

        self.assertEqual(detector[0], "legacy-detector")

    def test_create_face_recognizer_supports_modern_opencv_api(self):
        recognizer = create_face_recognizer(ModernCv2, "recognizer.onnx")

        self.assertEqual(recognizer[0], "modern-recognizer")

    def test_create_face_recognizer_supports_legacy_opencv_api(self):
        recognizer = create_face_recognizer(LegacyCv2, "recognizer.onnx")

        self.assertEqual(recognizer[0], "legacy-recognizer")


if __name__ == "__main__":
    unittest.main()
