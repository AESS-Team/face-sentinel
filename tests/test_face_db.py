import tempfile
import unittest
from pathlib import Path

from pi_agent.face_db import append_embedding, load_embeddings, save_embeddings


class FaceDbTests(unittest.TestCase):
    def test_append_embedding_creates_identity_list(self):
        database = {}

        append_embedding(database, "alice", [1.0, 2.0])

        self.assertEqual(database, {"alice": [[1.0, 2.0]]})

    def test_load_missing_embeddings_returns_empty_database(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(load_embeddings(Path(temp_dir) / "missing.json"), {})

    def test_save_and_load_embeddings_round_trip(self):
        database = {"alice": [[1.0, 2.0]], "bob": [[0.0, 1.0]]}

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "embeddings.json"
            save_embeddings(path, database)

            self.assertEqual(load_embeddings(path), database)


if __name__ == "__main__":
    unittest.main()
