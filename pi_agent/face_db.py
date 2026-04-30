from __future__ import annotations

import json
from pathlib import Path


EmbeddingDatabase = dict[str, list[list[float]]]


def append_embedding(database: EmbeddingDatabase, identity: str, embedding: list[float]) -> None:
    database.setdefault(identity, []).append([float(value) for value in embedding])


def load_embeddings(path: Path) -> EmbeddingDatabase:
    if not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    database: EmbeddingDatabase = {}
    for identity, embeddings in raw.items():
        database[str(identity)] = [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]
    return database


def save_embeddings(path: Path, database: EmbeddingDatabase) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(database, indent=2, sort_keys=True),
        encoding="utf-8",
    )
