from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .face_db import EmbeddingDatabase, append_embedding, save_embeddings
from .vision import FaceEngine, VisionSetupError, iter_image_paths


def build_database(config_path: Path, only_name: str | None = None) -> EmbeddingDatabase:
    config = load_config(config_path)
    engine = FaceEngine(config)
    database: EmbeddingDatabase = {}

    config.known_faces_dir.mkdir(parents=True, exist_ok=True)
    identity_dirs = [path for path in sorted(config.known_faces_dir.iterdir()) if path.is_dir()]

    for identity_dir in identity_dirs:
        identity = identity_dir.name
        if only_name is not None and identity != only_name:
            continue

        for image_path in iter_image_paths(identity_dir):
            try:
                embedding, face = engine.embedding_for_image(image_path)
            except VisionSetupError as exc:
                print(f"Skipping {image_path}: {exc}")
                continue

            append_embedding(database, identity, embedding)
            bbox = [round(float(value), 2) for value in face[:4]]
            print(f"Enrolled {identity}: {image_path} bbox={bbox}")

    return database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the known-face embedding database.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to pi_agent/config.json",
    )
    parser.add_argument("--name", help="Only enroll one folder name from known_faces")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    database = build_database(config_path, args.name)

    if not database:
        print(f"No faces enrolled. Add photos under {config.known_faces_dir / '<name>'}")
        raise SystemExit(2)

    save_embeddings(config.embeddings_path, database)
    total = sum(len(embeddings) for embeddings in database.values())
    print(f"Saved {total} embeddings for {len(database)} identities to {config.embeddings_path}")


if __name__ == "__main__":
    main()
