from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable, Mapping, Sequence


Embedding = Sequence[float]
KnownEmbeddings = Mapping[str, Sequence[Embedding]]


@dataclass(frozen=True)
class MatchDecision:
    identity: str
    score: float
    is_known: bool


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]

    if len(left_values) != len(right_values):
        raise ValueError("Embeddings must have the same length")

    left_norm = sqrt(sum(value * value for value in left_values))
    right_norm = sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(a * b for a, b in zip(left_values, right_values))
    return dot / (left_norm * right_norm)


def best_identity_match(
    embedding: Embedding,
    known_embeddings: KnownEmbeddings,
    threshold: float,
) -> MatchDecision:
    best_identity = "unknown"
    best_score = 0.0

    for identity, identity_embeddings in known_embeddings.items():
        for known_embedding in identity_embeddings:
            score = cosine_similarity(embedding, known_embedding)
            if score > best_score:
                best_identity = identity
                best_score = score

    if best_identity != "unknown" and best_score >= threshold:
        return MatchDecision(best_identity, best_score, True)

    return MatchDecision("unknown", best_score, False)
