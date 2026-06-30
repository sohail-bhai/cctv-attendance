from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .utils import cosine_similarity, normalize_embedding


@dataclass
class MatchResult:
    accepted: bool
    roll_no: str | None
    best_score: float
    second_roll_no: str | None
    second_score: float
    margin: float
    reason: str


class StudentEmbeddingDB:
    def __init__(self, records: List[dict], metadata: Optional[dict] = None) -> None:
        self.records = records
        self.metadata = metadata or {}
        self.by_roll: Dict[str, List[np.ndarray]] = defaultdict(list)
        for record in records:
            roll_no = str(record["roll_no"])
            emb = normalize_embedding(record["embedding"])
            self.by_roll[roll_no].append(emb)

        self.roll_numbers = sorted(self.by_roll.keys())
        self.centroids: Dict[str, np.ndarray] = {}
        for roll_no, embeddings in self.by_roll.items():
            centroid = normalize_embedding(np.mean(np.stack(embeddings), axis=0))
            self.centroids[roll_no] = centroid

    def __len__(self) -> int:
        return len(self.records)

    @property
    def student_count(self) -> int:
        return len(self.roll_numbers)

    def save(self, path: Path) -> None:
        payload = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "metadata": self.metadata,
            "records": self.records,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> "StudentEmbeddingDB":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        return cls(records=payload["records"], metadata=payload.get("metadata", {}))

    def match(
        self,
        query_embedding: np.ndarray,
        match_threshold: float = 0.38,
        margin_threshold: float = 0.03,
        aggregate: str = "centroid",
    ) -> MatchResult:
        """
        Match a query face embedding against enrolled students.

        match_threshold: higher means stricter for cosine similarity.
        margin_threshold: best score must beat second-best by this much.
        aggregate:
            - centroid: compare with one average embedding per student. Fast and stable.
            - max: compare with all images and keep the student's best score. Useful with varied dataset.
            - top3: average student's top 3 image scores. Balanced.
        """
        if not self.roll_numbers:
            return MatchResult(False, None, 0.0, None, 0.0, 0.0, "empty_database")

        q = normalize_embedding(query_embedding)
        scores: Dict[str, float] = {}

        for roll_no in self.roll_numbers:
            if aggregate == "centroid":
                scores[roll_no] = cosine_similarity(q, self.centroids[roll_no])
            else:
                sims = np.array([cosine_similarity(q, emb) for emb in self.by_roll[roll_no]], dtype=np.float32)
                if aggregate == "max":
                    scores[roll_no] = float(np.max(sims))
                elif aggregate == "top3":
                    top = np.sort(sims)[-min(3, len(sims)):]
                    scores[roll_no] = float(np.mean(top))
                else:
                    raise ValueError(f"Unknown aggregate mode: {aggregate}")

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_roll, best_score = ranked[0]
        if len(ranked) > 1:
            second_roll, second_score = ranked[1]
        else:
            second_roll, second_score = None, -1.0

        margin = best_score - second_score if second_roll is not None else best_score

        if best_score < match_threshold:
            return MatchResult(False, best_roll, best_score, second_roll, second_score, margin, "score_below_threshold")
        if margin < margin_threshold:
            return MatchResult(False, best_roll, best_score, second_roll, second_score, margin, "margin_too_small")

        return MatchResult(True, best_roll, best_score, second_roll, second_score, margin, "accepted")
