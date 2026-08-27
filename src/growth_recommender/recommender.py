from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .vectorizer import TfidfVectorizer, tokenize


@dataclass
class Recommendation:
    resource_id: str
    title: str
    score: float
    reason: str


class HybridRecommender:
    def __init__(self, content_weight: float = 0.30, cf_weight: float = 0.60, popularity_weight: float = 0.10):
        self.content_weight = content_weight
        self.cf_weight = cf_weight
        self.popularity_weight = popularity_weight
        self.resources: pd.DataFrame | None = None
        self.interactions: pd.DataFrame | None = None
        self.vectorizer = TfidfVectorizer()
        self.item_matrix: np.ndarray | None = None
        self.item_ids: list[str] = []
        self.item_index: dict[str, int] = {}
        self.popularity: np.ndarray | None = None
        self.item_similarity: np.ndarray | None = None

    def fit(self, resources: pd.DataFrame, interactions: pd.DataFrame) -> "HybridRecommender":
        self.resources = resources.reset_index(drop=True).copy()
        self.interactions = interactions.copy()
        self.item_ids = self.resources["resource_id"].tolist()
        self.item_index = {item_id: i for i, item_id in enumerate(self.item_ids)}
        item_text = (
            self.resources["title"].astype(str)
            + " "
            + self.resources["topic"].astype(str)
            + " "
            + self.resources["difficulty"].astype(str)
            + " "
            + self.resources["tags"].astype(str)
        ).tolist()
        self.item_matrix = self.vectorizer.fit_transform(item_text)
        self.popularity = self._build_popularity()
        self.item_similarity = self._build_item_similarity()
        return self

    def _build_popularity(self) -> np.ndarray:
        assert self.interactions is not None
        scores = np.zeros(len(self.item_ids), dtype=float)
        grouped = self.interactions.groupby("resource_id").agg({"rating": "mean", "completion": "mean", "user_id": "count"})
        for item_id, row in grouped.iterrows():
            if item_id in self.item_index:
                scores[self.item_index[item_id]] = row["rating"] * 0.6 + row["completion"] * 5 * 0.3 + min(row["user_id"], 20) / 20 * 0.5
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def _build_item_similarity(self) -> np.ndarray:
        assert self.interactions is not None
        users = sorted(self.interactions["user_id"].unique())
        user_index = {user_id: i for i, user_id in enumerate(users)}
        matrix = np.zeros((len(users), len(self.item_ids)), dtype=float)
        for row in self.interactions.itertuples(index=False):
            if row.resource_id in self.item_index:
                matrix[user_index[row.user_id], self.item_index[row.resource_id]] = max(float(row.rating), 0) * max(float(row.completion), 0.05)
        norms = np.linalg.norm(matrix, axis=0, keepdims=True)
        normalized = matrix / np.maximum(norms, 1e-12)
        return normalized.T @ normalized

    def _user_profile(self, user_id: str) -> tuple[np.ndarray, set[str], set[str]]:
        assert self.item_matrix is not None and self.interactions is not None
        user_rows = self.interactions[self.interactions["user_id"] == user_id]
        seen = set(user_rows["resource_id"])
        if user_rows.empty:
            return np.zeros(self.item_matrix.shape[1]), seen, set()
        weights = []
        vectors = []
        goals = set()
        for row in user_rows.itertuples(index=False):
            if row.resource_id in self.item_index:
                vectors.append(self.item_matrix[self.item_index[row.resource_id]])
                weights.append(max(float(row.rating), 1.0) * max(float(row.completion), 0.05))
            goals.update(str(row.goals).split())
        if not vectors:
            return np.zeros(self.item_matrix.shape[1]), seen, goals
        profile = np.average(np.vstack(vectors), axis=0, weights=np.array(weights))
        norm = np.linalg.norm(profile)
        return profile / norm if norm else profile, seen, goals

    def recommend(self, user_id: str, top_k: int = 5) -> list[Recommendation]:
        assert self.resources is not None and self.item_matrix is not None and self.popularity is not None and self.item_similarity is not None
        profile, seen, goals = self._user_profile(user_id)
        content_scores = self.item_matrix @ profile if profile.any() else np.zeros(len(self.item_ids))
        seen_indices = [self.item_index[item_id] for item_id in seen if item_id in self.item_index]
        cf_scores = self.item_similarity[seen_indices].mean(axis=0) if seen_indices else np.zeros(len(self.item_ids))
        scores = self.content_weight * content_scores + self.cf_weight * cf_scores + self.popularity_weight * self.popularity
        for item_id in seen:
            if item_id in self.item_index:
                scores[self.item_index[item_id]] = -1
        ranked = np.argsort(scores)[::-1][:top_k]
        recs: list[Recommendation] = []
        for idx in ranked:
            row = self.resources.iloc[int(idx)]
            tags = set(tokenize(row["tags"]))
            matched = sorted(goals & tags)
            reason = f"匹配学习目标：{', '.join(matched)}" if matched else f"同类资源热度较高：{row['topic']}"
            recs.append(Recommendation(str(row["resource_id"]), str(row["title"]), round(float(scores[idx]), 4), reason))
        return recs
