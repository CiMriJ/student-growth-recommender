from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"[a-zA-Z0-9_+#]+|[\u4e00-\u9fff]{1,4}")


def tokenize(text: object) -> list[str]:
    if text is None:
        return []
    return [token.lower() for token in TOKEN_RE.findall(str(text))]


class TfidfVectorizer:
    def __init__(self, max_features: int = 600):
        self.max_features = max_features
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}

    def fit(self, texts: list[str]) -> "TfidfVectorizer":
        doc_freq: Counter[str] = Counter()
        term_freq: Counter[str] = Counter()
        for text in texts:
            tokens = tokenize(text)
            term_freq.update(tokens)
            doc_freq.update(set(tokens))
        self.vocabulary = [token for token, _ in term_freq.most_common(self.max_features)]
        n_docs = len(texts)
        self.idf = {token: math.log((1 + n_docs) / (1 + doc_freq[token])) + 1 for token in self.vocabulary}
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=float)
        index = {token: i for i, token in enumerate(self.vocabulary)}
        for row, text in enumerate(texts):
            counts = Counter(token for token in tokenize(text) if token in index)
            total = sum(counts.values()) or 1
            for token, count in counts.items():
                matrix[row, index[token]] = (count / total) * self.idf[token]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)
