from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growth_recommender.dataset import generate_dataset
from growth_recommender.metrics import evaluate_recommender
from growth_recommender.recommender import HybridRecommender


def test_recommender_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        resources, interactions = generate_dataset(Path(tmp), seed=8)
        report = evaluate_recommender(resources, interactions, top_k=5)
        model = HybridRecommender().fit(resources, interactions)
        recs = model.recommend("user_001", top_k=5)
        assert len(resources) >= 50
        assert len(interactions) >= 1000
        assert len(recs) == 5
        assert all(rec.score >= 0 for rec in recs)
        assert report["evaluated_users"] > 100
        assert report["catalog_coverage"] > 0.10


if __name__ == "__main__":
    test_recommender_pipeline()
    print("all tests passed")
