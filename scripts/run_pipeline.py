from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from growth_recommender.dataset import generate_dataset
from growth_recommender.metrics import write_evaluation
from growth_recommender.recommender import HybridRecommender


def main() -> None:
    resources, interactions = generate_dataset(ROOT / "data/raw")
    report = write_evaluation(resources, interactions, ROOT / "reports/evaluation.json")
    model = HybridRecommender().fit(resources, interactions)
    examples = model.recommend("user_001", top_k=5)
    lines = ["# Recommendation Examples", "", "用户：user_001", ""]
    for rec in examples:
        lines.append(f"- {rec.title}（{rec.resource_id}）：score={rec.score}，{rec.reason}")
    (ROOT / "reports/recommendation_examples.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "resources": len(resources),
        "interactions": len(interactions),
        "hit_rate_at_5": report["hit_rate"],
        "topic_hit_rate_at_5": report["topic_hit_rate"],
        "ndcg_at_5": report["ndcg"],
        "catalog_coverage": report["catalog_coverage"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
