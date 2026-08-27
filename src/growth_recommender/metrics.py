from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .recommender import HybridRecommender


def leave_one_out_split(interactions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_rows = []
    train_parts = []
    for _, group in interactions.sort_values("timestamp").groupby("user_id"):
        if len(group) < 4:
            train_parts.append(group)
            continue
        positive = group[group["rating"] >= 4]
        holdout = positive.tail(1) if not positive.empty else group.tail(1)
        holdout_rows.append(holdout)
        train_parts.append(group.drop(holdout.index))
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(holdout_rows).reset_index(drop=True)


def evaluate_recommender(resources: pd.DataFrame, interactions: pd.DataFrame, top_k: int = 5) -> dict[str, object]:
    train, holdout = leave_one_out_split(interactions)
    model = HybridRecommender().fit(resources, train)
    hits = 0
    topic_hits = 0
    ndcg_values = []
    recommended_items = set()
    evaluated = 0
    examples = []
    topic_by_item = dict(zip(resources["resource_id"], resources["topic"]))
    for row in holdout.itertuples(index=False):
        recs = model.recommend(row.user_id, top_k=top_k)
        ids = [rec.resource_id for rec in recs]
        recommended_items.update(ids)
        evaluated += 1
        if row.resource_id in ids:
            hits += 1
            rank = ids.index(row.resource_id) + 1
            ndcg_values.append(1 / np.log2(rank + 1))
        else:
            ndcg_values.append(0.0)
        holdout_topic = topic_by_item.get(row.resource_id)
        if any(topic_by_item.get(item_id) == holdout_topic for item_id in ids):
            topic_hits += 1
        if len(examples) < 5:
            examples.append({"user_id": row.user_id, "holdout": row.resource_id, "recommended": ids})
    return {
        "top_k": top_k,
        "evaluated_users": int(evaluated),
        "hit_rate": round(hits / evaluated, 4) if evaluated else 0.0,
        "topic_hit_rate": round(topic_hits / evaluated, 4) if evaluated else 0.0,
        "ndcg": round(float(np.mean(ndcg_values)), 4) if ndcg_values else 0.0,
        "catalog_coverage": round(len(recommended_items) / len(resources), 4) if len(resources) else 0.0,
        "train_interactions": int(len(train)),
        "holdout_interactions": int(len(holdout)),
        "examples": examples,
    }


def write_evaluation(resources: pd.DataFrame, interactions: pd.DataFrame, output_path: str | Path) -> dict[str, object]:
    report = evaluate_recommender(resources, interactions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
