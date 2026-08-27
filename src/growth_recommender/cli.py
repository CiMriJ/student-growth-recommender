from __future__ import annotations

import argparse
import json

from .dataset import generate_dataset, load_dataset
from .metrics import write_evaluation
from .recommender import HybridRecommender


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hybrid student learning resource recommender")
    sub = parser.add_subparsers(dest="command", required=True)
    p_generate = sub.add_parser("generate", help="Generate sample resource and interaction data")
    p_generate.add_argument("--out", default="data/raw")
    p_eval = sub.add_parser("evaluate", help="Run leave-one-out offline evaluation")
    p_eval.add_argument("--data", default="data/raw")
    p_eval.add_argument("--report", default="reports/evaluation.json")
    p_rec = sub.add_parser("recommend", help="Recommend resources for a user")
    p_rec.add_argument("user_id")
    p_rec.add_argument("--data", default="data/raw")
    p_rec.add_argument("--top-k", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        resources, interactions = generate_dataset(args.out)
        print(json.dumps({"resources": len(resources), "interactions": len(interactions), "path": args.out}, ensure_ascii=False))
    elif args.command == "evaluate":
        resources, interactions = load_dataset(args.data)
        report = write_evaluation(resources, interactions, args.report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "recommend":
        resources, interactions = load_dataset(args.data)
        model = HybridRecommender().fit(resources, interactions)
        recs = model.recommend(args.user_id, top_k=args.top_k)
        print(json.dumps([rec.__dict__ for rec in recs], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
