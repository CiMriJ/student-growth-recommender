from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

TOPICS = {
    "python": ["Python 基础语法", "pandas 数据处理", "Python 自动化脚本", "NumPy 数值计算"],
    "machine_learning": ["机器学习入门", "分类模型评估", "特征工程实战", "模型调参与交叉验证"],
    "deep_learning": ["PyTorch 张量基础", "神经网络训练流程", "CNN 图像分类", "文本分类模型"],
    "nlp": ["中文分词与文本清洗", "TF-IDF 与文本相似度", "RAG 知识库问答", "Prompt 工程实践"],
    "web_backend": ["Flask 接口开发", "REST API 设计", "SQLite 项目实践", "用户登录与权限"],
    "linux": ["Linux 常用命令", "Shell 脚本", "Socket 网络编程", "Git 团队协作"],
    "math": ["线性代数复习", "概率论基础", "最优化方法", "统计分析入门"],
    "career": ["算法岗简历优化", "项目面试表达", "技术博客写作", "GitHub 项目维护"],
}

DIFFICULTY = ["基础", "进阶", "实战"]
RESOURCE_TYPES = ["课程", "实验", "文章", "项目", "题单"]


def generate_resources(seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    idx = 1
    for topic, titles in TOPICS.items():
        for title in titles:
            for resource_type in rng.sample(RESOURCE_TYPES, 2):
                difficulty = rng.choice(DIFFICULTY)
                tags = [topic, difficulty, resource_type]
                if topic in {"nlp", "machine_learning", "deep_learning"}:
                    tags.append("algorithm")
                if topic in {"python", "linux", "web_backend"}:
                    tags.append("engineering")
                rows.append(
                    {
                        "resource_id": f"res_{idx:03d}",
                        "title": f"{title}{resource_type}",
                        "topic": topic,
                        "difficulty": difficulty,
                        "type": resource_type,
                        "duration_min": rng.choice([25, 40, 60, 90, 120]),
                        "tags": " ".join(tags),
                    }
                )
                idx += 1
    return pd.DataFrame(rows)


def generate_interactions(resources: pd.DataFrame, users: int = 160, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    started_at = datetime(2026, 3, 1)
    rows: list[dict[str, object]] = []
    topics = list(TOPICS)
    resource_records = resources.to_dict("records")
    for user_no in range(1, users + 1):
        user_id = f"user_{user_no:03d}"
        goals = set(rng.sample(topics, 3))
        level = rng.choice(["基础", "进阶", "实战"])
        for resource in rng.sample(resource_records, rng.randint(10, 18)):
            match = resource["topic"] in goals
            difficulty_match = resource["difficulty"] == level
            base = 2.3 + (1.5 if match else 0) + (0.5 if difficulty_match else 0) + rng.uniform(-0.7, 0.7)
            rating = max(1, min(5, round(base)))
            completion = max(0.05, min(1.0, 0.35 + 0.12 * rating + rng.uniform(-0.18, 0.25)))
            rows.append(
                {
                    "user_id": user_id,
                    "resource_id": resource["resource_id"],
                    "rating": rating,
                    "completion": round(completion, 2),
                    "goals": " ".join(sorted(goals)),
                    "level": level,
                    "timestamp": (started_at + timedelta(days=rng.randint(0, 120), hours=rng.randint(0, 23))).isoformat(timespec="seconds"),
                }
            )
    return pd.DataFrame(rows).sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def generate_dataset(raw_dir: str | Path, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    resources = generate_resources(seed)
    interactions = generate_interactions(resources, seed=seed)
    resources.to_csv(raw_dir / "resources.csv", index=False, encoding="utf-8-sig")
    interactions.to_csv(raw_dir / "interactions.csv", index=False, encoding="utf-8-sig")
    return resources, interactions


def load_dataset(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    resources = pd.read_csv(raw_dir / "resources.csv", encoding="utf-8-sig")
    interactions = pd.read_csv(raw_dir / "interactions.csv", encoding="utf-8-sig")
    return resources, interactions
