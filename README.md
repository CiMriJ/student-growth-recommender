# 学生成长学习资源混合推荐系统

该项目模拟“学生学习目标 - 学习资源 - 行为交互”的推荐业务，使用内容相似度、协同过滤和资源热度融合打分，为学生推荐适合当前阶段的课程、实验、文章、项目和题单。

## 项目亮点

- 构建学习资源数据、学生目标画像和学习行为交互数据。
- 使用 TF-IDF 表征资源标题、主题、难度和标签，形成资源向量。
- 使用用户历史行为加权生成用户画像向量。
- 基于用户-资源交互矩阵计算 Item-Item 共现相似度。
- 融合内容推荐、协同过滤和热度分，输出 Top-K 推荐及原因解释。
- 使用留一法离线评估 HitRate@5、NDCG@5 和 Catalog Coverage。

## 快速开始

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py
```

推荐示例：

```bash
set PYTHONPATH=src
python -m growth_recommender.cli recommend user_001 --top-k 5
```

## 项目结构

```text
student-growth-recommender/
  src/growth_recommender/
    dataset.py       # 模拟资源与行为数据
    vectorizer.py    # TF-IDF 向量化
    recommender.py   # 混合推荐模型
    metrics.py       # 离线评估
    cli.py           # 命令行入口
  scripts/run_pipeline.py
  tests/run_tests.py
  docs/resume_snippet.md
```

## 验收结果

已通过 `scripts/run_pipeline.py` 和 `tests/run_tests.py` 验收：

- 学习资源：64 个，用户行为交互：2,257 条。
- 留一法评估用户：160 个。
- HitRate@5：0.1500，NDCG@5：0.0948。
- Topic-Hit@5：0.4250，用于衡量是否推荐到同学习方向资源。
- Catalog Coverage：0.9844，说明推荐结果覆盖了绝大多数资源，不是只推热门内容。

> 说明：仓库内数据为可复现的模拟样例数据，重点展示推荐算法、用户画像和离线评估流程。
