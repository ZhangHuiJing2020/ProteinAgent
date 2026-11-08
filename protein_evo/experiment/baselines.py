"""
- 所有方法都在 test_pool 上评分（保证有真实 fitness）
- random: 从 test_pool 随机采样 N 个，按真实 fitness 取 Top-k
- model_only: 从 test_pool 随机采样 N 个，按模型预测取 Top-k，再查真实 fitness
- LLM agent: 从 test_pool 中根据 agent 推荐打分取 Top-k
"""
from __future__ import annotations

import random
from typing import Optional

import pandas as pd

from ..data.wt import WT_SEQUENCE, MUTATION_SITES
from ..knowledge.amino_acid_props import STANDARD_AAS
from ..model.fitness_model import FitnessModel
from ..agent.types import RoundResult


def random_baseline(
    history_df: pd.DataFrame,
    test_pool: dict[str, float],
    n_candidates: int = 50,
    topk: int = 10,
    seed: int = 42,
    round_number: int = 1,
) -> RoundResult:
    """随机突变基线。

    从 test_pool 随机采样 N 个候选，按真实 fitness 取 Top-k。
    公平性：与 LLM agent 一样，从同一个 test_pool 采样。
    """
    rng = random.Random(seed + round_number)
    pool_keys = list(test_pool.keys())
    if not pool_keys:
        return RoundResult(round_number=round_number, method="random")

    n_sample = min(n_candidates, len(pool_keys))
    sampled = rng.sample(pool_keys, n_sample)
    # 按真实 fitness 排序
    sampled_with_fitness = [(m, test_pool[m]) for m in sampled]
    sampled_with_fitness.sort(key=lambda x: -x[1])

    topk_items = sampled_with_fitness[:topk]
    real_fitness_values = [f for _, f in topk_items]

    return RoundResult(
        round_number=round_number,
        method="random",
        topk=[
            {"mutant": m, "rank": i + 1, "real_fitness": f, "score_source": "real"}
            for i, (m, f) in enumerate(topk_items)
        ],
        topk_real_fitness=real_fitness_values,
        metrics={
            "topk_max": max(real_fitness_values) if real_fitness_values else 0,
            "topk_mean": sum(real_fitness_values) / len(real_fitness_values) if real_fitness_values else 0,
            "topk_min": min(real_fitness_values) if real_fitness_values else 0,
        },
    )


def model_only_baseline(
    history_df: pd.DataFrame,
    test_pool: dict[str, float],
    model: FitnessModel,
    n_candidates: int = 50,
    topk: int = 10,
    seed: int = 42,
    round_number: int = 1,
) -> RoundResult:
    """模型直接推荐基线。

    公平性：从 test_pool 采样 N 个候选（与 random 同采样池），
    用模型预测排序取 Top-k，报告真实 fitness。
    对比 random：如果模型预测有用，Top-k 真实 fitness 应高于 random。
    """
    rng = random.Random(seed + round_number)
    pool_keys = list(test_pool.keys())
    if not pool_keys:
        return RoundResult(round_number=round_number, method="model_only")

    n_sample = min(n_candidates, len(pool_keys))
    sampled = rng.sample(pool_keys, n_sample)

    # 模型预测
    df = pd.DataFrame([{"mutant": m, "mutated_sequence": ""} for m in sampled])
    preds = model.predict(df)
    if isinstance(preds, float):
        preds = [preds]

    # 按预测排序
    scored = list(zip(sampled, preds))
    scored.sort(key=lambda x: -x[1])

    topk_items = scored[:topk]
    # 真实 fitness 来自 test_pool
    real_fitness_values = [test_pool[m] for m, _ in topk_items]

    return RoundResult(
        round_number=round_number,
        method="model_only",
        topk=[
            {"mutant": m, "rank": i + 1, "predicted_fitness": float(p),
             "real_fitness": test_pool[m], "score_source": "real"}
            for i, (m, p) in enumerate(topk_items)
        ],
        topk_real_fitness=real_fitness_values,
        metrics={
            "topk_max": max(real_fitness_values) if real_fitness_values else 0,
            "topk_mean": sum(real_fitness_values) / len(real_fitness_values) if real_fitness_values else 0,
            "topk_min": min(real_fitness_values) if real_fitness_values else 0,
        },
    )


if __name__ == "__main__":
    # 自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    model = FitnessModel.load("results/models/fitness_model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    test_pool = {str(r["mutant"]): float(r["DMS_score"]) for _, r in test_df.iterrows() if r["mutant"] != "wt"}
    train_df = pd.read_csv("data/processed/train.csv")

    r1 = random_baseline(train_df, test_pool, n_candidates=50, topk=5)
    print(f"Random: {r1.metrics}")

    r2 = model_only_baseline(train_df, test_pool, model, n_candidates=50, topk=5)
    print(f"Model_only: {r2.metrics}")
