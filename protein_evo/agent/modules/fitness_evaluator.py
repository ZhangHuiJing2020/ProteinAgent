"""Fitness Evaluator 模块（工具调用）。

调用适应度预测模型对候选突变打分。
这是 Agent 中唯一的"工具调用"环节，不是 LLM 自己拍脑袋。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ...model.fitness_model import FitnessModel
from ...knowledge.mutation_rules import validate_mutant, MutationValidation
from ..types import Candidate


def evaluate_candidates(
    candidates: list[Candidate],
    model: FitnessModel,
    history_df: Optional[pd.DataFrame] = None,
    use_rules: bool = True,
    max_mutations: int = 4,
) -> list[Candidate]:
    """对候选突变打分。

    Args:
        candidates: 候选列表
        model: 训练好的适应度预测模型
        history_df: 历史数据（用于规则验证）
        use_rules: 是否应用突变规则过滤
        max_mutations: 单次最大突变数

    Returns:
        更新后的候选列表（含 predicted_fitness 和 validation 字段）
    """
    if not candidates:
        return []

    # 批量构造 DataFrame 用于模型预测
    df = pd.DataFrame([
        {"mutant": c.mutant, "mutated_sequence": ""}
        for c in candidates
    ])
    # 模型预测
    preds = model.predict(df)
    if isinstance(preds, float):
        preds = [preds]

    # 应用规则过滤
    for c, pred in zip(candidates, preds):
        c.predicted_fitness = float(pred)
        if use_rules:
            v = validate_mutant(c.mutant, history_df=history_df, max_mutations=max_mutations)
            c.validation = v.to_dict()

    # 过滤无效候选
    if use_rules:
        candidates = [c for c in candidates if c.validation is None or c.validation.get("is_valid", True)]

    # 按预测 fitness 降序排序
    candidates.sort(key=lambda c: c.predicted_fitness if c.predicted_fitness is not None else -1e9, reverse=True)

    return candidates


if __name__ == "__main__":
    # 自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    model_path = Path("results/models/fitness_model.joblib")
    if not model_path.exists():
        print("模型未训练，请先运行 03_train_model.py")
        sys.exit(0)

    model = FitnessModel.load(model_path)
    cands = [
        Candidate(mutant="V39A", n_mutations=1),
        Candidate(mutant="V39A:V54I", n_mutations=2),
        Candidate(mutant="V39*", n_mutations=1),  # 应被过滤
    ]
    scored = evaluate_candidates(cands, model)
    print(f"打分后 {len(scored)} 个候选:")
    for c in scored:
        print(f"  {c.mutant}: pred={c.predicted_fitness:.3f}, valid={c.validation['is_valid'] if c.validation else 'N/A'}")
