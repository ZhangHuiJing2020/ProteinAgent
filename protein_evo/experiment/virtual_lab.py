"""虚拟实验评估器。
候选若在测试集子集中用真实 fitness，否则用模型预测。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from ..data.loader import parse_mutant_str, infer_mutations_from_sequence, mutations_to_str
from ..data.wt import WT_SEQUENCE
from ..model.fitness_model import FitnessModel


@dataclass
class ScoreResult:
    """虚拟实验评分结果。"""
    mutant: str
    fitness: float
    source: str  # "real" | "predicted"


class VirtualLab:
    """虚拟实验评估器。

    - test_pool: 测试集的一个子集，模拟"已做的虚拟实验"
    - 不在 test_pool 中的候选用模型预测
    """

    def __init__(
        self,
        test_pool: dict[str, float],
        model: FitnessModel,
    ):
        """
        Args:
            test_pool: {mutant_str: real_fitness} 测试集子集
            model: 训练好的适应度预测模型
        """
        self.test_pool = test_pool
        self.model = model
        self._stats = {"real": 0, "predicted": 0}

    @classmethod
    def from_test_df(
        cls,
        test_df: pd.DataFrame,
        model: FitnessModel,
        pool_size: int = 500,
        seed: int = 42,
    ) -> "VirtualLab":
        """从测试集 DataFrame 构造 VirtualLab。

        Args:
            test_df: 测试集，含 mutant, DMS_score
            model: 模型
            pool_size: test_pool 大小
            seed: 随机种子
        """
        if len(test_df) == 0:
            return cls({}, model)
        # 子采样
        if len(test_df) > pool_size:
            test_df = test_df.sample(pool_size, random_state=seed)
        test_pool = {}
        for _, row in test_df.iterrows():
            mutant = str(row.get("mutant", ""))
            if mutant and mutant != "wt":
                test_pool[mutant] = float(row["DMS_score"])
        return cls(test_pool, model)

    def score(self, mutant: str) -> ScoreResult:
        """评分单个突变。

        优先返回 test_pool 中的真实 fitness；不在 test_pool 中用模型预测。
        当 orchestrator 用 candidate_pool 模式时，Top-k 候选必然在 test_pool 中。
        """
        if mutant in self.test_pool:
            self._stats["real"] += 1
            return ScoreResult(mutant=mutant, fitness=self.test_pool[mutant], source="real")
        # 用模型预测（兜底）
        pred = self.model.predict(mutant)
        self._stats["predicted"] += 1
        return ScoreResult(mutant=mutant, fitness=float(pred), source="predicted")

    def score_batch(self, mutants: list[str]) -> list[ScoreResult]:
        """批量评分。"""
        return [self.score(m) for m in mutants]

    def stats(self) -> dict:
        return dict(self._stats)

    def reset_stats(self):
        self._stats = {"real": 0, "predicted": 0}


if __name__ == "__main__":
    # 自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    model = FitnessModel.load("results/models/fitness_model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    lab = VirtualLab.from_test_df(test_df, model, pool_size=100)

    # 评分一些候选
    for mutant in ["V39A", "V39A:V54I", "D40E:G41A"]:
        r = lab.score(mutant)
        print(f"  {mutant}: fitness={r.fitness:.3f} ({r.source})")
    print(f"\n统计: {lab.stats()}")
