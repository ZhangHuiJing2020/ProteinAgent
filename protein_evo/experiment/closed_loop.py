"""闭环 DBTL 迭代框架。

参考：
- EVOLVEpro (bioRxiv 2024): 每轮用模型对候选打分 → 选 Top-k → 实验评估 → 模型用新数据重新训练
  https://www.biorxiv.org/content/10.1101/2024.07.17.604015v1
  仓库: https://github.com/mat10d/EvolvePro
- PLMeAE: 96 变体/轮闭环自动化
  https://www.semanticscholar.org/paper/Integrating-protein-language-models-and-automatic-Zhang-Chen/a57c3571bf3156bc5241e6b1c0c982bf83cf9607

"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..data.loader import parse_mutant_str
from ..data.wt import WT_SEQUENCE, FULL_LENGTH_SEQUENCE
from ..model.fitness_model import FitnessModel, FeatureConfig
from ..model.metrics import evaluate_all
from .virtual_lab import VirtualLab


class ClosedLoopDBTL:
    """闭环 DBTL 迭代框架。

    参考: EVOLVEpro (bioRxiv 2024) 的闭环迭代策略。

    每轮迭代：
    1. Propose: 模型对未评估候选打分 → 选 Top-k（支持 UCB/EI 主动学习）
    2. Evaluate: VirtualLab 评分（真实 fitness）
    3. Update: Top-k + 真实 fitness 加入训练集 → 重新训练模型
    4. Assess: 评估模型在剩余测试集上的 Spearman

    与原 batch selection（comparison.py）的区别：
    - batch selection：固定候选池筛选，模型不更新
    - closed-loop：每轮模型从新实验数据中重新训练
    """

    def __init__(
        self,
        model: FitnessModel,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        n_rounds: int = 3,
        topk: int = 10,
        acquisition: str = "mean",  # mean | ucb | ei
        ucb_beta: float = 2.0,
        seed: int = 42,
    ):
        """
        Args:
            model: 初始模型（会被每轮重新训练覆盖）
            train_df: 初始训练数据
            test_df: 测试集（作为候选池 + 评估集）
            n_rounds: 迭代轮数
            topk: 每轮选 Top-k 候选
            acquisition: 候选选择策略
                - "mean": 纯预测均值
                - "ucb": UCB = mean + beta * std（主动学习，参考 EVOLVEpro）
                - "ei": Expected Improvement
            ucb_beta: UCB 探索系数
            seed: 随机种子
        """
        self.model = model
        self.initial_train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.n_rounds = n_rounds
        self.topk = topk
        self.acquisition = acquisition
        self.ucb_beta = ucb_beta
        self.seed = seed
        self.config = model.config

    def run(self, results_dir: str = "results/closed_loop") -> list[dict]:
        """运行闭环 DBTL 迭代。

        Returns:
            list[dict]: 每轮的结果摘要
        """
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # 构造 VirtualLab（用全部测试集作为虚拟实验评估器）
        virtual_lab = VirtualLab.from_test_df(
            self.test_df, self.model, pool_size=len(self.test_df), seed=self.seed
        )

        # 初始训练集（每轮会扩充）
        train_df = self.initial_train_df.copy()
        # 已评估的候选（不重复评估）
        evaluated_mutants: set = set()
        # 累计所有轮次的 Top-k fitness
        all_topk_fitness: list[float] = []

        all_rounds = []

        for round_num in range(1, self.n_rounds + 1):
            round_start = time.time()
            print(f"\n{'='*60}")
            print(f"闭环 DBTL 第 {round_num} 轮")
            print(f"{'='*60}")
            print(f"  训练数据量: {len(train_df)}")
            print(f"  已评估候选: {len(evaluated_mutants)}")

            # === Step 1: Propose ===
            # 对测试集中未评估的候选打分
            available_test = self.test_df[
                ~self.test_df["mutant"].isin(evaluated_mutants)
            ].copy()

            if len(available_test) < self.topk:
                print(f"  候选不足（{len(available_test)} < {self.topk}），重置已评估")
                evaluated_mutants.clear()
                available_test = self.test_df.copy()

            print(f"  可用候选: {len(available_test)}")

            # 模型打分
            if self.acquisition in ("ucb", "ei"):
                # 主动学习：用不确定性估计
                mean_preds, std_preds = self.model.predict_with_uncertainty(available_test)
                if self.acquisition == "ucb":
                    acq_scores = mean_preds + self.ucb_beta * std_preds
                else:  # ei
                    best_so_far = float(train_df["DMS_score"].max())
                    from scipy.stats import norm
                    improvements = mean_preds - best_so_far
                    z = improvements / np.maximum(std_preds, 1e-6)
                    ei = improvements * norm.cdf(z) + std_preds * norm.pdf(z)
                    acq_scores = ei
                available_test["pred_score"] = acq_scores
            else:
                # 纯预测均值
                preds = self.model.predict(available_test)
                available_test["pred_score"] = preds

            # 选 Top-k
            topk_df = available_test.nlargest(self.topk, "pred_score")
            print(f"  选出 Top-{self.topk} 候选")

            # === Step 2: Evaluate ===
            # 用 VirtualLab 评分（真实 fitness）
            topk_mutants = topk_df["mutant"].tolist()
            topk_real_fitness = []
            for mutant in topk_mutants:
                score_result = virtual_lab.score(mutant)
                topk_real_fitness.append(score_result.fitness)
                evaluated_mutants.add(mutant)

            topk_mean = float(np.mean(topk_real_fitness))
            topk_max = float(np.max(topk_real_fitness))

            # 累计最优
            all_topk_fitness.extend(topk_real_fitness)
            cumulative_topk = sorted(all_topk_fitness, reverse=True)[:self.topk]
            cumulative_mean = float(np.mean(cumulative_topk))

            print(f"  Top-k 真实 fitness: mean={topk_mean:.3f}, max={topk_max:.3f}")
            print(f"  累计最优 Top-k mean: {cumulative_mean:.3f}")

            # === Step 3: Update ===
            # Top-k + 真实 fitness 加入训练集
            new_rows = []
            for mutant, fitness in zip(topk_mutants, topk_real_fitness):
                # 推断 mutated_sequence
                mutations = parse_mutant_str(mutant, WT_SEQUENCE)
                seq_chars = list(FULL_LENGTH_SEQUENCE)
                for m in mutations:
                    if 0 <= m.position < len(seq_chars):
                        seq_chars[m.position] = m.mut_aa
                new_rows.append({
                    "mutant": mutant,
                    "mutated_sequence": "".join(seq_chars),
                    "DMS_score": fitness,
                    "n_mutations": len(mutations) if mutations else 1,
                })

            if new_rows:
                train_df = pd.concat([train_df, pd.DataFrame(new_rows)], ignore_index=True)
                print(f"  训练集扩充: +{len(new_rows)} → {len(train_df)}")

            # === Step 4: Learn (重新训练模型) ===
            print(f"  重新训练模型 ...")
            self.model.train(train_df)

            # 评估模型在剩余测试集上的性能
            remaining_test = self.test_df[
                ~self.test_df["mutant"].isin(evaluated_mutants)
            ].copy()
            if len(remaining_test) > 0:
                y_true = remaining_test["DMS_score"].values
                y_pred = self.model.predict(remaining_test)
                metrics = evaluate_all(y_true, y_pred, k=min(10, len(remaining_test)))
            else:
                metrics = {}

            round_time = time.time() - round_start

            round_result = {
                "round": round_num,
                "train_size": len(train_df),
                "evaluated": len(evaluated_mutants),
                "topk_mutants": topk_mutants,
                "topk_real_fitness": topk_real_fitness,
                "topk_mean": topk_mean,
                "topk_max": topk_max,
                "cumulative_topk_mean": cumulative_mean,
                "model_spearman": metrics.get("spearman", 0),
                "model_topk_hit": metrics.get(f"topk_hit@{min(10, len(remaining_test))}", 0),
                "acquisition": self.acquisition,
                "round_time_sec": round(round_time, 1),
            }
            all_rounds.append(round_result)

            print(f"  模型 Spearman: {metrics.get('spearman', 0):.4f}")
            print(f"  轮次耗时: {round_time:.1f}s")

            # 保存本轮结果
            round_file = results_path / f"round_{round_num}.json"
            with open(round_file, "w", encoding="utf-8") as f:
                json.dump(round_result, f, ensure_ascii=False, indent=2, default=str)

        # 保存汇总
        summary = {
            "n_rounds": self.n_rounds,
            "acquisition": self.acquisition,
            "topk": self.topk,
            "rounds": [
                {
                    "round": r["round"],
                    "train_size": r["train_size"],
                    "topk_mean": r["topk_mean"],
                    "topk_max": r["topk_max"],
                    "cumulative_topk_mean": r["cumulative_topk_mean"],
                    "model_spearman": r["model_spearman"],
                    "round_time_sec": r["round_time_sec"],
                }
                for r in all_rounds
            ],
        }
        summary_file = results_path / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print("闭环 DBTL 迭代完成")
        print(f"{'='*60}")
        print(f"  方法: {self.acquisition}")
        for r in all_rounds:
            print(f"  R{r['round']}: topk_mean={r['topk_mean']:.3f}, "
                  f"cumulative={r['cumulative_topk_mean']:.3f}, "
                  f"spearman={r['model_spearman']:.4f}")
        print(f"  结果保存至: {results_path}")

        return all_rounds
