"""

方法：
- random: 随机突变
- model_only: 模型直接推荐
- llm_agent: LLM Agent 无知识库
- llm_agent_kb: LLM Agent 知识增强
每方法跑 N 轮，记录每轮 Top-k 真实 fitness。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ..agent.orchestrator import AgentOrchestrator
from ..agent.llm_backend import LLMBackend, MockBackend
from ..data.wt import WT_SEQUENCE
from ..data.loader import parse_mutant_str
from ..model.fitness_model import FitnessModel
from .virtual_lab import VirtualLab
from .baselines import random_baseline, model_only_baseline


def run_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: FitnessModel,
    backend: Optional[LLMBackend] = None,
    methods: Optional[list[str]] = None,
    n_rounds: int = 3,
    n_candidates: int = 50,
    topk: int = 10,
    test_pool_size: int = 500,
    seed: int = 42,
    results_dir: str = "results/comparison",
) -> dict:

    if methods is None:
        methods = ["random", "model_only", "llm_agent", "llm_agent_kb"]

    backend = backend or MockBackend()
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # 构造 VirtualLab（test_pool 用于 random / model_only / Agent 验证）
    # test_pool 取全部测试集
    virtual_lab = VirtualLab.from_test_df(test_df, model, pool_size=test_pool_size, seed=seed)
    test_pool = virtual_lab.test_pool
    print(f"VirtualLab test_pool 大小: {len(test_pool)}（全部测试集）")

    all_results: dict = {}

    for method in methods:
        print(f"\n{'='*60}")
        print(f"方法: {method}")
        print(f"{'='*60}")

        method_results = []

        if method == "random":
            # 累计最优：跨轮迭代筛选（与 LLM Agent 同口径）
            all_rounds_fitness_random = []
            for r in range(1, n_rounds + 1):
                result = random_baseline(
                    train_df, test_pool,
                    n_candidates=n_candidates, topk=topk,
                    seed=seed, round_number=r,
                )
                # 计算累计最优
                round_fitness = result.topk_real_fitness
                all_rounds_fitness_random.extend(round_fitness)
                sorted_all = sorted(all_rounds_fitness_random, reverse=True)
                cumulative_topk = sorted_all[:topk]
                result.metrics["cumulative_topk_mean"] = float(sum(cumulative_topk) / len(cumulative_topk))
                result.metrics["cumulative_topk_max"] = float(max(all_rounds_fitness_random))
                method_results.append(result.to_dict())
                print(f"  Round {r}: topk_mean={result.metrics.get('topk_mean', 0):.3f}, "
                      f"cumulative={result.metrics.get('cumulative_topk_mean', 0):.3f}")

        elif method == "model_only":
            all_rounds_fitness_model = []
            for r in range(1, n_rounds + 1):
                result = model_only_baseline(
                    train_df, test_pool, model,
                    n_candidates=n_candidates, topk=topk,
                    seed=seed, round_number=r,
                )
                round_fitness = result.topk_real_fitness
                all_rounds_fitness_model.extend(round_fitness)
                sorted_all = sorted(all_rounds_fitness_model, reverse=True)
                cumulative_topk = sorted_all[:topk]
                result.metrics["cumulative_topk_mean"] = float(sum(cumulative_topk) / len(cumulative_topk))
                result.metrics["cumulative_topk_max"] = float(max(all_rounds_fitness_model))
                method_results.append(result.to_dict())
                print(f"  Round {r}: topk_mean={result.metrics.get('topk_mean', 0):.3f}, "
                      f"cumulative={result.metrics.get('cumulative_topk_mean', 0):.3f}")

        elif method in ("llm_agent", "llm_agent_kb"):
            use_kb = (method == "llm_agent_kb")
            orchestrator = AgentOrchestrator(
                model=model,
                backend=backend,
                use_knowledge=use_kb,
                n_candidates=n_candidates,
                topk=topk,
                seed=seed,
            )
            # 多轮迭代（每轮 Top-k 加入下一轮历史）
            # 关键：传入 candidate_pool=test_df，让 Agent 从测试集候选池中筛选
            multi_results = orchestrator.run_multi_round(
                initial_history_df=train_df,
                n_rounds=n_rounds,
                method=method,
                virtual_lab=virtual_lab,
                results_dir=str(results_path / method),
                candidate_pool=test_df,
            )
            for r in multi_results:
                method_results.append(r.to_dict())

        all_results[method] = method_results

        # 保存单方法结果
        method_file = results_path / f"{method}_results.json"
        with open(method_file, "w", encoding="utf-8") as f:
            json.dump(method_results, f, ensure_ascii=False, indent=2)

    # 生成汇总表
    summary = generate_summary(all_results)
    summary_file = results_path / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n汇总结果已保存至 {summary_file}")

    # 生成 CSV 汇总（含累计最优指标）
    csv_rows = []
    for method, rounds in all_results.items():
        for r_idx, r_data in enumerate(rounds, 1):
            metrics = r_data.get("metrics", {})
            csv_rows.append({
                "method": method,
                "round": r_idx,
                "topk_max": metrics.get("topk_max", 0),
                "topk_mean": metrics.get("topk_mean", 0),
                "topk_min": metrics.get("topk_min", 0),
                "cumulative_topk_mean": metrics.get("cumulative_topk_mean", 0),
                "cumulative_topk_max": metrics.get("cumulative_topk_max", 0),
            })
    summary_df = pd.DataFrame(csv_rows)
    csv_path = results_path / "comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"CSV 汇总已保存至 {csv_path}")

    return all_results


def generate_summary(all_results: dict) -> dict:
    """生成汇总统计。"""
    summary = {}
    for method, rounds in all_results.items():
        maxes = [r.get("metrics", {}).get("topk_max", 0) for r in rounds]
        means = [r.get("metrics", {}).get("topk_mean", 0) for r in rounds]
        summary[method] = {
            "n_rounds": len(rounds),
            "topk_max_per_round": maxes,
            "topk_mean_per_round": means,
            "best_topk_max": max(maxes) if maxes else 0,
            "best_topk_mean": max(means) if means else 0,
            "final_topk_mean": means[-1] if means else 0,
            "improvement": (means[-1] - means[0]) if len(means) >= 2 else 0,
        }
    return summary


if __name__ == "__main__":
    # 自测
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    model = FitnessModel.load("results/models/fitness_model.joblib")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"train={len(train_df)}, test={len(test_df)}")

    results = run_comparison(
        train_df=train_df,
        test_df=test_df,
        model=model,
        methods=["random", "model_only", "llm_agent", "llm_agent_kb"],
        n_rounds=3,
        n_candidates=30,
        topk=5,
        test_pool_size=100,
    )
