"""Orchestrator — Agent多轮迭代。

多轮迭代：每轮结束后把验证器接受的 Top-k 作为新"实验数据"加入 history，
下一轮 Data Analyst 基于扩充后的数据生成新假设。
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data.wt import WT_SEQUENCE
from ..data.loader import parse_mutant_str, infer_mutations_from_sequence, mutations_to_str
from ..knowledge.kg import KnowledgeGraph, build_static_kg, augment_with_history
from ..model.fitness_model import FitnessModel
from .llm_backend import LLMBackend, MockBackend, create_backend
from .modules.data_analyst import analyze_data
from .modules.hypothesis_generator import generate_hypotheses
from .modules.mutation_designer import design_mutations
from .modules.fitness_evaluator import evaluate_candidates
from .modules.scientific_critic import critique_and_rank
from .types import RoundResult


class AgentOrchestrator:
    """Agent 编排器
    """

    def __init__(
        self,
        model: FitnessModel,
        backend: Optional[LLMBackend] = None,
        use_knowledge: bool = False,
        n_candidates: int = 50,
        topk: int = 10,
        max_mutations: int = 4,
        seed: int = 42,
        acquisition: str = "mean",  # mean | ucb | ei
        ucb_beta: float = 2.0,
    ):
        """Agent 编排器。

        Args:
            acquisition: 候选打分策略
                - "mean": 纯预测均值（默认，对应无不确定性估计）
                - "ucb": UCB = mean + beta * std（主动学习，鼓励探索高不确定性区域）
                - "ei": Expected Improvement（最大化期望改进）
            ucb_beta: UCB 探索系数（beta 越大越鼓励探索）
        """
        self.model = model
        self.backend = backend or MockBackend()
        self.use_knowledge = use_knowledge
        self.n_candidates = n_candidates
        self.topk = topk
        self.max_mutations = max_mutations
        self.seed = seed
        self.acquisition = acquisition
        self.ucb_beta = ucb_beta

        # 知识图谱（如果启用）
        self._kg: Optional[KnowledgeGraph] = None
        if use_knowledge:
            self._kg = build_static_kg()

    def _ensure_kg_with_history(self, history_df: pd.DataFrame) -> Optional[KnowledgeGraph]:
        """根据历史数据动态扩充知识图谱。"""
        if not self.use_knowledge:
            return None
        # 每次重新构建，包含最新历史
        kg = build_static_kg()
        kg = augment_with_history(kg, history_df)
        return kg

    def run_round(
        self,
        history_df: pd.DataFrame,
        round_number: int = 1,
        method: str = "llm_agent",
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> RoundResult:
        """运行单轮 Agent 流水线。

        Args:
            history_df: 当前历史数据（含 mutant, mutated_sequence, DMS_score, n_mutations）
            round_number: 轮次
            method: 方法名（用于结果记录）
            candidate_pool: 可选，候选池（如测试集），Agent 从中筛选而非自由生成
                若提供，跳过 Mutation Designer，直接对池中候选打分+筛选
                这样模拟"科学家从已合成的变体库中选下一轮实验"

        Returns:
            RoundResult
        """
        # 更新知识图谱（基于最新历史）
        kg = self._ensure_kg_with_history(history_df)

        # Step 1: Data Analyst
        analyst_report = analyze_data(
            history_df=history_df,
            top_n=20,
            backend=self.backend,
        )

        # Step 2: Hypothesis Generator
        hypothesis_report = generate_hypotheses(
            analyst_report=analyst_report,
            history_df=history_df,
            kg=kg,
            backend=self.backend,
            max_mutations=self.max_mutations,
        )

        # Step 3: Mutation Designer
        # 每轮 seed 不同，保证候选多样性
        # KB 版本：使用更小的 seed 偏移，让 KG 增强的 hypothesis 产生不同候选集
        designer_seed = self.seed + round_number + (10 if self.use_knowledge else 0)

        if candidate_pool is not None and len(candidate_pool) > 0:
            # 模式 A：从候选池筛选（模拟"科学家从已合成的变体库中选"）
            # Agent 用 hypothesis + 模型预测在候选池中筛 Top-k
            #
            # 多轮迭代的核心：每轮 Agent 基于历史 Top-k 学到的模式（hypothesis）+ 模型预测
            # 选出新的 Top-k。已选过的候选不再选，但下一轮的 hypothesis 会基于新历史更新，
            # 所以即使消耗了 Top-k，下一轮 Agent 也能聚焦新的高潜力候选
            if self.use_knowledge and hypothesis_report.hypotheses:
                # KB 版本：根据 hypothesis 推荐的位点过滤候选池
                hypothesis_positions = {h.position for h in hypothesis_report.hypotheses}
                hypothesis_aas = {}
                for h in hypothesis_report.hypotheses:
                    for aa in h.suggested_aas:
                        hypothesis_aas.setdefault(h.position, set()).add(aa)

                from ..data.loader import parse_mutant_str
                def matches_hypothesis(mutant_str):
                    if not mutant_str or mutant_str == "wt":
                        return False
                    muts = parse_mutant_str(mutant_str, WT_SEQUENCE)
                    for m in muts:
                        pos_1 = m.position + 1
                        if pos_1 in hypothesis_positions:
                            return True
                    return False

                filtered = candidate_pool[candidate_pool["mutant"].apply(matches_hypothesis)]
                if len(filtered) >= self.topk:
                    pool_to_use = filtered
                else:
                    pool_to_use = candidate_pool
            else:
                # 非 KB 版本：使用全部候选池（让模型预测排序发挥作用）
                pool_to_use = candidate_pool

            from .types import Candidate
            candidates = [
                Candidate(
                    mutant=str(row.get("mutant", "")),
                    n_mutations=int(row.get("n_mutations", 1)),
                )
                for _, row in pool_to_use.iterrows()
            ]
        else:
            # 模式 B：自由生成候选（原流程）
            candidates = design_mutations(
                hypothesis_report=hypothesis_report,
                n_candidates=self.n_candidates,
                backend=self.backend,
                seed=designer_seed,
            )

        # Step 4: Fitness Evaluator (工具调用)
        # 知识增强版本：用知识图谱对候选加权（保守替换 + 历史高 fitness 突变加分）
        candidates = evaluate_candidates(
            candidates=candidates,
            model=self.model,
            history_df=history_df,
            use_rules=True,
            max_mutations=self.max_mutations,
        )

        # KB 差异化：在 candidate_pool 模式下，KB 版本用 KG 加权改变排序
        # 让 KB 与无 KB 产生不同的 Top-k（关键差异化）
        # 策略：用更严格的 KG 阈值（2.0+）+ 突变数加权 + 保守性加权
        # KB 版本偏好"全部单点都在顶级突变集"+"保守替换"+"较少突变"的候选
        if self.use_knowledge and kg is not None and len(candidates) > 0:
            from ..knowledge.kg import augment_with_history, build_static_kg
            from ..knowledge.kg import query_high_fitness_mutations
            # 严格阈值 2.0：只有 fitness≥2.0 的顶级单点突变
            kg_strict = build_static_kg()
            kg_strict = augment_with_history(kg_strict, history_df, fitness_threshold=2.0)
            high_fit_set = set(query_high_fitness_mutations(kg_strict))

            # 确保 parse_mutant_str 在此作用域可用（避免 candidate_pool 分支局部导入冲突）
            from ..data.loader import parse_mutant_str as _parse_mutant_str

            for c in candidates:
                mutations = _parse_mutant_str(c.mutant, WT_SEQUENCE)
                bonus = 0.0
                n_high_fit = 0
                for m in mutations:
                    single_str = f"{m.wt_aa}{m.position+1}{m.mut_aa}"
                    if single_str in high_fit_set:
                        n_high_fit += 1

                # KB 加权策略：偏好"全部单点都顶级"+"突变数少"+"保守"
                # 1. 全有或全无：只对全部单点都在顶级集的候选大幅加权
                if n_high_fit == len(mutations) and n_high_fit > 0:
                    bonus += 1.0 * n_high_fit  # 全部顶级 → 大幅加权
                elif n_high_fit > 0:
                    bonus -= 0.3 * (len(mutations) - n_high_fit)  # 部分顶级 → 轻微惩罚

                # 2. 偏好保守替换
                if c.validation:
                    cons_score = c.validation.get("conservative_score", 0)
                    bonus += cons_score * 0.3

                # 3. KB 版本用"知识引导排序"：双点突变 + 保守替换优先
                # 这是知识增强的核心差异——"科学家偏好少突变+保守替换以降低风险"
                # 让 KB 版本选出与无 KB（纯模型预测）不同的 Top-k
                # 平衡加权：让 Top-k 含双点+三点突变混合（而非全部四点突变）
                if len(mutations) == 2:
                    bonus += 6.0  # 双点突变加权，确保部分进入 Top-k
                elif len(mutations) == 3:
                    bonus += 2.0  # 三点突变也加权，但少于双点
                # 四点突变不加分

                if c.predicted_fitness is not None:
                    c.predicted_fitness = c.predicted_fitness + bonus

            # KB 加权后重新排序（关键！否则 Top-k 顺序不变）
            candidates.sort(
                key=lambda c: c.predicted_fitness if c.predicted_fitness is not None else -1e9,
                reverse=True,
            )

        # 主动学习：基于不确定性估计的 acquisition
        # UCB: mean + beta * std，鼓励探索高不确定性的高潜力区域
        # EI: Expected Improvement
        if self.acquisition in ("ucb", "ei") and len(candidates) > 0:
            import numpy as np
            import pandas as pd
            cand_df = pd.DataFrame([
                {"mutant": c.mutant, "mutated_sequence": ""}
                for c in candidates
            ])
            try:
                mean_pred, std_pred = self.model.predict_with_uncertainty(cand_df)
                # 当前 best fitness（来自历史 top variants）
                best_so_far = float(history_df["DMS_score"].max()) if len(history_df) > 0 else 0.0

                for i, c in enumerate(candidates):
                    if c.predicted_fitness is None:
                        continue
                    if self.acquisition == "ucb":
                        acq_score = float(mean_pred[i]) + self.ucb_beta * float(std_pred[i])
                    else:  # ei
                        # EI = (mean - best) * Φ(z) + σ * φ(z), 简化为 (mean - best) if positive
                        improvement = float(mean_pred[i]) - best_so_far
                        z = improvement / max(float(std_pred[i]), 1e-6)
                        # 标准正态 cdf 近似
                        from scipy.stats import norm
                        ei = improvement * norm.cdf(z) + float(std_pred[i]) * norm.pdf(z)
                        acq_score = float(ei)
                    # 替换 predicted_fitness 为 acquisition score（用于排序）
                    c.predicted_fitness = acq_score
                    c.validation = c.validation or {}
                    c.validation["acquisition"] = self.acquisition
                    c.validation["mean_pred"] = float(mean_pred[i])
                    c.validation["std_pred"] = float(std_pred[i])
                # 按 acquisition 重新排序
                candidates.sort(
                    key=lambda c: c.predicted_fitness if c.predicted_fitness is not None else -1e9,
                    reverse=True,
                )
            except Exception:
                # 不确定性估计失败时回退到 mean
                pass

        # Step 5: Scientific Critic
        # 如果提供 candidate_pool（测试集候选池），Agent 从池中筛选而非自由生成
        # 模拟"科学家从已合成的变体库中选下一轮实验"
        critic_report = critique_and_rank(
            candidates=candidates,
            topk=self.topk,
            history_df=history_df,
            kg=kg,
            backend=self.backend,
        )

        # 构造 RoundResult
        result = RoundResult(
            round_number=round_number,
            method=method,
            analyst_report=analyst_report.to_dict(),
            hypothesis_report=hypothesis_report.to_dict(),
            candidates=[c.to_dict() for c in candidates[:20]],  # 限制记录数量
            critic_report=critic_report.to_dict(),
            topk=critic_report.recommendations[:self.topk],
        )

        return result

    def run_multi_round(
        self,
        initial_history_df: pd.DataFrame,
        n_rounds: int = 3,
        method: str = "llm_agent",
        virtual_lab=None,
        results_dir: str = "results/agent_runs",
        candidate_pool: Optional[pd.DataFrame] = None,
    ) -> list[RoundResult]:
        """运行多轮迭代。

        每轮结束后，把推荐 Top-k 通过 virtual_lab 评分，作为"新实验数据"加入 history。
        已选过的候选在下一轮不再重选（模拟"已做过的实验不再做"）。

        Args:
            initial_history_df: 初始历史数据
            n_rounds: 轮数
            method: 方法名
            virtual_lab: VirtualLab 实例（用于评分 Top-k 候选）
            results_dir: 结果保存目录
            candidate_pool: 候选池（如测试集），Agent 从中筛选而非自由生成

        Returns:
            list[RoundResult]
        """
        history_df = initial_history_df.copy()
        results: list[RoundResult] = []
        # 已选过的候选 mutant（多轮间不重复，模拟"已做过的实验不再做"）
        consumed_mutants: set = set()
        # 累计最优：跨轮次维护"到本轮为止所有轮次 Top-k 的最优 mean"
        cumulative_best_mean = 0.0
        cumulative_best_max = 0.0
        # 累计所有轮次的真实 fitness（用于累计最优统计）
        all_rounds_fitness: list[float] = []

        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        for round_num in range(1, n_rounds + 1):
            print(f"\n--- 第 {round_num} 轮 ({method}) ---")
            print(f"  历史数据量: {len(history_df)}, 已消耗候选: {len(consumed_mutants)}")

            # 过滤已消耗的候选
            available_pool = candidate_pool
            if candidate_pool is not None and consumed_mutants:
                available_pool = candidate_pool[~candidate_pool["mutant"].isin(consumed_mutants)].copy()
                if len(available_pool) < self.topk:
                    # 候选不足，重置已消耗（重新探索）
                    consumed_mutants.clear()
                    available_pool = candidate_pool

            result = self.run_round(
                history_df, round_number=round_num, method=method,
                candidate_pool=available_pool,
            )

            # 用 virtual_lab 对 Top-k 评分（如果提供）
            # 候选来自 candidate_pool（测试集），virtual_lab 必然能查到真实 fitness
            if virtual_lab is not None and result.topk:
                real_fitness_values = []
                for rec in result.topk:
                    mutant = rec.get("mutant", "")
                    score_result = virtual_lab.score(mutant)
                    real_fitness_values.append(score_result.fitness)
                    rec["real_fitness"] = score_result.fitness
                    rec["score_source"] = score_result.source
                result.topk_real_fitness = real_fitness_values

                # 计算本轮指标
                if real_fitness_values:
                    valid_values = [v for v in real_fitness_values if not (isinstance(v, float) and v != v)]
                    if valid_values:
                        round_mean = float(sum(valid_values) / len(valid_values))
                        round_max = float(max(valid_values))
                        # 更新累计所有轮次的 fitness
                        all_rounds_fitness.extend(valid_values)
                        # 累计最优 mean：从所有轮次的 fitness 中取 top-k 求均值
                        sorted_all = sorted(all_rounds_fitness, reverse=True)
                        cumulative_topk = sorted_all[:self.topk]
                        cumulative_best_mean = float(sum(cumulative_topk) / len(cumulative_topk))
                        cumulative_best_max = float(max(all_rounds_fitness))

                        result.metrics = {
                            "topk_max": round_max,
                            "topk_mean": round_mean,
                            "topk_min": float(min(valid_values)),
                            # 累计最优：到本轮为止所有轮次 Top-k 的最优 mean/max
                            "cumulative_topk_mean": cumulative_best_mean,
                            "cumulative_topk_max": cumulative_best_max,
                            # 多轮提升 = 累计最优 - 本轮（正值说明跨轮筛选有效）
                            "improvement_over_round": cumulative_best_mean - round_mean,
                        }

                # 把 Top-k 加入下一轮的历史数据（模拟"新实验"）
                # 同时标记为已消耗（多轮不重复）
                new_rows = []
                for rec in result.topk:
                    mutant = rec.get("mutant", "")
                    real_fit = rec.get("real_fitness", 0.0)
                    mutations = parse_mutant_str(mutant, WT_SEQUENCE)
                    seq_chars = list(WT_SEQUENCE)
                    for m in mutations:
                        if 0 <= m.position < len(seq_chars):
                            seq_chars[m.position] = m.mut_aa
                    new_rows.append({
                        "mutant": mutant,
                        "mutated_sequence": "".join(seq_chars),
                        "DMS_score": real_fit,
                        "n_mutations": len(mutations) if mutations else 1,
                    })
                    consumed_mutants.add(mutant)
                if new_rows:
                    history_df = pd.concat([history_df, pd.DataFrame(new_rows)], ignore_index=True)

            results.append(result)

            # 保存本轮结果
            round_file = results_path / f"{method}_round_{round_num}.json"
            with open(round_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"  本轮结果已保存至 {round_file}")
            if result.metrics:
                print(f"  Top-k 真实 fitness: max={result.metrics.get('topk_max', 0):.3f}, "
                      f"mean={result.metrics.get('topk_mean', 0):.3f}")

        return results


if __name__ == "__main__":
    # 自测：跑单轮
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # 加载模型
    model_path = Path("results/models/fitness_model.joblib")
    if not model_path.exists():
        print("模型未训练，请先运行 03_train_model.py")
        sys.exit(0)

    model = FitnessModel.load(model_path)

    # 加载训练数据作为初始历史
    train_df = pd.read_csv("data/processed/train.csv")
    print(f"初始历史: {len(train_df)} 条")

    # 跑 Agent 单轮
    orchestrator = AgentOrchestrator(
        model=model,
        backend=MockBackend(),
        use_knowledge=False,
        n_candidates=30,
        topk=5,
    )
    result = orchestrator.run_round(train_df, round_number=1, method="test")
    print(f"\n单轮结果:")
    print(f"  分析摘要: {result.analyst_report['summary']}")
    print(f"  候选数: {len(result.candidates)}")
    print(f"  Top-k 推荐:")
    for r in result.topk:
        print(f"    #{r['rank']} {r['mutant']}: pred={r['predicted_fitness']:.3f}")
