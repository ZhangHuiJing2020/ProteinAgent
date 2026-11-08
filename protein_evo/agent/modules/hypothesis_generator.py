"""Hypothesis Generator 模块。

基于 Data Analyst 报告，提出突变假设（哪些位点可能有益，建议哪些氨基酸）。
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from ...data.wt import WT_SEQUENCE, MUTATION_SITES, FULL_LENGTH_SEQUENCE
from ...knowledge.amino_acid_props import AA_PROPS, get_category, is_hydrophobic, is_charged, is_small
from ...knowledge.kg import KnowledgeGraph, query_high_fitness_mutations, query_mutations_at_position
from .data_analyst import AnalystReport
from ..llm_backend import LLMBackend, MockBackend
from ..types import Hypothesis, HypothesisReport


SYSTEM_PROMPT = """You are a Hypothesis Generator for protein directed evolution.

Your task: Based on the data analyst report (top variants, position frequencies, fitness stats),
generate mutation hypotheses about which positions may be beneficial and which amino acids to try.

Consider:
- Positions with high frequency in top variants
- Amino acid properties (hydrophobicity, charge, size, polarity)
- Conservative vs radical substitutions (BLOSUM scores)
- Combination strategy for multi-point mutations

Output strict JSON with fields:
- hypotheses: [{position (1-based), current_aa, suggested_aas: [string], rationale}]
- combination_strategy: string
- max_mutations: int
"""


def generate_hypotheses(
    analyst_report: AnalystReport,
    history_df: Optional[pd.DataFrame] = None,
    kg: Optional[KnowledgeGraph] = None,
    backend: Optional[LLMBackend] = None,
    max_mutations: int = 3,
) -> HypothesisReport:
    """生成突变假设。

    Args:
        analyst_report: Data Analyst 的输出
        history_df: 历史数据（用于查询单点突变 fitness）
        kg: 知识图谱（None 时不使用知识增强）
        backend: LLM 后端
        max_mutations: 单次最大突变数

    Returns:
        HypothesisReport
    """
    hypotheses: list[Hypothesis] = []

    # 规则 1：从 position_frequency 选高频位点
    pos_freq = analyst_report.position_frequency
    sorted_positions = sorted(pos_freq.items(), key=lambda x: -x[1])

    # 规则 2：从历史数据找各单点突变平均 fitness
    single_fitness = {}  # {(pos, aa): mean_fitness}
    if history_df is not None and len(history_df) > 0:
        single_df = history_df[history_df["n_mutations"] == 1].copy()
        for _, row in single_df.iterrows():
            mutant = str(row.get("mutant", ""))
            mutations = parse_mutant_str_safe(mutant)
            if len(mutations) == 1:
                m = mutations[0]
                key = (m.position + 1, m.mut_aa)
                single_fitness.setdefault(key, []).append(float(row["DMS_score"]))
        single_fitness = {k: sum(v) / len(v) for k, v in single_fitness.items()}

    # 规则 3：为每个候选位点推荐氨基酸
    # 优先级：历史高 fitness > 知识图谱推荐 > 理化性质相似
    for pos_1, count in sorted_positions[:4]:  # 最多 4 个位点
        pos_0 = pos_1 - 1
        # 用全长序列查找 WT 氨基酸（位点编号是全长蛋白编号）
        if pos_0 >= len(FULL_LENGTH_SEQUENCE):
            continue
        current_aa = FULL_LENGTH_SEQUENCE[pos_0]

        # 候选氨基酸优先级
        suggested = []

        # 3a. 历史中该位点的高 fitness 突变
        if history_df is not None:
            for (p, aa), fit in single_fitness.items():
                if p == pos_1 and fit > 0:
                    suggested.append(aa)

        # 3b. 知识图谱推荐（如有）
        if kg is not None:
            kg_mutations = query_mutations_at_position(kg, pos_1)
            for m_str in kg_mutations:
                # m_str 格式 'V39A'
                if len(m_str) >= 3:
                    aa = m_str[-1]
                    if aa != current_aa and aa not in suggested:
                        suggested.append(aa)

        # 3c. 理化性质相似替换（兜底）
        if len(suggested) < 3:
            current_cat = get_category(current_aa)
            for aa, props in AA_PROPS.items():
                if aa == current_aa or aa in suggested:
                    continue
                if props["category"] == current_cat:
                    suggested.append(aa)
                if len(suggested) >= 5:
                    break

        # 限制 3 个建议
        suggested = suggested[:3]

        rationale_parts = [f"位点 {pos_1} 在 top variants 中出现 {count} 次"]
        rationale_parts.append(f"野生型 {current_aa} ({get_category(current_aa)})")
        if suggested:
            rationale_parts.append(f"建议尝试 {suggested}")

        hypotheses.append(Hypothesis(
            position=pos_1,
            current_aa=current_aa,
            suggested_aas=suggested,
            rationale="; ".join(rationale_parts),
        ))

    # 组合策略
    combination_strategy = "优先组合高频位点 + 历史高 fitness 单点突变"
    if kg is not None:
        combination_strategy += "（知识图谱增强）"

    # 调用 LLM 增强假设（可选）
    if backend is not None:
        try:
            user_data = {
                "analyst_summary": analyst_report.summary,
                "position_frequency": {str(k): v for k, v in pos_freq.items()},
                "top_variants": [
                    {"mutant": v.mutant, "fitness": v.fitness}
                    for v in analyst_report.top_variants[:5]
                ],
                "current_hypotheses": [h.__dict__ for h in hypotheses],
            }
            resp = backend.chat(
                system=SYSTEM_PROMPT,
                user="Generate hypotheses based on:\n```json\n" + json.dumps(user_data, ensure_ascii=False) + "\n```",
                response_format={"type": "json_object"},
            )
            llm_data = json.loads(resp.text)
            if "combination_strategy" in llm_data:
                combination_strategy = llm_data["combination_strategy"]
            # LLM 的 hypotheses 可能补充新的，但保留规则的为主
            if "hypotheses" in llm_data and isinstance(llm_data["hypotheses"], list):
                for h in llm_data["hypotheses"][:2]:  # 最多补充 2 个
                    if isinstance(h, dict) and "position" in h:
                        # 避免重复
                        if not any(x.position == h["position"] for x in hypotheses):
                            hypotheses.append(Hypothesis(
                                position=int(h["position"]),
                                current_aa=h.get("current_aa", "X"),
                                suggested_aas=h.get("suggested_aas", [])[:3],
                                rationale=h.get("rationale", "LLM 生成"),
                            ))
        except (json.JSONDecodeError, Exception):
            pass

    return HypothesisReport(
        hypotheses=hypotheses,
        combination_strategy=combination_strategy,
        max_mutations=max_mutations,
    )


def parse_mutant_str_safe(mutant_str: str):
    """安全解析 mutant 字符串（避免循环导入）。"""
    from ...data.loader import parse_mutant_str
    return parse_mutant_str(mutant_str, WT_SEQUENCE)


if __name__ == "__main__":
    # 自测
    from ..types import VariantInfo
    df = pd.DataFrame([
        {"mutant": "V39A", "mutated_sequence": "", "DMS_score": 0.8, "n_mutations": 1},
        {"mutant": "V54I", "mutated_sequence": "", "DMS_score": 0.7, "n_mutations": 1},
        {"mutant": "V39A:V54I", "mutated_sequence": "", "DMS_score": 1.5, "n_mutations": 2},
    ])
    report = AnalystReport(
        summary="test",
        top_variants=[
            VariantInfo(mutant="V39A:V54I", mutated_sequence="", fitness=1.5, n_mutations=2),
            VariantInfo(mutant="V39A", mutated_sequence="", fitness=0.8, n_mutations=1),
            VariantInfo(mutant="V54I", mutated_sequence="", fitness=0.7, n_mutations=1),
        ],
        position_frequency={39: 2, 54: 2},
        fitness_stats={"max": 1.5, "mean": 1.0},
    )
    hyp = generate_hypotheses(report, history_df=df, backend=MockBackend())
    print(f"组合策略: {hyp.combination_strategy}")
    for h in hyp.hypotheses:
        print(f"  位点 {h.position} ({h.current_aa}→{h.suggested_aas}): {h.rationale}")
