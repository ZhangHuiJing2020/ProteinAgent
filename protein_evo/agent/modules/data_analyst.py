"""Data Analyst 模块。

读取当前实验数据，分析 top variants，总结哪些突变位点可能有益。
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Optional

import pandas as pd

from ...data.loader import parse_mutant_str
from ...data.wt import WT_SEQUENCE
from ..llm_backend import LLMBackend, MockBackend
from ..types import AnalystReport, VariantInfo


SYSTEM_PROMPT = """You are a Data Analyst for protein directed evolution.

Your task: Analyze the current experimental data and top variants to identify:
1. Which mutation positions appear frequently in high-fitness variants
2. The distribution of mutations (single/double/triple)
3. Fitness statistics (max, mean, std)
4. Key insights about beneficial mutations

Output strict JSON with fields:
- summary: string
- top_positions: [{position, count}]
- mutation_count_distribution: {single, double, triple_plus}
- fitness_stats: {max, mean, std}
- insights: [string]
"""


def analyze_data(
    history_df: pd.DataFrame,
    top_n: int = 20,
    backend: Optional[LLMBackend] = None,
) -> AnalystReport:
    """分析历史数据，识别 top variants 与有益位点。

    Args:
        history_df: 历史/当前实验数据，含 mutant, mutated_sequence, DMS_score, n_mutations
        top_n: 取 fitness 最高的 N 个变体作为 top variants
        backend: LLM 后端（None 时用纯规则分析）

    Returns:
        AnalystReport
    """
    if len(history_df) == 0:
        return AnalystReport(summary="无历史数据")

    # 按 fitness 排序取 top_n
    top_df = history_df.nlargest(top_n, "DMS_score")

    # 构造 VariantInfo 列表
    top_variants = []
    for _, row in top_df.iterrows():
        top_variants.append(VariantInfo(
            mutant=str(row.get("mutant", "")),
            mutated_sequence=str(row.get("mutated_sequence", "")),
            fitness=float(row["DMS_score"]),
            n_mutations=int(row.get("n_mutations", 1)),
        ))

    # 位点频率统计（基于 top variants）
    position_counter = Counter()
    for v in top_variants:
        mutations = parse_mutant_str(v.mutant, WT_SEQUENCE)
        for m in mutations:
            position_counter[m.position + 1] += 1  # 1-based

    # fitness 统计（基于全部历史数据）
    fitness_values = history_df["DMS_score"].values
    fitness_stats = {
        "max": float(fitness_values.max()),
        "mean": float(fitness_values.mean()),
        "std": float(fitness_values.std()),
        "n_total": int(len(history_df)),
    }

    # 突变数分布
    n_mut_dist = {
        "single": int((history_df["n_mutations"] == 1).sum()),
        "double": int((history_df["n_mutations"] == 2).sum()),
        "triple_plus": int((history_df["n_mutations"] >= 3).sum()),
    }

    # 规则派生的 insights
    insights = []
    if position_counter:
        top_pos = position_counter.most_common(3)
        insights.append(f"Top variants 中位点 {top_pos[0][0]} 出现频率最高 ({top_pos[0][1]} 次)")
        if len(top_pos) > 1:
            insights.append(f"次高频位点 {top_pos[1][0]} ({top_pos[1][1]} 次)")
    if fitness_stats["max"] > fitness_stats["mean"] * 1.5:
        insights.append("Top variant fitness 显著高于均值，提示存在强信号")
    if n_mut_dist["double"] > 0:
        insights.append("存在双点突变样本，可分析协同效应")

    # 调用 LLM 生成更深入的分析（可选）
    summary = f"分析了 {len(history_df)} 条历史数据，识别 top {len(top_variants)} variants"
    if backend is not None:
        # 准备 user prompt
        user_data = {
            "top_variants": [
                {"mutant": v.mutant, "fitness": v.fitness, "n_mutations": v.n_mutations}
                for v in top_variants[:10]
            ],
            "position_frequency": dict(position_counter),
            "fitness_stats": fitness_stats,
            "n_mutations_distribution": n_mut_dist,
        }
        try:
            resp = backend.chat(
                system=SYSTEM_PROMPT,
                user="Analyze the following data:\n```json\n" + json.dumps(user_data, ensure_ascii=False) + "\n```",
                response_format={"type": "json_object"},
            )
            # 尝试解析 LLM 返回的 JSON
            llm_data = json.loads(resp.text)
            # 用 LLM 的 summary 覆盖，insights 合并
            if "summary" in llm_data:
                summary = llm_data["summary"]
            if "insights" in llm_data and isinstance(llm_data["insights"], list):
                insights.extend(llm_data["insights"][:3])  # 限制数量避免冗长
        except (json.JSONDecodeError, Exception):
            pass  # LLM 解析失败时用规则结果

    return AnalystReport(
        summary=summary,
        top_variants=top_variants,
        position_frequency=dict(position_counter),
        fitness_stats=fitness_stats,
        insights=insights,
    )


if __name__ == "__main__":
    # 自测
    df = pd.DataFrame([
        {"mutant": "V39A", "mutated_sequence": "", "DMS_score": 0.8, "n_mutations": 1},
        {"mutant": "V54I", "mutated_sequence": "", "DMS_score": 0.7, "n_mutations": 1},
        {"mutant": "V39A:V54I", "mutated_sequence": "", "DMS_score": 1.5, "n_mutations": 2},
        {"mutant": "D40E", "mutated_sequence": "", "DMS_score": -0.2, "n_mutations": 1},
    ])
    report = analyze_data(df, top_n=3)
    print(f"Summary: {report.summary}")
    print(f"Position frequency: {report.position_frequency}")
    print(f"Insights: {report.insights}")

    print("\n--- 用 MockBackend ---")
    report = analyze_data(df, top_n=3, backend=MockBackend())
    print(f"Summary: {report.summary}")
    print(f"Insights: {report.insights}")
