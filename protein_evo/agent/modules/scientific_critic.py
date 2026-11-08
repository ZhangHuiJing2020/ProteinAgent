"""Scientific Critic 模块。

检查突变是否合理，给出推荐理由，输出 Top-k。
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd

from ...knowledge.mutation_rules import validate_mutant
from ...knowledge.amino_acid_props import get_category, is_hydrophobic, is_charged
from ...knowledge.kg import KnowledgeGraph, query_high_fitness_mutations
from ...data.loader import parse_mutant_str
from ...data.wt import WT_SEQUENCE
from ..llm_backend import LLMBackend, MockBackend
from ..types import Candidate, CriticReport


SYSTEM_PROMPT = """You are a Scientific Critic for protein directed evolution.

Your task: Review scored candidate mutations and produce final Top-k recommendations with rationale.

For each recommendation, explain:
1. Why this mutation is selected (predicted fitness, conservative substitution, synergy)
2. Which positions are involved and their biological rationale
3. Risk factors (radical substitution, too many mutations, novelty)

Output strict JSON with fields:
- recommendations: [{mutant, rank, predicted_fitness, rationale, category_scores: {conservative, novelty, synergy}}]
- overall_assessment: string
- rejected_count: int
"""


def critique_and_rank(
    candidates: list[Candidate],
    topk: int = 10,
    history_df: Optional[pd.DataFrame] = None,
    kg: Optional[KnowledgeGraph] = None,
    backend: Optional[LLMBackend] = None,
) -> CriticReport:
    """对候选打分排序并给出推荐理由。

    Args:
        candidates: 已打分的候选列表（含 predicted_fitness）
        topk: 取前 K 个推荐
        history_df: 历史数据
        kg: 知识图谱（用于推荐理由引用）
        backend: LLM 后端

    Returns:
        CriticReport
    """
    if not candidates:
        return CriticReport(overall_assessment="无候选")

    # 取前 topk
    top_candidates = candidates[:topk]

    recommendations = []
    for rank, c in enumerate(top_candidates, 1):
        # 构造推荐理由
        rationale_parts = []
        if c.predicted_fitness is not None:
            rationale_parts.append(f"预测 fitness = {c.predicted_fitness:.3f}")

        # 突变分析
        mutations = parse_mutant_str(c.mutant, WT_SEQUENCE)
        if c.n_mutations == 1:
            rationale_parts.append("单点突变，风险低")
        elif c.n_mutations == 2:
            rationale_parts.append("双点突变，可能存在协同效应")
        else:
            rationale_parts.append(f"{c.n_mutations} 点突变，需关注上位效应")

        # 位点分析
        positions = [m.position + 1 for m in mutations]
        # 动态从 MUTATION_SITES 检查协同位点对
        from ...data.wt import MUTATION_SITES
        site_list = sorted(MUTATION_SITES.keys())  # 0-based
        for p1_0, p2_0 in zip(site_list[:-1], site_list[1:]):
            p1_1, p2_1 = p1_0 + 1, p2_0 + 1
            if p1_1 in positions and p2_1 in positions:
                wt1 = MUTATION_SITES[p1_0]
                wt2 = MUTATION_SITES[p2_0]
                from ...knowledge.amino_acid_props import get_category
                cat1 = get_category(wt1)
                cat2 = get_category(wt2)
                if cat1 == cat2:
                    rationale_parts.append(f"位点 {p1_1}-{p2_1} 协同（{cat1}位点对）")
        # 检查电荷位点
        for p1_0, aa in MUTATION_SITES.items():
            from ...knowledge.amino_acid_props import is_charged
            if is_charged(aa) and (p1_0 + 1) in positions:
                rationale_parts.append(f"位点 {p1_0 + 1} 电荷替换")

        # 保守性
        if c.validation:
            cons_score = c.validation.get("conservative_score", 0)
            if cons_score > 0.5:
                rationale_parts.append(f"保守替换 (score={cons_score:.2f})")
            elif cons_score > 0:
                rationale_parts.append(f"半保守替换 (score={cons_score:.2f})")

        # 知识图谱引用
        if kg is not None:
            high_fit_muts = set(query_high_fitness_mutations(kg))
            for m in mutations:
                single_str = f"{m.wt_aa}{m.position+1}{m.mut_aa}"
                if single_str in high_fit_muts:
                    rationale_parts.append(f"{single_str} 为历史高 fitness 突变")

        # 评分维度
        category_scores = {
            "conservative": c.validation.get("conservative_score", 0) if c.validation else 0,
            "novelty": c.validation.get("novelty_score", 0.5) if c.validation else 0.5,
            "synergy": _estimate_synergy(mutations),
        }

        recommendations.append({
            "mutant": c.mutant,
            "rank": rank,
            "predicted_fitness": c.predicted_fitness,
            "rationale": "; ".join(rationale_parts),
            "category_scores": category_scores,
        })

    # 调用 LLM 生成整体评估
    overall_assessment = f"推荐 {len(recommendations)} 个候选，预测 fitness 范围 [{top_candidates[-1].predicted_fitness:.2f}, {top_candidates[0].predicted_fitness:.2f}]"
    if kg is not None:
        overall_assessment += "（知识图谱增强）"

    if backend is not None:
        try:
            user_data = {
                "top_candidates": [
                    {"mutant": c.mutant, "predicted_fitness": c.predicted_fitness,
                     "n_mutations": c.n_mutations}
                    for c in top_candidates[:5]
                ],
                "history_available": history_df is not None,
                "kg_available": kg is not None,
            }
            resp = backend.chat(
                system=SYSTEM_PROMPT,
                user="Critique and rank:\n```json\n" + json.dumps(user_data, ensure_ascii=False) + "\n```",
                response_format={"type": "json_object"},
            )
            llm_data = json.loads(resp.text)
            if "overall_assessment" in llm_data:
                overall_assessment = llm_data["overall_assessment"]
        except (json.JSONDecodeError, Exception):
            pass

    rejected_count = len(candidates) - len(top_candidates)
    return CriticReport(
        recommendations=recommendations,
        overall_assessment=overall_assessment,
        rejected_count=rejected_count,
    )


def _estimate_synergy(mutations) -> float:
    """估算协同效应得分（基于相邻位点对）。"""
    if len(mutations) < 2:
        return 0.0
    # 动态从 MUTATION_SITES 构建协同位点对（相邻位点）
    from ...data.wt import MUTATION_SITES
    site_list = sorted(MUTATION_SITES.keys())  # 0-based
    synergy_pairs = set()
    for p1_0, p2_0 in zip(site_list[:-1], site_list[1:]):
        synergy_pairs.add((min(p1_0, p2_0), max(p1_0, p2_0)))
    positions = [m.position for m in mutations]
    score = 0.0
    for p1, p2 in zip(positions[:-1], positions[1:]):
        pair = (min(p1, p2), max(p1, p2))
        if pair in synergy_pairs:
            score += 0.3
    return min(score, 1.0)


if __name__ == "__main__":
    # 自测
    cands = [
        Candidate(mutant="V39A:V54I", n_mutations=2, predicted_fitness=1.2,
                  validation={"is_valid": True, "conservative_score": 0.8, "novelty_score": 0.5}),
        Candidate(mutant="V39A", n_mutations=1, predicted_fitness=0.8,
                  validation={"is_valid": True, "conservative_score": 0.7, "novelty_score": 0.5}),
    ]
    report = critique_and_rank(cands, topk=2, backend=MockBackend())
    print(f"评估: {report.overall_assessment}")
    for r in report.recommendations:
        print(f"  #{r['rank']} {r['mutant']}: {r['rationale']}")
