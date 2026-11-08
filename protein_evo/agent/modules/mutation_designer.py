"""Mutation Designer 模块。

基于假设生成候选突变序列。
"""
from __future__ import annotations

import itertools
import json
import random
from typing import Optional

from ...data.wt import WT_SEQUENCE, MUTATION_SITES, FULL_LENGTH_SEQUENCE
from ...knowledge.amino_acid_props import STANDARD_AAS
from ...knowledge.mutation_rules import validate_mutant
from .hypothesis_generator import HypothesisReport, generate_hypotheses
from ..llm_backend import LLMBackend, MockBackend
from ..types import Candidate


SYSTEM_PROMPT = """You are a Mutation Designer for protein directed evolution.

Your task: Based on the hypothesis report, generate candidate mutation sequences.

Constraints:
- Each candidate is a mutant string like 'Q228R' or 'Q228R:T279L'
- Positions must be in MUTATION_SITES (1-based, full-length protein G numbering)
- Max 4 mutations per variant
- Mix single, double, and triple point mutations
- Prioritize conservative substitutions
- Combine high-fitness single mutations from history

Output strict JSON with fields:
- candidates: [{mutant: string, n_mutations: int}]
- generation_strategy: string
"""


def design_mutations(
    hypothesis_report: HypothesisReport,
    n_candidates: int = 50,
    backend: Optional[LLMBackend] = None,
    seed: int = 42,
) -> list[Candidate]:
    """生成候选突变。

    Args:
        hypothesis_report: Hypothesis Generator 的输出
        n_candidates: 目标候选数
        backend: LLM 后端
        seed: 随机种子

    Returns:
        list[Candidate]
    """
    rng = random.Random(seed)
    candidates: list[Candidate] = []
    seen_mutants: set[str] = set()

    # 从假设提取位点+建议氨基酸
    pos_aas: dict[int, list[str]] = {}
    for h in hypothesis_report.hypotheses:
        if h.suggested_aas:
            pos_aas[h.position] = list(h.suggested_aas)

    # 如果假设为空，用 MUTATION_SITES 的全部位点 + 全部氨基酸兜底
    if not pos_aas:
        for pos_0 in sorted(MUTATION_SITES.keys()):
            pos_aas[pos_0 + 1] = list(STANDARD_AAS)  # 0-based → 1-based

    max_mut = hypothesis_report.max_mutations

    # 策略 1：单点突变（每个位点 × 建议氨基酸）
    for pos_1, aas in pos_aas.items():
        for aa in aas[:3]:  # 每个位点最多 3 个
            mutant = _build_mutant([(pos_1, aa)])
            if mutant and mutant not in seen_mutants:
                candidates.append(Candidate(mutant=mutant, n_mutations=1))
                seen_mutants.add(mutant)

    # 策略 2：双点突变（位点对 × 建议组合）
    positions = sorted(pos_aas.keys())
    for pos1, pos2 in itertools.combinations(positions, 2):
        for aa1 in pos_aas[pos1][:2]:
            for aa2 in pos_aas[pos2][:2]:
                mutant = _build_mutant([(pos1, aa1), (pos2, aa2)])
                if mutant and mutant not in seen_mutants:
                    candidates.append(Candidate(mutant=mutant, n_mutations=2))
                    seen_mutants.add(mutant)

    # 策略 3：三点突变（位点三元组 × 建议组合，限定数量）
    if len(positions) >= 3:
        for pos_tuple in itertools.combinations(positions, 3):
            for aa_tuple in itertools.product(*[pos_aas[p][:1] for p in pos_tuple]):
                mutant = _build_mutant(list(zip(pos_tuple, aa_tuple)))
                if mutant and mutant not in seen_mutants:
                    candidates.append(Candidate(mutant=mutant, n_mutations=3))
                    seen_mutants.add(mutant)

    # 策略 4：随机扩展到 n_candidates
    all_aas = list(STANDARD_AAS)
    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 5:
        attempts += 1
        n = rng.choice([1, 2, 3])
        chosen_positions = rng.sample(positions, min(n, len(positions)))
        muts = []
        for p in chosen_positions:
            # 用全长序列查找 WT 氨基酸（位点编号是全长蛋白编号）
            wt = FULL_LENGTH_SEQUENCE[p - 1] if p - 1 < len(FULL_LENGTH_SEQUENCE) else "X"
            new_aa = rng.choice([a for a in all_aas if a != wt])
            muts.append((p, new_aa))
        mutant = _build_mutant(muts)
        if mutant and mutant not in seen_mutants:
            candidates.append(Candidate(mutant=mutant, n_mutations=n))
            seen_mutants.add(mutant)

    # 策略 5：调用 LLM 补充（可选）
    if backend is not None and len(candidates) < n_candidates:
        try:
            user_data = {
                "hypotheses": [h.__dict__ for h in hypothesis_report.hypotheses],
                "max_mutations": hypothesis_report.max_mutations,
                "current_count": len(candidates),
                "need": n_candidates - len(candidates),
            }
            resp = backend.chat(
                system=SYSTEM_PROMPT,
                user="Generate candidates based on:\n```json\n" + json.dumps(user_data, ensure_ascii=False) + "\n```",
                response_format={"type": "json_object"},
            )
            llm_data = json.loads(resp.text)
            if "candidates" in llm_data and isinstance(llm_data["candidates"], list):
                for c in llm_data["candidates"]:
                    if isinstance(c, dict) and "mutant" in c:
                        mutant = c["mutant"]
                        if mutant not in seen_mutants:
                            # 简单校验
                            v = validate_mutant(mutant)
                            if v.is_valid:
                                candidates.append(Candidate(
                                    mutant=mutant,
                                    n_mutations=int(c.get("n_mutations", 1)),
                                ))
                                seen_mutants.add(mutant)
        except (json.JSONDecodeError, Exception):
            pass

    return candidates[:n_candidates]


def _build_mutant(pos_aa_pairs: list[tuple[int, str]]) -> str:
    """根据 (position_1based, new_aa) 列表构造 mutant 字符串。

    校验：
    - position 必须 1-based 在 [1, len(FULL_LENGTH_SEQUENCE)]
    - new_aa 必须是标准氨基酸且不同于野生型
    """
    parts = []
    for pos_1, new_aa in pos_aa_pairs:
        pos_0 = pos_1 - 1
        if pos_0 < 0 or pos_0 >= len(FULL_LENGTH_SEQUENCE):
            return ""
        wt_aa = FULL_LENGTH_SEQUENCE[pos_0]
        if new_aa not in STANDARD_AAS or new_aa == wt_aa:
            return ""
        parts.append(f"{wt_aa}{pos_1}{new_aa}")
    if not parts:
        return ""
    # 排序保证唯一性
    return ":".join(sorted(parts))


if __name__ == "__main__":
    # 自测
    from ..types import Hypothesis, HypothesisReport
    hyp = HypothesisReport(
        hypotheses=[
            Hypothesis(position=39, current_aa="V", suggested_aas=["A", "L", "I"], rationale="test"),
            Hypothesis(position=54, current_aa="V", suggested_aas=["I", "L"], rationale="test"),
        ],
        combination_strategy="test",
        max_mutations=3,
    )
    cands = design_mutations(hyp, n_candidates=30)
    print(f"生成 {len(cands)} 个候选:")
    for c in cands[:10]:
        print(f"  {c.mutant} (n={c.n_mutations})")
