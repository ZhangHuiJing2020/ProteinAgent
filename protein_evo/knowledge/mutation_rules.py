"""突变规则库 — 知识库 L2。

实现过滤/评分规则函数，用于 Agent 推荐时筛选候选突变。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .amino_acid_props import (
    AA_PROPS, STANDARD_AAS, is_standard, is_terminator,
    is_hydrophobic, is_charged, is_polar, is_small,
)
from ..features.blosum import blosum_score, BLOSUM62
from ..data.loader import parse_mutant_str, Mutation
from ..data.wt import WT_SEQUENCE, MUTATION_SITES


@dataclass
class RuleViolation:
    """规则违反记录。"""
    rule: str
    severity: str  # "error" | "warning"
    message: str


@dataclass
class MutationValidation:
    """突变验证结果。"""
    mutant_str: str
    is_valid: bool
    violations: list[RuleViolation]
    conservative_score: float  # 0..1，越高越保守
    novelty_score: float = 0.0  # 0..1，越高越新颖

    def to_dict(self) -> dict:
        return {
            "mutant_str": self.mutant_str,
            "is_valid": self.is_valid,
            "conservative_score": self.conservative_score,
            "novelty_score": self.novelty_score,
            "violations": [
                {"rule": v.rule, "severity": v.severity, "message": v.message}
                for v in self.violations
            ],
        }


def parse_mutations(mutant_str: str) -> list[Mutation]:
    """便利函数：解析 mutant 字符串。"""
    return parse_mutant_str(mutant_str, WT_SEQUENCE)


def check_terminator_or_nonstandard(mutations: list[Mutation]) -> list[RuleViolation]:
    """规则 1：禁止引入终止符或非标准氨基酸。"""
    violations = []
    for m in mutations:
        if is_terminator(m.mut_aa):
            violations.append(RuleViolation(
                rule="no_terminator",
                severity="error",
                message=f"位置 {m.position+1} 引入终止符 *",
            ))
        elif not is_standard(m.mut_aa):
            violations.append(RuleViolation(
                rule="standard_aa_only",
                severity="error",
                message=f"位置 {m.position+1} 引入非标准氨基酸 {m.mut_aa}",
            ))
    return violations


def check_mutation_count(
    mutations: list[Mutation],
    max_mutations: int = 4,
) -> list[RuleViolation]:
    """规则 2：单次突变数量上限。"""
    violations = []
    if len(mutations) > max_mutations:
        violations.append(RuleViolation(
            rule="max_mutations",
            severity="error",
            message=f"突变数 {len(mutations)} 超过上限 {max_mutations}",
        ))
    elif len(mutations) > max_mutations - 1:
        violations.append(RuleViolation(
            rule="many_mutations",
            severity="warning",
            message=f"突变数 {len(mutations)} 接近上限，上位效应风险",
        ))
    return violations


def check_conservative_substitution(mutations: list[Mutation]) -> tuple[float, list[RuleViolation]]:
    """规则 3：保守替换优先。

    返回 (conservative_score, violations)。conservative_score ∈ [0, 1]。
    """
    if not mutations:
        return 0.0, []
    scores = []
    violations = []
    for m in mutations:
        bscore = blosum_score(m.wt_aa, m.mut_aa)
        # 归一化到 [0, 1]，BLOSUM ∈ [-4, 4+]，>=0 为保守
        norm = max(0.0, min(1.0, bscore / 4.0))
        scores.append(norm)
        if bscore < 0:
            violations.append(RuleViolation(
                rule="conservative_substitution",
                severity="warning",
                message=f"位置 {m.position+1} 替换 {m.wt_aa}→{m.mut_aa} BLOSUM={bscore}（激进替换）",
            ))
    return float(np.mean(scores)), violations


def check_position_in_mutation_sites(mutations: list[Mutation]) -> list[RuleViolation]:
    """规则 4：突变应在已知的可突变位点（Olson 2014 实验位点）。

    对 GB1 来说，合法突变位点为 39/40/41/54（1-based）。
    Agent 生成的突变应该在这些位点上（否则超出实验范围）。
    """
    violations = []
    valid_sites = set(MUTATION_SITES.keys())  # 0-based
    for m in mutations:
        if m.position not in valid_sites:
            violations.append(RuleViolation(
                rule="position_in_experiment_sites",
                severity="warning",
                message=f"位置 {m.position+1} 不在实验位点 (39/40/41/54) 内",
            ))
    return violations


def check_wt_match(mutations: list[Mutation]) -> list[RuleViolation]:
    """规则 5：mutant 字符串中的 wt_aa 应与实际野生型匹配。"""
    violations = []
    for m in mutations:
        if 0 <= m.position < len(WT_SEQUENCE) and WT_SEQUENCE[m.position] != m.wt_aa:
            violations.append(RuleViolation(
                rule="wt_aa_match",
                severity="error",
                message=f"位置 {m.position+1} 野生型应为 {WT_SEQUENCE[m.position]}，但 mutant 写作 {m.wt_aa}",
            ))
    return violations


def check_historical_combination(
    mutations: list[Mutation],
    history_df: Optional[pd.DataFrame] = None,
    fitness_threshold: float = 0.5,
) -> tuple[float, list[RuleViolation]]:
    """规则 6：优先组合历史上表现较好的单点突变。

    计算单点突变是否在历史高 fitness 突变集中。
    返回 (novelty_score, violations)。
    novelty_score 越高表示越多"新组合"（不在历史中）。
    """
    if history_df is None or len(history_df) == 0:
        return 0.5, []

    # 提取历史高 fitness 单点突变
    single_mutations = history_df[history_df["n_mutations"] == 1].copy()
    if len(single_mutations) == 0:
        return 0.5, []

    high_fitness = single_mutations[single_mutations["DMS_score"] >= fitness_threshold]
    high_fitness_set = set(high_fitness["mutant"].tolist())

    # 检查候选突变中每个单点是否在历史高 fitness 集合中
    n_in_history = 0
    n_total = 0
    violations = []
    for m in mutations:
        single_str = f"{m.wt_aa}{m.position+1}{m.mut_aa}"
        n_total += 1
        if single_str in high_fitness_set:
            n_in_history += 1
        else:
            violations.append(RuleViolation(
                rule="historical_combination",
                severity="warning",
                message=f"单点 {single_str} 不在历史高 fitness 集合中",
            ))

    novelty = 1.0 - (n_in_history / max(n_total, 1))
    return novelty, violations


def validate_mutant(
    mutant_str: str,
    history_df: Optional[pd.DataFrame] = None,
    max_mutations: int = 4,
    fitness_threshold: float = 0.5,
) -> MutationValidation:
    """综合验证一个 mutant。

    返回 MutationValidation，包含 is_valid、违规列表、保守性分数、新颖性分数。
    """
    mutations = parse_mutations(mutant_str)

    violations: list[RuleViolation] = []
    violations.extend(check_terminator_or_nonstandard(mutations))
    violations.extend(check_mutation_count(mutations, max_mutations))
    conservative_score, cons_violations = check_conservative_substitution(mutations)
    violations.extend(cons_violations)
    violations.extend(check_position_in_mutation_sites(mutations))
    violations.extend(check_wt_match(mutations))
    novelty_score, hist_violations = check_historical_combination(
        mutations, history_df, fitness_threshold
    )
    violations.extend(hist_violations)

    # error 级别违规视为不通过
    has_error = any(v.severity == "error" for v in violations)

    return MutationValidation(
        mutant_str=mutant_str,
        is_valid=not has_error,
        violations=violations,
        conservative_score=conservative_score,
        novelty_score=novelty_score,
    )


def filter_candidates(
    mutant_strs: list[str],
    history_df: Optional[pd.DataFrame] = None,
    max_mutations: int = 4,
    allow_warnings: bool = True,
) -> list[MutationValidation]:
    """批量过滤候选突变列表。

    Returns:
        list[MutationValidation]
    """
    results = []
    for m in mutant_strs:
        v = validate_mutant(m, history_df, max_mutations)
        if v.is_valid or (allow_warnings and not any(x.severity == "error" for x in v.violations)):
            results.append(v)
    return results


if __name__ == "__main__":
    # 自测
    print("测试突变规则验证 ...")
    for mutant in ["V39A", "V39A:D40E", "V39A:G41A:V54I", "V39*", "V39X", "A39P:G40D:E41F:V54A:F55L"]:
        v = validate_mutant(mutant)
        print(f"\n{mutant}: valid={v.is_valid}, conservative={v.conservative_score:.2f}, novelty={v.novelty_score:.2f}")
        for viol in v.violations:
            print(f"  [{viol.severity}] {viol.rule}: {viol.message}")
