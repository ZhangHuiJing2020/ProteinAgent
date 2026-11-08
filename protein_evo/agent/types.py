"""Agent 数据结构定义。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VariantInfo:
    """变体信息。"""
    mutant: str
    mutated_sequence: str
    fitness: float
    n_mutations: int = 1


@dataclass
class AnalystReport:
    """Data Analyst 输出。"""
    summary: str
    top_variants: list[VariantInfo] = field(default_factory=list)
    position_frequency: dict[int, int] = field(default_factory=dict)  # {position_1based: count}
    fitness_stats: dict = field(default_factory=dict)
    insights: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "top_variants": [asdict(v) for v in self.top_variants],
            "position_frequency": {str(k): v for k, v in self.position_frequency.items()},
            "fitness_stats": self.fitness_stats,
            "insights": self.insights,
        }


@dataclass
class Hypothesis:
    """单个突变假设。"""
    position: int  # 1-based
    current_aa: str
    suggested_aas: list[str]
    rationale: str


@dataclass
class HypothesisReport:
    """Hypothesis Generator 输出。"""
    hypotheses: list[Hypothesis] = field(default_factory=list)
    combination_strategy: str = ""
    max_mutations: int = 3

    def to_dict(self) -> dict:
        return {
            "hypotheses": [asdict(h) for h in self.hypotheses],
            "combination_strategy": self.combination_strategy,
            "max_mutations": self.max_mutations,
        }


@dataclass
class Candidate:
    """候选突变。"""
    mutant: str
    n_mutations: int = 1
    predicted_fitness: Optional[float] = None
    validation: Optional[dict] = None  # MutationValidation.to_dict()
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CriticReport:
    """Scientific Critic 输出。"""
    recommendations: list[dict] = field(default_factory=list)  # 每项含 mutant, rank, rationale, scores
    overall_assessment: str = ""
    rejected_count: int = 0
    rejected_examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoundResult:
    """单轮 Agent 完整结果。"""
    round_number: int
    method: str  # random | model_only | llm_agent | llm_agent_kb
    analyst_report: Optional[dict] = None
    hypothesis_report: Optional[dict] = None
    candidates: list[dict] = field(default_factory=list)  # 候选 + 预测
    critic_report: Optional[dict] = None
    topk: list[dict] = field(default_factory=list)  # 最终 Top-k 推荐
    topk_real_fitness: list[float] = field(default_factory=list)  # 真实 fitness（来自 VirtualLab）
    metrics: dict = field(default_factory=dict)  # 本轮指标

    def to_dict(self) -> dict:
        return {
            "round_number": self.round_number,
            "method": self.method,
            "analyst_report": self.analyst_report,
            "hypothesis_report": self.hypothesis_report,
            "candidates": self.candidates,
            "critic_report": self.critic_report,
            "topk": self.topk,
            "topk_real_fitness": self.topk_real_fitness,
            "metrics": self.metrics,
        }
