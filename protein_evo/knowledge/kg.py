"""知识图谱 — 知识库 L3。

- (Amino Acid, has_property, Property)
- (Mutation, occurs_at, Position)
- (Mutation, improves, Fitness) — 从历史数据动态填充
- (Variant, contains, Mutation)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .amino_acid_props import AA_PROPS, get_category, is_hydrophobic, is_charged, is_polar, is_small
from ..data.loader import parse_mutant_str, Mutation
from ..data.wt import WT_SEQUENCE, MUTATION_SITES


@dataclass(frozen=True)
class Triple:
    """知识图谱三元组。"""
    subject: str
    predicate: str
    object: str

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class KnowledgeGraph:
    """简单三元组知识图谱。"""

    def __init__(self):
        self._triples: list[Triple] = []
        # 索引（按 predicate 索引）
        self._by_pred: dict[str, list[Triple]] = defaultdict(list)
        self._by_subj: dict[str, list[Triple]] = defaultdict(list)
        self._by_obj: dict[str, list[Triple]] = defaultdict(list)

    def add(self, subject: str, predicate: str, object: str) -> None:
        """添加三元组。"""
        t = Triple(subject, predicate, object)
        self._triples.append(t)
        self._by_pred[predicate].append(t)
        self._by_subj[subject].append(t)
        self._by_obj[object].append(t)

    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
    ) -> list[Triple]:
        """查询三元组。任一参数为 None 表示通配。"""
        results = []
        for t in self._triples:
            if subject and t.subject != subject:
                continue
            if predicate and t.predicate != predicate:
                continue
            if object and t.object != object:
                continue
            results.append(t)
        return results

    def query_subjects(self, predicate: str, object: str) -> list[str]:
        """查询符合 (subject, predicate, object) 的所有 subject。"""
        return list({t.subject for t in self.query(predicate=predicate, object=object)})

    def query_objects(self, subject: str, predicate: str) -> list[str]:
        """查询符合 (subject, predicate, ?) 的所有 object。"""
        return [t.object for t in self.query(subject=subject, predicate=predicate)]

    def count(self) -> int:
        return len(self._triples)

    def all_triples(self) -> list[Triple]:
        return list(self._triples)

    def __repr__(self) -> str:
        return f"KnowledgeGraph({self.count()} triples)"


def build_static_kg() -> KnowledgeGraph:
    """构建静态知识图谱：氨基酸性质 + 位点信息。"""
    kg = KnowledgeGraph()

    # (AminoAcid, has_property, Property)
    for aa, props in AA_PROPS.items():
        # 主分类
        kg.add(f"AA:{aa}", "has_property", f"Category:{props['category']}")
        # 疏水性
        if props["hydrophobicity"] > 0:
            kg.add(f"AA:{aa}", "has_property", "Hydrophobic")
        else:
            kg.add(f"AA:{aa}", "has_property", "Hydrophilic")
        # 电荷
        if props["charge"] > 0:
            kg.add(f"AA:{aa}", "has_property", "PositivelyCharged")
        elif props["charge"] < 0:
            kg.add(f"AA:{aa}", "has_property", "NegativelyCharged")
        else:
            kg.add(f"AA:{aa}", "has_property", "Neutral")
        # 极性
        if props["polarity"]:
            kg.add(f"AA:{aa}", "has_property", "Polar")
        else:
            kg.add(f"AA:{aa}", "has_property", "NonPolar")
        # 大小
        if props["size"] < 50:
            kg.add(f"AA:{aa}", "has_property", "Small")
        elif props["size"] < 90:
            kg.add(f"AA:{aa}", "has_property", "Medium")
        else:
            kg.add(f"AA:{aa}", "has_property", "Large")

    # (Position, is_site_of, Protein) — 4 个突变位点
    for pos_0, aa in MUTATION_SITES.items():
        pos_1 = pos_0 + 1
        kg.add(f"Position:{pos_1}", "is_mutation_site_of", "Protein:GB1")
        kg.add(f"Position:{pos_1}", "wild_type_aa", f"AA:{aa}")

    return kg


def augment_with_history(
    kg: KnowledgeGraph,
    history_df: pd.DataFrame,
    fitness_threshold: float = 0.5,
) -> KnowledgeGraph:
    """从历史数据扩充知识图谱。

    添加：
    - (Mutation, occurs_at, Position)
    - (Mutation, improves, Fitness) — 高 fitness 突变
    - (Variant, contains, Mutation)
    """
    if "mutant" not in history_df.columns:
        return kg

    for _, row in history_df.iterrows():
        mutant_str = str(row.get("mutant", ""))
        if not mutant_str or mutant_str == "wt":
            continue
        fitness = float(row.get("DMS_score", 0.0))
        mutations = parse_mutant_str(mutant_str, WT_SEQUENCE)

        # 每个突变作为一个节点
        for m in mutations:
            single_str = f"{m.wt_aa}{m.position+1}{m.mut_aa}"
            kg.add(f"Mutation:{single_str}", "occurs_at", f"Position:{m.position+1}")
            kg.add(f"Mutation:{single_str}", "substitutes_to", f"AA:{m.mut_aa}")
            kg.add(f"Variant:{mutant_str}", "contains", f"Mutation:{single_str}")

            # 高 fitness 标记
            if fitness >= fitness_threshold:
                kg.add(f"Mutation:{single_str}", "improves", "Fitness:High")
            elif fitness >= 0.0:
                kg.add(f"Mutation:{single_str}", "improves", "Fitness:Medium")
            else:
                kg.add(f"Mutation:{single_str}", "improves", "Fitness:Low")

        # 整体 variant 的 fitness
        kg.add(f"Variant:{mutant_str}", "has_fitness", f"{fitness:.3f}")

    return kg


def query_hydrophobic_aas(kg: KnowledgeGraph) -> list[str]:
    """查询疏水性氨基酸。"""
    return [s.replace("AA:", "") for s in kg.query_subjects("has_property", "Hydrophobic")]


def query_charged_aas(kg: KnowledgeGraph) -> list[str]:
    """查询带电氨基酸。"""
    pos = [s.replace("AA:", "") for s in kg.query_subjects("has_property", "PositivelyCharged")]
    neg = [s.replace("AA:", "") for s in kg.query_subjects("has_property", "NegativelyCharged")]
    return pos + neg


def query_high_fitness_mutations(kg: KnowledgeGraph) -> list[str]:
    """查询历史高 fitness 突变。"""
    return [s.replace("Mutation:", "") for s in kg.query_subjects("improves", "Fitness:High")]


def query_mutations_at_position(kg: KnowledgeGraph, position_1based: int) -> list[str]:
    """查询某位点的所有历史突变。"""
    triples = kg.query(predicate="occurs_at", object=f"Position:{position_1based}")
    return [t.subject.replace("Mutation:", "") for t in triples]


if __name__ == "__main__":
    # 自测
    print("构建静态 KG ...")
    kg = build_static_kg()
    print(f"  三元组数: {kg.count()}")

    print(f"\n查询疏水性氨基酸: {query_hydrophobic_aas(kg)}")
    print(f"查询带电氨基酸: {query_charged_aas(kg)}")

    print(f"\n位点 39 的野生型: {kg.query_objects('Position:39', 'wild_type_aa')}")
    print(f"位点 39 的历史突变: {query_mutations_at_position(kg, 39)}")

    # 添加历史数据
    import pandas as pd
    history = pd.DataFrame([
        {"mutant": "V39A", "DMS_score": 0.8, "n_mutations": 1},
        {"mutant": "D40E", "DMS_score": 0.6, "n_mutations": 1},
        {"mutant": "V39A:D40E", "DMS_score": 1.2, "n_mutations": 2},
    ])
    kg = augment_with_history(kg, history)
    print(f"\n添加历史后三元组数: {kg.count()}")
    print(f"高 fitness 突变: {query_high_fitness_mutations(kg)}")
