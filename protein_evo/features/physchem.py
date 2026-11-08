"""理化性质特征编码。

将突变位置 (wt_aa, mut_aa) 编码为理化性质变化向量：
- Δ hydrophobicity（疏水性变化）
- Δ charge（电荷变化）
- Δ size（侧链大小变化）
- Δ polarity（极性变化）
- 是否保守替换
- 是否引入终止/异常
"""
from __future__ import annotations

import numpy as np

from ..knowledge.amino_acid_props import (
    AA_PROPS, STANDARD_AAS, is_standard, is_terminator,
    is_hydrophobic, is_charged, is_polar, is_small,
)
from .blosum import blosum_score

# 理化性质特征维度：6 个标量
PHYSCHEM_DIM = 6


def encode_mutation_pair(wt_aa: str, mut_aa: str) -> np.ndarray:
    """编码单突变对的理化性质变化。

    Returns:
        np.ndarray shape=(6,), dtype=float32
        [Δhydrophobicity, Δcharge, Δsize, Δpolarity, conservative_score, is_problematic]
    """
    wt_props = AA_PROPS.get(wt_aa, {"hydrophobicity": 0, "charge": 0, "size": 0, "polarity": False})
    mut_props = AA_PROPS.get(mut_aa, {"hydrophobicity": 0, "charge": 0, "size": 0, "polarity": False})

    delta_hydro = mut_props["hydrophobicity"] - wt_props["hydrophobicity"]
    delta_charge = mut_props["charge"] - wt_props["charge"]
    delta_size = (mut_props["size"] - wt_props["size"]) / 100.0  # 归一化
    delta_polarity = float(mut_props["polarity"]) - float(wt_props["polarity"])

    # 保守替换分数（基于 BLOSUM）
    bscore = blosum_score(wt_aa, mut_aa)
    conservative_score = max(0.0, bscore / 4.0)  # 0~1，4=完全保守

    # 异常标记：终止符或非标准氨基酸
    is_problematic = 1.0 if (is_terminator(mut_aa) or not is_standard(mut_aa)) else 0.0

    return np.array([
        delta_hydro, delta_charge, delta_size, delta_polarity,
        conservative_score, is_problematic,
    ], dtype=np.float32)


def encode_mutations(mutations: list[tuple[int, str, str]]) -> np.ndarray:
    """编码突变列表，对每个突变取理化性质变化求和/最大。

    Returns:
        np.ndarray shape=(PHYSCHEM_DIM,), 所有突变的理化性质变化求和
    """
    if not mutations:
        return np.zeros(PHYSCHEM_DIM, dtype=np.float32)
    vec = np.zeros(PHYSCHEM_DIM, dtype=np.float32)
    for pos, wt, mut in mutations:
        vec += encode_mutation_pair(wt, mut)
    return vec


def encode_mutant_str(mutant_str: str) -> np.ndarray:
    """从 'A1P:D2N' 格式编码。"""
    vec = np.zeros(PHYSCHEM_DIM, dtype=np.float32)
    if not mutant_str or mutant_str == "wt":
        return vec
    for token in mutant_str.split(":"):
        token = token.strip()
        if len(token) < 3:
            continue
        wt_aa = token[0]
        mut_aa = token[-1]
        vec += encode_mutation_pair(wt_aa, mut_aa)
    return vec


def encode_sequence(mutated_seq: str, wt_seq: str) -> np.ndarray:
    """从完整序列对比 wt 推断突变并编码。"""
    mutations = []
    for i, (w, m) in enumerate(zip(wt_seq, mutated_seq)):
        if w != m:
            mutations.append((i, w, m))
    return encode_mutations(mutations)


if __name__ == "__main__":
    # 自测
    v = encode_mutation_pair("V", "A")
    print(f"V→A: Δhydro={v[0]:+.2f} Δcharge={v[1]:+.2f} Δsize={v[2]:+.2f} "
          f"conservative={v[4]:.2f} problematic={v[5]}")
    v = encode_mutation_pair("D", "E")
    print(f"D→E: Δhydro={v[0]:+.2f} Δcharge={v[1]:+.2f} Δsize={v[2]:+.2f} "
          f"conservative={v[4]:.2f}")
    v = encode_mutations([(38, "V", "A"), (39, "D", "E")])
    print(f"V39A:D40E sum: {v}")
