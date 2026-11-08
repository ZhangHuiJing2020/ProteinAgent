"""位点相互作用特征编码。

显式编码位点 pair 的上位效应信号：
- mutation_count: 突变总数
- pair_indicator: 每对位点是否同时突变
- pair_blosum_sum: 每对位点 BLOSUM 得分之和
- pair_charge_product: 每对位点电荷乘积
- pair_hydro_sum: 每对位点疏水性之和

"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from ..data.wt import MUTATION_SITES
from ..knowledge.amino_acid_props import AA_PROPS
from .blosum import blosum_score

# 4 个突变位点的 6 个 pair
SITE_POSITIONS = sorted(MUTATION_SITES.keys())  # [38, 39, 40, 53] (0-based)
SITE_PAIRS = list(combinations(SITE_POSITIONS, 2))  # 6 个 pair
N_PAIRS = len(SITE_PAIRS)  # 6

# 特征维度：1 + 6×4 = 25
INTERACTION_DIM = 1 + N_PAIRS * 4


def encode_mutations(mutations: list[tuple[int, str, str]]) -> np.ndarray:
    """编码突变列表为位点相互作用特征向量。

    Args:
        mutations: [(position_0based, wt_aa, mut_aa), ...]

    Returns:
        np.ndarray shape=(25,), dtype=float32
        [mutation_count, pair_indicator(6), pair_blosum_sum(6),
         pair_charge_product(6), pair_hydro_sum(6)]
    """
    vec = np.zeros(INTERACTION_DIM, dtype=np.float32)

    if not mutations:
        return vec

    # 突变总数（归一化）
    vec[0] = float(len(mutations)) / 4.0

    # 构造位点 → 突变字典
    pos_to_mut = {m[0]: m for m in mutations}

    # 对每个 pair 编码
    for i, (p1, p2) in enumerate(SITE_PAIRS):
        m1 = pos_to_mut.get(p1)
        m2 = pos_to_mut.get(p2)
        offset = 1 + i * 4

        # pair_indicator: 两个位点都突变
        if m1 is not None and m2 is not None:
            vec[offset] = 1.0
        elif m1 is not None or m2 is not None:
            vec[offset] = 0.5
        # 否则 0

        # pair_blosum_sum: 两个位点的 BLOSUM 替换得分之和（单突变也算）
        blosum_sum = 0.0
        if m1 is not None:
            blosum_sum += blosum_score(m1[1], m1[2])
        if m2 is not None:
            blosum_sum += blosum_score(m2[1], m2[2])
        vec[offset + 1] = blosum_sum / 8.0  # 归一化 [-1, 1]

        # pair_charge_product: 电荷乘积（捕捉电荷-电荷相互作用）
        charge1 = 0.0
        charge2 = 0.0
        if m1 is not None:
            charge1 = float(AA_PROPS.get(m1[2], {}).get("charge", 0))
        elif p1 in MUTATION_SITES:
            charge1 = float(AA_PROPS.get(MUTATION_SITES[p1], {}).get("charge", 0))
        if m2 is not None:
            charge2 = float(AA_PROPS.get(m2[2], {}).get("charge", 0))
        elif p2 in MUTATION_SITES:
            charge2 = float(AA_PROPS.get(MUTATION_SITES[p2], {}).get("charge", 0))
        vec[offset + 2] = charge1 * charge2

        # pair_hydro_sum: 疏水性之和
        hydro1 = 0.0
        hydro2 = 0.0
        if m1 is not None:
            hydro1 = float(AA_PROPS.get(m1[2], {}).get("hydrophobicity", 0))
        elif p1 in MUTATION_SITES:
            hydro1 = float(AA_PROPS.get(MUTATION_SITES[p1], {}).get("hydrophobicity", 0))
        if m2 is not None:
            hydro2 = float(AA_PROPS.get(m2[2], {}).get("hydrophobicity", 0))
        elif p2 in MUTATION_SITES:
            hydro2 = float(AA_PROPS.get(MUTATION_SITES[p2], {}).get("hydrophobicity", 0))
        vec[offset + 3] = (hydro1 + hydro2) / 10.0  # 归一化

    return vec


def encode_mutant_str(mutant_str: str) -> np.ndarray:
    """从 'A1P:D2N' 格式编码。"""
    if not mutant_str or mutant_str == "wt":
        return np.zeros(INTERACTION_DIM, dtype=np.float32)
    from ..data.loader import parse_mutant_str
    from ..data.wt import WT_SEQUENCE
    mutations = parse_mutant_str(mutant_str, WT_SEQUENCE)
    return encode_mutations([(m.position, m.wt_aa, m.mut_aa) for m in mutations])


if __name__ == "__main__":
    # 自测
    print(f"特征维度: {INTERACTION_DIM}")
    print(f"位点 pair 数: {N_PAIRS}")

    # 单点突变
    v = encode_mutant_str("V39A")
    print(f"\nV39A (单点): count={v[0]}, pair_indicators={v[1:7]}")

    # 双点突变
    v = encode_mutant_str("V39A:V54I")
    print(f"\nV39A:V54I (双点 39-54): count={v[0]}, pair_indicators={v[1:7]}")
    # 39-54 是第 5 个 pair (38-39, 38-40, 38-53, 39-40, 39-53, 40-53)
    # 实际 SITE_PAIRS 用 0-based: (38,39), (38,40), (38,53), (39,40), (39,53), (40,53)
    # 39-54 对应 1-based 39 和 54 → 0-based 38 和 53 → pair (38, 53) = 第 3 个

    # 三点突变
    v = encode_mutant_str("V39A:D40E:V54I")
    print(f"\nV39A:D40E:V54I (三点): count={v[0]}, pair_indicators={v[1:7]}")
