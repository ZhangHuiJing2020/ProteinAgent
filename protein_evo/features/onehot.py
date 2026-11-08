"""突变位点 one-hot 编码。
"""
from __future__ import annotations

import numpy as np

from ..knowledge.amino_acid_props import AA_PROPS, STANDARD_AAS
from ..data.wt import WT_SEQUENCE, MUTATION_SITES, FULL_LENGTH_SEQUENCE

# 氨基酸顺序
AA_LIST = sorted(STANDARD_AAS)  # 字典序，20 个
_AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AAS = 20


_DEFAULT_SITES = sorted(MUTATION_SITES.keys())
N_SITES = len(_DEFAULT_SITES)
ONEHOT_DIM = N_SITES * N_AAS

# 位点索引映射
SITE_POSITIONS = _DEFAULT_SITES
_SITE_TO_IDX = {pos: i for i, pos in enumerate(SITE_POSITIONS)}

# 全局位点集合（可被外部调用扩展）
_ALL_SITES = set(SITE_POSITIONS)


def update_sites(positions: list[int]):
    """动态更新位点集合（在加载数据后调用）。

    重新计算 ONEHOT_DIM 和索引映射。
    """
    global _ALL_SITES, SITE_POSITIONS, _SITE_TO_IDX, N_SITES, ONEHOT_DIM
    _ALL_SITES = set(positions)
    SITE_POSITIONS = sorted(_ALL_SITES)
    _SITE_TO_IDX = {pos: i for i, pos in enumerate(SITE_POSITIONS)}
    N_SITES = len(SITE_POSITIONS)
    ONEHOT_DIM = N_SITES * N_AAS


def encode_mutations(mutations: list[tuple[int, str, str]]) -> np.ndarray:
    """编码突变列表为 one-hot 向量。

    Args:
        mutations: [(position, wt_aa, mut_aa), ...]

    Returns:
        np.ndarray shape=(ONEHOT_DIM=80,), dtype=float32
        位置 i*20+j = 1 表示位点 i 替换为氨基酸 j
        未突变位点全 0
    """
    vec = np.zeros(ONEHOT_DIM, dtype=np.float32)
    for pos, wt_aa, mut_aa in mutations:
        if pos in _SITE_TO_IDX and mut_aa in _AA_TO_IDX:
            site_idx = _SITE_TO_IDX[pos]
            aa_idx = _AA_TO_IDX[mut_aa]
            vec[site_idx * N_AAS + aa_idx] = 1.0
    return vec


def encode_mutant_str(mutant_str: str) -> np.ndarray:
    """从 'A1P:D2N' 格式编码。"""
    vec = np.zeros(ONEHOT_DIM, dtype=np.float32)
    if not mutant_str or mutant_str == "wt":
        return vec
    for token in mutant_str.split(":"):
        token = token.strip()
        if len(token) < 3:
            continue
        mut_aa = token[-1]
        try:
            pos = int(token[1:-1]) - 1  # 1-based → 0-based
        except ValueError:
            continue
        if pos in _SITE_TO_IDX and mut_aa in _AA_TO_IDX:
            site_idx = _SITE_TO_IDX[pos]
            aa_idx = _AA_TO_IDX[mut_aa]
            vec[site_idx * N_AAS + aa_idx] = 1.0
    return vec


def encode_sequence(mutated_seq: str, wt_seq: str = WT_SEQUENCE) -> np.ndarray:
    """从完整突变序列编码（对比 wt 推断突变位点）。"""
    mutations = []
    for i, (w, m) in enumerate(zip(wt_seq, mutated_seq)):
        if w != m and i in _SITE_TO_IDX:
            mutations.append((i, w, m))
    return encode_mutations(mutations)


if __name__ == "__main__":
    # 自测
    v = encode_mutations([(38, "V", "A"), (39, "D", "E")])
    print(f"维度: {v.shape}, 非零数: {(v != 0).sum()}")
    # V39A → site 0 (位置38), A 在 AA_LIST 的索引
    site0 = 0 * N_AAS + _AA_TO_IDX["A"]
    site1 = 1 * N_AAS + _AA_TO_IDX["E"]
    print(f"V39A 位置 ({site0}): {v[site0]}")
    print(f"D40E 位置 ({site1}): {v[site1]}")
    print(f"其余位置: {(v[:site0].sum() + v[site0+1:site1].sum() + v[site1+1:].sum()) == 0}")
