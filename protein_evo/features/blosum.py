"""BLOSUM62 替换矩阵编码。

将突变位置 (wt_aa, mut_aa) 编码为 BLOSUM62 替换得分。
BLOSUM62 是蛋白质序列比对的标准打分矩阵，正值=保守替换，负值=激进替换。
"""
from __future__ import annotations

import numpy as np

# BLOSUM62 替换矩阵（标准 NCBI 版本，行=wt，列=mut）
# 顺序：ARNDCQEGHILKMFPSTWYV
_BLOSUM62_ROWS = [
    # A    R    N    D    C    Q    E    G    H    I    L    K    M    F    P    S    T    W    Y    V
    [   4,  -1,  -2,  -2,   0,  -1,  -1,   0,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   0,  -3,  -2,   0],  # A
    [  -1,   5,   0,  -2,  -3,   1,   0,  -2,   0,  -3,  -2,   2,  -1,  -3,  -2,  -1,  -1,  -3,  -2,  -3],  # R
    [  -2,   0,   6,   1,  -3,   0,   0,   0,   1,  -3,  -3,   0,  -2,  -3,  -2,   1,   0,  -4,  -2,  -3],  # N
    [  -2,  -2,   1,   6,  -3,   0,   2,  -1,  -1,  -3,  -4,  -1,  -3,  -3,  -1,   0,  -1,  -4,  -3,  -3],  # D
    [   0,  -3,  -3,  -3,   9,  -3,  -4,  -3,  -3,  -1,  -1,  -3,  -1,  -2,  -3,  -1,   0,  -2,  -2,  -1],  # C
    [  -1,   1,   0,   0,  -3,   5,   2,  -2,   0,  -3,  -2,   1,   0,  -3,  -1,   0,  -1,  -2,  -1,  -2],  # Q
    [  -1,   0,   0,   2,  -4,   2,   5,  -2,   0,  -3,  -3,   1,  -2,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # E
    [   0,  -2,   0,  -1,  -3,  -2,  -2,   6,  -2,  -4,  -4,  -2,  -3,  -3,  -2,   0,  -2,  -2,  -3,  -3],  # G
    [  -2,   0,   1,  -1,  -3,   0,   0,  -2,   8,  -3,  -3,  -1,  -2,  -1,  -2,  -1,  -2,  -2,   2,  -3],  # H
    [  -1,  -3,  -3,  -3,  -1,  -3,  -3,  -4,  -3,   4,   2,  -3,   1,   0,  -3,  -2,  -1,  -3,  -1,   3],  # I
    [  -1,  -2,  -3,  -4,  -1,  -2,  -3,  -4,  -3,   2,   4,  -2,   2,   0,  -3,  -2,  -1,  -2,  -1,   1],  # L
    [  -1,   2,   0,  -1,  -3,   1,   1,  -2,  -1,  -3,  -2,   5,  -1,  -3,  -1,   0,  -1,  -3,  -2,  -2],  # K
    [  -1,  -1,  -2,  -3,  -1,   0,  -2,  -3,  -2,   1,   2,  -1,   5,   0,  -2,  -1,  -1,  -1,  -1,   1],  # M
    [  -2,  -3,  -3,  -3,  -2,  -3,  -3,  -3,  -1,   0,   0,  -3,   0,   6,  -4,  -2,  -2,   1,   3,  -1],  # F
    [  -1,  -2,  -2,  -1,  -3,  -1,  -1,  -2,  -2,  -3,  -3,  -1,  -2,  -4,   7,  -1,  -1,  -4,  -3,  -2],  # P
    [   1,  -1,   1,   0,  -1,   0,   0,   0,  -1,  -2,  -2,   0,  -1,  -2,  -1,   4,   1,  -3,  -2,  -2],  # S
    [   0,  -1,   0,  -1,  -1,  -1,  -1,  -2,  -2,  -1,  -1,  -1,  -1,  -2,  -1,   1,   5,  -2,  -2,   0],  # T
    [  -3,  -3,  -4,  -4,  -2,  -2,  -3,  -2,  -2,  -3,  -2,  -3,  -1,   1,  -4,  -3,  -2,  11,   2,  -3],  # W
    [  -2,  -2,  -2,  -3,  -2,  -1,  -2,  -3,   2,  -1,  -1,  -2,  -1,   3,  -3,  -2,  -2,   2,   7,  -1],  # Y
    [   0,  -3,  -3,  -3,  -1,  -2,  -2,  -3,  -3,   3,   1,  -2,   1,  -1,  -2,  -2,   0,  -3,  -1,   4],  # V
]

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
_AA_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}
BLOSUM62 = {wt: {mut: _BLOSUM62_ROWS[_AA_INDEX[wt]][_AA_INDEX[mut]] for mut in AA_ORDER}
            for wt in AA_ORDER}


def blosum_score(wt_aa: str, mut_aa: str) -> int:
    """单替换的 BLOSUM62 得分。未知氨基酸返回 -4（保守惩罚）。"""
    if wt_aa in BLOSUM62 and mut_aa in BLOSUM62[wt_aa]:
        return BLOSUM62[wt_aa][mut_aa]
    return -4


def encode_mutations(mutations: list[tuple[int, str, str]], seq_len: int = 448) -> np.ndarray:
    """将突变列表编码为长度=seq_len 的 BLOSUM 得分向量。

    每个位置 0（无突变）/ BLOSUM 分数（有突变）。

    Args:
        mutations: [(position, wt_aa, mut_aa), ...]
        seq_len: 序列长度（默认 448 = 全长蛋白 G 长度）

    Returns:
        np.ndarray shape=(seq_len,), dtype=float
    """
    vec = np.zeros(seq_len, dtype=np.float32)
    for pos, wt, mut in mutations:
        if 0 <= pos < seq_len:
            vec[pos] = blosum_score(wt, mut)
    return vec


def encode_mutant_str(mutant_str: str, wt_seq: str = "", seq_len: int = 448) -> np.ndarray:
    """从 'A1P:D2N' 格式编码。

    seq_len 默认 448（全长蛋白 G），支持全长编号的突变（如 Q228R）。
    """
    vec = np.zeros(seq_len, dtype=np.float32)
    if not mutant_str or mutant_str == "wt":
        return vec
    for token in mutant_str.split(":"):
        token = token.strip()
        if len(token) < 3:
            continue
        wt_aa = token[0]
        mut_aa = token[-1]
        try:
            pos = int(token[1:-1]) - 1  # 1-based → 0-based
        except ValueError:
            continue
        if 0 <= pos < seq_len:
            vec[pos] = blosum_score(wt_aa, mut_aa)
    return vec


if __name__ == "__main__":
    # 自测
    print(f"BLOSUM62(V,A) = {blosum_score('V', 'A')}")  # 0
    print(f"BLOSUM62(V,V) = {blosum_score('V', 'V')}")  # 4
    print(f"BLOSUM62(D,E) = {blosum_score('D', 'E')}")  # 2 (保守)
    print(f"BLOSUM62(V,D) = {blosum_score('V', 'D')}")  # -3 (激进)
    v = encode_mutations([(38, "V", "A"), (39, "D", "E")])
    print(f"V39A:D40E → {v[38]}, {v[39]}, others={v[:38].sum()}")
