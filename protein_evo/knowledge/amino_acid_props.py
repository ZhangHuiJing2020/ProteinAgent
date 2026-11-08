"""20 种氨基酸的理化性质表。

数据来源：Kyte-Doolittle 疏水性标度、Lehninger 生物化学标准表。
用于：
1. 特征工程（physchem.py 编码）
2. 知识库（mutation_rules.py 规则、kg.py 三元组）
3. Agent 推理时查询
"""

# 20 种标准氨基酸的理化性质
# - hydrophobicity: Kyte-Doolittle 标度（-4.5 到 +4.5，正值=疏水）
# - charge: pH 7.0 净电荷
# - size: 侧链分子量（Da）
# - polarity: 是否极性
# - category: 简化分类
AA_PROPS = {
    "A": {"name": "Alanine",       "hydrophobicity":  1.8, "charge":  0, "size":  15, "polarity": False, "category": "hydrophobic"},
    "R": {"name": "Arginine",      "hydrophobicity": -4.5, "charge": +1, "size": 101, "polarity": True,  "category": "charged"},
    "N": {"name": "Asparagine",    "hydrophobicity": -3.5, "charge":  0, "size":  58, "polarity": True,  "category": "polar"},
    "D": {"name": "Aspartate",     "hydrophobicity": -3.5, "charge": -1, "size":  59, "polarity": True,  "category": "charged"},
    "C": {"name": "Cysteine",      "hydrophobicity":  2.5, "charge":  0, "size":  47, "polarity": True,  "category": "polar"},
    "Q": {"name": "Glutamine",     "hydrophobicity": -3.5, "charge":  0, "size":  72, "polarity": True,  "category": "polar"},
    "E": {"name": "Glutamate",     "hydrophobicity": -3.5, "charge": -1, "size":  73, "polarity": True,  "category": "charged"},
    "G": {"name": "Glycine",       "hydrophobicity": -0.4, "charge":  0, "size":   1, "polarity": False, "category": "small"},
    "H": {"name": "Histidine",     "hydrophobicity": -3.2, "charge":  0, "size":  81, "polarity": True,  "category": "charged"},
    "I": {"name": "Isoleucine",    "hydrophobicity":  4.5, "charge":  0, "size":  57, "polarity": False, "category": "hydrophobic"},
    "L": {"name": "Leucine",       "hydrophobicity":  3.8, "charge":  0, "size":  57, "polarity": False, "category": "hydrophobic"},
    "K": {"name": "Lysine",        "hydrophobicity": -3.9, "charge": +1, "size":  72, "polarity": True,  "category": "charged"},
    "M": {"name": "Methionine",    "hydrophobicity":  1.9, "charge":  0, "size":  75, "polarity": False, "category": "hydrophobic"},
    "F": {"name": "Phenylalanine", "hydrophobicity":  2.8, "charge":  0, "size":  91, "polarity": False, "category": "hydrophobic"},
    "P": {"name": "Proline",       "hydrophobicity": -1.6, "charge":  0, "size":  42, "polarity": False, "category": "special"},
    "S": {"name": "Serine",        "hydrophobicity": -0.8, "charge":  0, "size":  31, "polarity": True,  "category": "polar"},
    "T": {"name": "Threonine",     "hydrophobicity": -0.7, "charge":  0, "size":  45, "polarity": True,  "category": "polar"},
    "W": {"name": "Tryptophan",   "hydrophobicity": -0.9, "charge":  0, "size": 130, "polarity": False, "category": "hydrophobic"},
    "Y": {"name": "Tyrosine",     "hydrophobicity": -1.3, "charge":  0, "size": 107, "polarity": True,  "category": "polar"},
    "V": {"name": "Valine",       "hydrophobicity":  4.2, "charge":  0, "size":  43, "polarity": False, "category": "hydrophobic"},
}

# 标准氨基酸集合（用于规则验证）
STANDARD_AAS = frozenset(AA_PROPS.keys())

# 终止符和异常氨基酸
NONSTANDARD_AAS = frozenset({"*", "X", "U", "O", "B", "Z", "J"})


def is_standard(aa: str) -> bool:
    """是否为标准 20 氨基酸之一。"""
    return aa in STANDARD_AAS


def is_terminator(aa: str) -> bool:
    """是否为终止符。"""
    return aa == "*"


def get_props(aa: str) -> dict:
    """获取氨基酸性质字典。未知返回空字典。"""
    return AA_PROPS.get(aa, {})


def get_category(aa: str) -> str:
    """获取简化分类。未知返回 'unknown'。"""
    return AA_PROPS.get(aa, {}).get("category", "unknown")


def is_hydrophobic(aa: str) -> bool:
    return AA_PROPS.get(aa, {}).get("hydrophobicity", 0) > 0


def is_charged(aa: str) -> bool:
    return AA_PROPS.get(aa, {}).get("charge", 0) != 0


def is_polar(aa: str) -> bool:
    return AA_PROPS.get(aa, {}).get("polarity", False)


def is_small(aa: str, threshold: int = 50) -> bool:
    return AA_PROPS.get(aa, {}).get("size", 999) < threshold


if __name__ == "__main__":
    # 自测
    print(f"标准氨基酸数：{len(STANDARD_AAS)}")
    for aa in "ACDE":
        p = AA_PROPS[aa]
        print(f"  {aa} ({p['name']}): hydro={p['hydrophobicity']:+.1f}, charge={p['charge']:+d}, "
              f"size={p['size']}, category={p['category']}")
    print(f"\nV39 (V) → hydrophobic: {is_hydrophobic('V')}")
    print(f"D40 (D) → charged: {is_charged('D')}")
    print(f"G41 (G) → small: {is_small('G')}")
