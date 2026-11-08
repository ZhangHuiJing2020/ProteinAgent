"""评估指标：Spearman / Pearson / MSE / Top-k 命中率 / NDCG@k。"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr, pearsonr


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman 秩相关系数。"""
    if len(y_true) < 2:
        return 0.0
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson 线性相关系数。"""
    if len(y_true) < 2:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方误差。"""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def topk_hit(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """Top-k 命中率：模型预测 Top-k 中包含的真实 Top-k 比例。

    |pred_topk ∩ true_topk| / k
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    k = min(k, n)
    true_topk_idx = set(np.argsort(y_true)[::-1][:k].tolist())
    pred_topk_idx = set(np.argsort(y_pred)[::-1][:k].tolist())
    return len(true_topk_idx & pred_topk_idx) / k


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> float:
    """NDCG@k：归一化折损累积增益。

    DCG@k = sum_{i=1}^{k} (2^rel_i - 1) / log2(i+1)
    其中 rel_i 是按预测排序的真实 fitness 值。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    k = min(k, n)
    if k == 0:
        return 0.0

    # 预测排序下的 DCG
    pred_order = np.argsort(y_pred)[::-1][:k]
    dcg = sum((2 ** y_true[idx] - 1) / np.log2(i + 2) for i, idx in enumerate(pred_order))

    # 真实排序下的 IDCG
    true_order = np.argsort(y_true)[::-1][:k]
    idcg = sum((2 ** y_true[idx] - 1) / np.log2(i + 2) for i, idx in enumerate(true_order))

    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> dict:
    """一次性计算所有指标。"""
    return {
        "spearman": spearman(y_true, y_pred),
        "pearson": pearson(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        f"topk_hit@{k}": topk_hit(y_true, y_pred, k),
        f"ndcg@{k}": ndcg_at_k(y_true, y_pred, k),
    }


if __name__ == "__main__":
    # 自测
    y_true = np.array([0.1, 0.5, 0.8, 0.3, 0.9, 0.2, 0.7, 0.4, 0.6, 0.0])
    y_pred = np.array([0.2, 0.6, 0.7, 0.4, 0.8, 0.3, 0.5, 0.5, 0.6, 0.1])
    m = evaluate_all(y_true, y_pred, k=3)
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
