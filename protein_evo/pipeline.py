"""Pipeline 模块 — 将数据管线、模型训练、实验评估封装为可调用的函数。
参考 EvolvePro (github.com/mat10d/EvolvePro) 的 pipeline
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .data.download import download_gb1, download_reference
from .data.split import main as split_main
from .data.loader import load_variants, parse_mutant_str
from .data.wt import WT_SEQUENCE, FULL_LENGTH_SEQUENCE
from .model.fitness_model import FitnessModel, FeatureConfig, create_default_model
from .model.metrics import evaluate_all
from .agent.orchestrator import AgentOrchestrator
from .agent.llm_backend import MockBackend
from .experiment.comparison import run_comparison
from .experiment.closed_loop import ClosedLoopDBTL
from .experiment.virtual_lab import VirtualLab
from .viz.plots import make_all_figures


# ===================== 数据管线 =====================

def data_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    subsample_size: Optional[int] = 8000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """数据管线：下载 → 解析 → 子采样 → 划分。

    Returns:
        (train_df, val_df, test_df)
    """
    print("[1/4] 下载数据 ...")
    download_gb1(raw_dir)
    download_reference(raw_dir)

    print("[2/4] 数据处理+划分 ...")
    split_main(
        raw_csv=f"{raw_dir}/SPG1_STRSG_Olson_2014.csv",
        processed_dir=processed_dir,
        subsample_size=subsample_size,
        seed=seed,
    )

    print("[3/4] 加载划分后的数据 ...")
    train_df = pd.read_csv(f"{processed_dir}/train.csv")
    val_df = pd.read_csv(f"{processed_dir}/val.csv")
    test_df = pd.read_csv(f"{processed_dir}/test.csv")
    print(f"  train={len(train_df)} / val={len(val_df)} / test={len(test_df)}")

    return train_df, val_df, test_df


# ===================== 模型训练 =====================

def train_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[FeatureConfig] = None,
    save_path: str = "results/models/fitness_model.joblib",
) -> FitnessModel:
    """模型训练管线：训练 → CV 评估 → hold-out 评估 → 保存。

    Returns:
        训练好的 FitnessModel
    """
    if config is None:
        config = FeatureConfig(use_blosum=True, use_onehot=True, use_physchem=True, use_interaction=True)

    print("[1/4] 训练 Stacking 模型 ...")
    model = create_default_model(config)
    info = model.train(train_df)
    print(f"  训练样本: {info['n_samples']}, 特征维度: {info['feature_dim']}")
    print(f"  使用模型: {info['models']}")
    if info.get("meta_model"):
        print(f"  元模型: {info['meta_model']}")

    print("[2/4] GroupKFold(5) 交叉验证 ...")
    from sklearn.model_selection import GroupKFold

    all_train = pd.concat([train_df, val_df], ignore_index=True)
    def group_key(mutant):
        if not mutant or mutant == "wt":
            return "wt"
        positions = []
        for token in mutant.split(":"):
            if len(token) >= 3:
                try:
                    positions.append(int(token[1:-1]))
                except:
                    pass
        return ":".join(str(p) for p in sorted(positions))

    groups = all_train["mutant"].apply(group_key).values
    n_splits = min(5, len(set(groups)))
    gkf = GroupKFold(n_splits=n_splits)
    fold_spearmans = []
    for fold_i, (train_idx, val_idx) in enumerate(gkf.split(all_train, groups=groups), 1):
        fold_model = create_default_model(config)
        fold_model.train(all_train.iloc[train_idx])
        y_pred = fold_model.predict(all_train.iloc[val_idx])
        m = evaluate_all(all_train.iloc[val_idx]["DMS_score"].values, y_pred, k=10)
        fold_spearmans.append(m["spearman"])
        print(f"  Fold {fold_i}: spearman={m['spearman']:.4f}")
    print(f"  CV Spearman: mean={np.mean(fold_spearmans):.4f}, std={np.std(fold_spearmans):.4f}")

    print("[3/4] hold-out 评估 ...")
    all_metrics = {}
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df) == 0:
            continue
        y_true = df["DMS_score"].values
        y_pred = model.predict(df)
        metrics = evaluate_all(y_true, y_pred, k=10)
        all_metrics[split_name] = metrics
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"  [{split_name:>5}] n={len(df):4d}  {metrics_str}")

    print("[4/4] 保存模型 ...")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"  模型: {save_path}")

    metrics_path = Path("results/model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  指标: {metrics_path}")

    return model


# ===================== Agent 演示 =====================

def agent_demo(
    model: FitnessModel,
    train_df: pd.DataFrame,
    output_path: str = "results/agent_demo.json",
) -> dict:
    """Agent 单轮推理演示（无 KB vs 有 KB）。

    Returns:
        Agent 推理结果字典
    """
    results = {}
    for use_kb in [False, True]:
        label = "llm_agent_kb" if use_kb else "llm_agent"
        print(f"\n--- {label} ---")

        orchestrator = AgentOrchestrator(
            model=model, backend=MockBackend(), use_knowledge=use_kb,
            n_candidates=30, topk=5, max_mutations=3,
        )
        result = orchestrator.run_round(train_df, round_number=1, method=label)

        print(f"  Top positions: {result.analyst_report['position_frequency']}")
        for h in result.hypothesis_report["hypotheses"]:
            print(f"  位点 {h['position']} ({h['current_aa']}->{h['suggested_aas']})")
        for r in result.topk:
            print(f"  #{r['rank']} {r['mutant']}: pred={r.get('predicted_fitness', 0):.3f}")

        results[label] = result.to_dict()

    # 保存完整结果
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results.get("llm_agent_kb", results), f, ensure_ascii=False, indent=2)
    print(f"\n结果保存至: {output_path}")

    return results


# ===================== 对比实验 =====================

def comparison_experiment(
    model: FitnessModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_rounds: int = 3,
    topk: int = 10,
    results_dir: str = "results/comparison",
) -> dict:
    """4 方法 × N 轮对比实验。

    Returns:
        各方法的实验结果字典
    """
    results = run_comparison(
        train_df=train_df, test_df=test_df, model=model,
        backend=MockBackend(),
        methods=["random", "model_only", "llm_agent", "llm_agent_kb"],
        n_rounds=n_rounds, n_candidates=50, topk=topk,
        test_pool_size=200, seed=42,
        results_dir=results_dir,
    )

    print(f"\n{'方法':<18} {'R1':>10} {'R2':>10} {'R3':>10}")
    for method, rounds in results.items():
        means = [r.get("metrics", {}).get("topk_mean", 0) for r in rounds]
        means_str = "  ".join(f"{m:>10.3f}" for m in means)
        print(f"{method:<18} {means_str}")

    return results


# ===================== 闭环 DBTL =====================

def closed_loop_experiment(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model: Optional[FitnessModel] = None,
    acquisition: str = "mean",
    n_rounds: int = 3,
    topk: int = 10,
    results_dir: str = "results/closed_loop",
) -> list[dict]:
    """闭环 DBTL 迭代实验。

    Returns:
        每轮结果摘要列表
    """
    if model is None:
        model = create_default_model()
        model.train(train_df)

    closed_loop = ClosedLoopDBTL(
        model=model, train_df=train_df, test_df=test_df,
        n_rounds=n_rounds, topk=topk,
        acquisition=acquisition, seed=42,
    )
    results = closed_loop.run(results_dir=f"{results_dir}/{acquisition}")
    return results


# ===================== 可视化 =====================

def generate_figures():
    """生成全部图表。"""
    make_all_figures()


# ===================== 全流程 =====================

def run_full_pipeline(
    subsample_size: Optional[int] = 8000,
    seed: int = 42,
    skip_download: bool = False,
) -> dict:
    """全流程：数据 → 模型 → Agent → 对比 → 图表。

    Returns:
        全流程结果摘要
    """
    # 1. 数据
    if skip_download:
        print("=" * 60)
        print("[1/5] 跳过下载，加载已有数据 ...")
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")
    else:
        print("=" * 60)
        print("[1/5] 数据管线 ...")
        train_df, val_df, test_df = data_pipeline(subsample_size=subsample_size, seed=seed)

    # 2. 模型
    print("\n" + "=" * 60)
    print("[2/5] 模型训练 ...")
    model = train_pipeline(train_df, val_df, test_df)

    # 3. Agent 演示
    print("\n" + "=" * 60)
    print("[3/5] Agent 演示 ...")
    agent_demo(model, train_df)

    # 4. 对比实验
    print("\n" + "=" * 60)
    print("[4/5] 对比实验 ...")
    comparison_experiment(model, train_df, test_df)

    # 5. 图表
    print("\n" + "=" * 60)
    print("[5/5] 生成图表 ...")
    generate_figures()

    print("\n" + "=" * 60)
    print("全流程完成！")
    print(f"  模型: results/models/fitness_model.joblib")
    print(f"  Agent 演示: results/agent_demo.json")
    print(f"  对比实验: results/comparison/comparison.csv")
    print(f"  图表: results/figures/*.png")

    return {"status": "ok", "train_size": len(train_df), "test_size": len(test_df)}
