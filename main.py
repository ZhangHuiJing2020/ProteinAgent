"""蛋白质定向进化科学智能体

参考 EvolvePro (github.com/mat10d/EvolvePro)

用法:
    python main.py                            # 全流程
    python main.py train                      # 仅训练
    python main.py demo                       # Agent 演示
    python main.py compare                     # 对比实验
    python main.py closed-loop -a ucb          # 闭环 DBTL
    python main.py zeroshot                    # ESM-2 零样本
    python main.py conservation                # 保守位点分析
    python main.py mutation-count              # 突变数对比
    python main.py active-learning             # UCB/EI 对比
    python main.py interactive                 # 交互式 demo
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from protein_evo.pipeline import (
    data_pipeline, train_pipeline, agent_demo,
    comparison_experiment, closed_loop_experiment,
    generate_figures, run_full_pipeline,
)
from protein_evo.data.wt import WT_SEQUENCE, FULL_LENGTH_SEQUENCE
from protein_evo.data.loader import parse_mutant_str, load_variants
from protein_evo.model.fitness_model import FitnessModel, create_default_model
from protein_evo.model.metrics import evaluate_all
from protein_evo.agent.orchestrator import AgentOrchestrator
from protein_evo.agent.llm_backend import MockBackend
from protein_evo.experiment.virtual_lab import VirtualLab
import numpy as np


def cmd_pipeline(args):
    """全流程"""
    run_full_pipeline(
        subsample_size=args.subsample,
        seed=args.seed,
        skip_download=args.skip_download,
    )


def cmd_train(args):
    """仅训练"""
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    train_pipeline(train_df, val_df, test_df)


def cmd_demo(args):
    """Agent 演示"""
    model = FitnessModel.load("results/models/fitness_model.joblib")
    train_df = pd.read_csv("data/processed/train.csv")
    agent_demo(model, train_df)


def cmd_compare(args):
    """对比实验"""
    model = FitnessModel.load("results/models/fitness_model.joblib")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    comparison_experiment(model, train_df, test_df)


def cmd_figures(args):
    """生成图表"""
    generate_figures()


def cmd_closed_loop(args):
    """闭环 DBTL"""
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    closed_loop_experiment(
        train_df, test_df,
        acquisition=args.acquisition,
        n_rounds=args.rounds, topk=args.topk,
    )


def cmd_zeroshot(args):
    """ESM-2 零样本评估"""
    from protein_evo.features.esm2 import esm2_zeroshot_score, _check_esm_available

    test_df = pd.read_csv("data/processed/test.csv")
    sequences = []
    for mutant in test_df["mutant"].astype(str):
        if not mutant or mutant == "wt":
            sequences.append(FULL_LENGTH_SEQUENCE)
            continue
        seq = list(FULL_LENGTH_SEQUENCE)
        mutations = parse_mutant_str(mutant, FULL_LENGTH_SEQUENCE)
        for m in mutations:
            if 0 <= m.position < len(seq):
                seq[m.position] = m.mut_aa
        sequences.append("".join(seq))

    y_true = test_df["DMS_score"].values
    esm_available = _check_esm_available()
    y_zeroshot = esm2_zeroshot_score(sequences, FULL_LENGTH_SEQUENCE, model_name=args.esm_model) if esm_available \
        else np.random.RandomState(42).normal(0, 0.1, len(sequences))

    zeroshot_metrics = evaluate_all(y_true, y_zeroshot, k=10)
    print(f"ESM-2 零样本: {zeroshot_metrics}")

    supervised_metrics = {}
    model_path = Path("results/models/fitness_model.joblib")
    if model_path.exists():
        model = FitnessModel.load(model_path)
        y_supervised = model.predict(test_df)
        supervised_metrics = evaluate_all(y_true, y_supervised, k=10)
        print(f"Stacking 有监督: {supervised_metrics}")

    with open("results/zeroshot_eval.json", "w", encoding="utf-8") as f:
        json.dump({"esm2_zeroshot": zeroshot_metrics, "stacking_supervised": supervised_metrics,
                   "esm_available": esm_available}, f, indent=2)


def cmd_mutation_count(args):
    """突变数对比分析"""
    model = FitnessModel.load("results/models/fitness_model.joblib")
    test_df = pd.read_csv("data/processed/test.csv")
    results = {}
    for n_mut in [1, 2, 3, 4]:
        sub = test_df[test_df["n_mutations"] == n_mut]
        if len(sub) == 0:
            continue
        preds = model.predict(sub)
        metrics = evaluate_all(sub["DMS_score"].values, np.asarray(preds), k=min(10, len(sub)))
        results[f"{n_mut}_point"] = {"n_samples": len(sub), **metrics}
        print(f"{n_mut}点: n={len(sub)}, spearman={metrics['spearman']:.3f}")
    with open("results/mutation_count_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def cmd_conservation(args):
    """保守位点分析"""
    from protein_evo.knowledge.amino_acid_props import get_category

    df = load_variants("data/processed/train.csv")

    def has_position(mutant_str, pos_1):
        if not mutant_str or mutant_str == "wt":
            return False
        for token in mutant_str.split(":"):
            token = token.strip()
            if len(token) >= 3:
                try:
                    if int(token[1:-1]) == pos_1:
                        return True
                except ValueError:
                    continue
        return False

    single = df[df["n_mutations"] == 1]
    all_positions = set()
    for mutant in single["mutant"].astype(str):
        for token in mutant.split(":"):
            if len(token) >= 3:
                try:
                    all_positions.add(int(token[1:-1]))
                except ValueError:
                    continue

    sensitivity_rows = []
    for pos_1 in sorted(all_positions):
        sub = single[single["mutant"].apply(lambda m: has_position(m, pos_1))]
        if len(sub) == 0:
            continue
        fitness = sub["DMS_score"].values
        wt_aa = FULL_LENGTH_SEQUENCE[pos_1 - 1] if pos_1 - 1 < len(FULL_LENGTH_SEQUENCE) else "X"
        sensitivity_rows.append({
            "position": pos_1, "wt_aa": wt_aa,
            "fitness_mean": float(fitness.mean()), "fitness_std": float(fitness.std()),
            "sensitivity": float(abs(fitness.mean()) + fitness.std()),
        })
    sensitivity = pd.DataFrame(sensitivity_rows)
    if len(sensitivity) > 0:
        median_sens = sensitivity["sensitivity"].median()
        sensitivity["type"] = sensitivity["sensitivity"].apply(lambda x: "高敏感性" if x > median_sens else "保守")
    print(f"位点敏感性: {len(sensitivity)} 个位点")

    with open("results/conservation_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"position_sensitivity": sensitivity.to_dict(orient="records")}, f, indent=2, default=str)


def cmd_active_learning(args):
    """UCB置信度上界和EI 主动学习对比"""
    model = FitnessModel.load("results/models/fitness_model.joblib")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    all_results = {}
    for strategy in ["mean", "ucb", "ei"]:
        orchestrator = AgentOrchestrator(
            model=model, backend=MockBackend(), use_knowledge=False,
            n_candidates=50, topk=10, max_mutations=3,
            acquisition=strategy, ucb_beta=2.0,
        )
        virtual_lab = VirtualLab.from_test_df(test_df, model, pool_size=200, seed=42)
        multi_results = orchestrator.run_multi_round(
            train_df, n_rounds=3, method=f"agent_{strategy}",
            virtual_lab=virtual_lab, results_dir=f"results/active_learning/{strategy}",
        )
        all_results[strategy] = [r.to_dict() for r in multi_results]
        means = [r.get("metrics", {}).get("topk_mean", 0) for r in all_results[strategy]]
        print(f"{strategy}: R1={means[0]:.3f} R2={means[1]:.3f} R3={means[2]:.3f}")

    with open("results/active_learning/summary.json", "w", encoding="utf-8") as f:
        json.dump({s: {"topk_mean_per_round": [r.get("metrics", {}).get("topk_mean", 0) for r in v]}
                    for s, v in all_results.items()}, f, indent=2)


def cmd_interactive(args):
    """交互式 demo"""
    from protein_evo.knowledge.amino_acid_props import is_standard
    model = FitnessModel.load("results/models/fitness_model.joblib")

    print("输入蛋白质序列（或 quit 退出，default 使用 GB1 野生型）:")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            break
        if user_input.lower() == "default":
            user_input = WT_SEQUENCE
        if not all(is_standard(aa) for aa in user_input):
            print("包含非标准氨基酸")
            continue

        history_df = pd.DataFrame([{"mutant": "wt", "mutated_sequence": user_input,
                                     "DMS_score": 0.0, "n_mutations": 0}])
        orchestrator = AgentOrchestrator(
            model=model, backend=MockBackend(), use_knowledge=args.kb,
            n_candidates=30, topk=5, max_mutations=3,
        )
        result = orchestrator.run_round(history_df, round_number=1, method="demo")
        for rec in result.topk:
            print(f"  {rec['mutant']}: pred={rec.get('predicted_fitness', 0):.3f}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="蛋白质定向进化科学智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n    python main.py              # 全流程\n    python main.py train        # 仅训练\n    python main.py closed-loop -a ucb",
    )
    subparsers = parser.add_subparsers(dest="command")

    p = subparsers.add_parser("pipeline", help="全流程")
    p.add_argument("--subsample", type=int, default=8000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true")

    subparsers.add_parser("train", help="训练模型")
    subparsers.add_parser("demo", help="Agent 演示")
    subparsers.add_parser("compare", help="对比实验")
    subparsers.add_parser("figures", help="生成图表")

    p = subparsers.add_parser("closed-loop", help="闭环 DBTL")
    p.add_argument("--acquisition", "-a", default="mean", choices=["mean", "ucb", "ei"])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--topk", type=int, default=10)

    p = subparsers.add_parser("zeroshot", help="ESM-2 零样本")
    p.add_argument("--esm-model", default="esm2_t12_35M_UR50D")

    subparsers.add_parser("mutation-count", help="突变数对比")
    subparsers.add_parser("conservation", help="保守位点分析")
    subparsers.add_parser("active-learning", help="UCB/EI 对比")

    p = subparsers.add_parser("interactive", help="交互式 demo")
    p.add_argument("--kb", action="store_true")

    args = parser.parse_args()

    dispatch = {
        "pipeline": cmd_pipeline, "train": cmd_train, "demo": cmd_demo,
        "compare": cmd_compare, "figures": cmd_figures, "closed-loop": cmd_closed_loop,
        "zeroshot": cmd_zeroshot, "mutation-count": cmd_mutation_count,
        "conservation": cmd_conservation, "active-learning": cmd_active_learning,
        "interactive": cmd_interactive,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    elif args.command is None:
        run_full_pipeline()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
