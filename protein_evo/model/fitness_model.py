"""适应度预测模型。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from xgboost import XGBRegressor

from ..data.loader import parse_mutant_str, infer_mutations_from_sequence
from ..data.wt import WT_SEQUENCE, FULL_LENGTH_SEQUENCE
from ..features.blosum import encode_mutant_str as blosum_encode, blosum_score
from ..features.onehot import encode_mutant_str as onehot_encode, ONEHOT_DIM
from ..features.physchem import encode_mutant_str as physchem_encode
from ..features.interaction import encode_mutant_str as interaction_encode, INTERACTION_DIM


@dataclass
class FeatureConfig:
    """特征工程配置。"""
    use_blosum: bool = True
    use_onehot: bool = True
    use_physchem: bool = True
    use_interaction: bool = True
    use_esm2: bool = False  # ESM-2 嵌入+零样本打分（参考 EVOLVEpro）
    # ESM-2 嵌入维度（640 维嵌入 + 1 维 LLR = 641）
    ESM2_FEATURE_DIM: int = 641

    @property
    def dim(self) -> int:
        d = 0
        if self.use_blosum:
            d += len(FULL_LENGTH_SEQUENCE)  # 448（全长蛋白 G）
        if self.use_onehot:
            d += ONEHOT_DIM  # 位点数 × 20 AA（动态适配）
        if self.use_physchem:
            d += 6  # 理化性质变化
        if self.use_interaction:
            d += INTERACTION_DIM  # 25 维位点 pair 交互
        if self.use_esm2:
            d += self.ESM2_FEATURE_DIM  # ESM-2 嵌入(640) + LLR(1) = 641
        return d


def encode_record(
    mutant_str: str,
    mutated_seq: Optional[str] = None,
    config: FeatureConfig = FeatureConfig(),
) -> np.ndarray:
    """编码单条记录为特征向量。"""
    parts = []
    if config.use_blosum:
        parts.append(blosum_encode(mutant_str, WT_SEQUENCE))
    if config.use_onehot:
        parts.append(onehot_encode(mutant_str))
    if config.use_physchem:
        parts.append(physchem_encode(mutant_str))
    if config.use_interaction:
        parts.append(interaction_encode(mutant_str))
    if config.use_esm2:
        # ESM-2 特征（参考 EVOLVEpro）
        from ..features.esm2 import encode_sequences_with_esm2
        esm2_feat = encode_sequences_with_esm2([mutant_str], FULL_LENGTH_SEQUENCE)[0]
        parts.append(esm2_feat)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def encode_df(df: pd.DataFrame, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """批量编码 DataFrame（对大数据集做了优化）。"""
    # 优先用 mutant 字段；无 mutant 或为 "wt" 的从序列推断
    mutant_col = df.get("mutant")
    if mutant_col is not None:
        mutant_strs = mutant_col.astype(str).tolist()
    else:
        from ..data.loader import infer_mutations_from_sequence, mutations_to_str
        mutant_strs = []
        for seq in df["mutated_sequence"].astype(str):
            muts = infer_mutations_from_sequence(seq, WT_SEQUENCE)
            mutant_strs.append(mutations_to_str(muts))

    # 手工特征批量编码
    rows = [encode_record(ms, "", config) for ms in mutant_strs]
    manual_features = np.stack(rows) if rows else np.zeros((0, config.dim), dtype=np.float32)

    # ESM-2 特征批量编码（如果开启）
    if config.use_esm2:
        from ..features.esm2 import encode_sequences_with_esm2
        esm2_features = encode_sequences_with_esm2(mutant_strs, FULL_LENGTH_SEQUENCE)
        manual_features = np.concatenate([manual_features, esm2_features], axis=1)

    return manual_features


def create_default_model(config: 'FeatureConfig' = None) -> 'FitnessModel':
    """创建默认配置的 Stacking 模型（工厂函数）。

    统一 03/08/11 脚本的模型初始化参数，避免重复代码。
    """
    if config is None:
        config = FeatureConfig(use_blosum=True, use_onehot=True, use_physchem=True, use_interaction=True)
    return FitnessModel(
        model_type="stacking",
        config=config,
        rf_params={
            "n_estimators": 500, "max_depth": 10,
            "min_samples_split": 5, "min_samples_leaf": 2,
            "max_features": "sqrt", "random_state": 42, "n_jobs": -1,
        },
        xgb_params={
            "n_estimators": 500, "max_depth": 5,
            "min_child_weight": 3, "reg_alpha": 0.2, "reg_lambda": 1.0,
            "subsample": 0.85, "colsample_bytree": 0.75,
            "learning_rate": 0.05, "random_state": 42, "n_jobs": -1,
        },
        gb_params={
            "n_estimators": 300, "max_depth": 4,
            "learning_rate": 0.05, "subsample": 0.8,
            "random_state": 42,
        },
        ridge_params={"alphas": [0.1, 0.5, 1.0, 5.0, 10.0]},
    )


class FitnessModel:
    """适应度预测模型。

    支持 rf / xgboost / ensemble（简单平均）/ stacking（RidgeCV meta-learner）。
    """

    def __init__(
        self,
        model_type: str = "rf",
        config: FeatureConfig = FeatureConfig(),
        rf_params: Optional[dict] = None,
        xgb_params: Optional[dict] = None,
        gb_params: Optional[dict] = None,
        ridge_params: Optional[dict] = None,
    ):
        self.model_type = model_type
        self.config = config
        self.rf_params = rf_params or {"n_estimators": 200, "max_depth": 12, "random_state": 42}
        self.xgb_params = xgb_params or {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
        self.gb_params = gb_params or {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "random_state": 42}
        self.ridge_params = ridge_params or {"alphas": [0.1, 0.5, 1.0, 5.0, 10.0]}
        self._models: dict = {}
        self._meta_model = None  # Stacking 元模型
        self._is_trained = False
        self._use_stacking = False

    @property
    def feature_dim(self) -> int:
        return self.config.dim

    def _init_models(self):

        gpu_available = False
        try:
            import subprocess
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=5)
            gpu_available = True
        except Exception:
            gpu_available = False

        if self.model_type in ("rf", "ensemble", "stacking"):
            rf_params = {**self.rf_params, "n_jobs": -1}
            self._models["rf"] = RandomForestRegressor(**rf_params)
        if self.model_type in ("xgboost", "ensemble", "stacking"):
            xgb_params = {**self.xgb_params}
            if gpu_available:
                xgb_params["device"] = "cuda"
                xgb_params["tree_method"] = "hist"
                print("  [XGBoost] 使用 GPU (cuda) 训练")
            else:
                xgb_params["n_jobs"] = -1
                print("  [XGBoost] 使用 CPU 训练（未检测到 GPU）")
            self._models["xgboost"] = XGBRegressor(**xgb_params)
        if self.model_type == "stacking":
            # GradientBoosting: scikit-learn 不支持 GPU
            self._models["gb"] = GradientBoostingRegressor(**self.gb_params)
            self._models["ridge"] = Ridge(alpha=1.0, random_state=42)

    def train(self, df: pd.DataFrame) -> dict:
        """训练模型。

        Args:
            df: 含 'mutant', 'mutated_sequence', 'DMS_score' 列

        Returns:
            训练信息字典（含训练样本数、特征维度）
        """
        X = encode_df(df, self.config)
        y = df["DMS_score"].values.astype(np.float32)

        if len(X) == 0:
            raise ValueError("训练数据为空")

        self._init_models()
        train_info = {"n_samples": len(X), "feature_dim": X.shape[1], "models": []}

        if self.model_type == "stacking":
            # Stacking: 用 5-fold OOF 训练元模型
            self._use_stacking = True
            n_splits = min(5, len(X) // 10)  # 至少每折 10 样本
            n_splits = max(2, n_splits)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

            n_base = len(self._models)
            oof_preds = np.zeros((len(X), n_base), dtype=np.float32)

            for name, model in self._models.items():
                model_idx = list(self._models.keys()).index(name)
                for train_idx, val_idx in kf.split(X):
                    # 克隆模型（用 sklearn clone）
                    from sklearn.base import clone
                    fold_model = clone(model)
                    fold_model.fit(X[train_idx], y[train_idx])
                    oof_preds[val_idx, model_idx] = fold_model.predict(X[val_idx])
                # 用全量数据训练最终 base 模型
                model.fit(X, y)
                train_info["models"].append(name)

            # 训练元模型（RidgeCV 自动选 alpha）
            self._meta_model = RidgeCV(**self.ridge_params)
            self._meta_model.fit(oof_preds, y)
            train_info["meta_model"] = "RidgeCV"
            train_info["meta_coef"] = self._meta_model.coef_.tolist()
        else:
            for name, model in self._models.items():
                model.fit(X, y)
                train_info["models"].append(name)

        self._is_trained = True
        return train_info

    def predict(self, df_or_mutant: pd.DataFrame | str) -> np.ndarray | float:
        """预测 fitness。

        Args:
            df_or_mutant: DataFrame 批量预测 或 单条 mutant 字符串

        Returns:
            np.ndarray（批量）或 float（单条）
        """
        if not self._is_trained:
            raise RuntimeError("模型未训练")

        single = isinstance(df_or_mutant, str)
        if single:
            df_or_mutant = pd.DataFrame([{"mutant": df_or_mutant, "mutated_sequence": ""}])

        X = encode_df(df_or_mutant, self.config)
        preds = []
        for name, model in self._models.items():
            preds.append(model.predict(X))

        if self._use_stacking and self._meta_model is not None:
            # Stacking: base预测→meta模型
            base_preds = np.stack(preds, axis=1)  # (n_samples, n_base)
            y_pred = self._meta_model.predict(base_preds)
        elif len(preds) > 1:
            # ensemble: 平均
            y_pred = np.mean(np.stack(preds), axis=0)
        else:
            y_pred = preds[0]

        return float(y_pred[0]) if single else y_pred

    def predict_with_uncertainty(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """预测 + 不确定性（基于 RF 树间方差或 ensemble 间方差）。

        Returns:
            (mean, std)
        """
        if not self._is_trained:
            raise RuntimeError("模型未训练")

        X = encode_df(df, self.config)
        preds = []

        # RF: 用每棵树预测求方差
        if "rf" in self._models:
            rf = self._models["rf"]
            tree_preds = np.stack([tree.predict(X) for tree in rf.estimators_])
            preds.append(tree_preds)

        # XGBoost:不确定性=0
        if "xgboost" in self._models:
            xgb = self._models["xgboost"]
            preds.append(xgb.predict(X)[np.newaxis, :])

        all_preds = np.concatenate(preds, axis=0)  # (n_models_or_trees, n_samples)
        mean = all_preds.mean(axis=0)
        std = all_preds.std(axis=0)
        return mean, std

    def save(self, path: str | Path):
        """保存模型。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 保存 sklearn/xgboost 模型
        joblib.dump({
            "model_type": self.model_type,
            "config": asdict(self.config),
            "rf_params": self.rf_params,
            "xgb_params": self.xgb_params,
            "models": self._models,
            "is_trained": self._is_trained,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "FitnessModel":
        """加载模型。"""
        data = joblib.load(path)
        model = cls(
            model_type=data["model_type"],
            config=FeatureConfig(**data["config"]),
            rf_params=data["rf_params"],
            xgb_params=data["xgb_params"],
        )
        model._models = data["models"]
        model._is_trained = data["is_trained"]
        return model


if __name__ == "__main__":
    from .metrics import evaluate_all
    m = FitnessModel(model_type="rf")
    print(f"Feature dim: {m.feature_dim}")
