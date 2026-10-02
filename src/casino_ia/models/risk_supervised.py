"""Modelo supervisado y calibrado de riesgo de juego responsable.

Entrena exclusivamente con la ABT temporal de Transparency Project. La selección
de modelo y umbrales usa validación; el conjunto test queda reservado para la
evaluación final. Los folds de calibración están agrupados por cliente.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from casino_ia.features.risk_supervised import TARGET_COLUMN, validate_risk_abt

MODEL_VERSION = "risk-supervised-v1.0.0"
FEATURE_SET_VERSION = "risk-transferable-v1"

# Variables que tienen una correspondencia operacional razonable entre el
# historico externo y las sesiones del casino. Se excluyen tipos de producto
# propios del operador externo y atributos demograficos/sensibles.
RISK_FEATURES_TRANSFERABLE = [
    "ActivityRows",
    "ActiveDays",
    "MonetaryObservedRows",
    "MissingMonetaryPct",
    "MissingBetsRows",
    "NumberOfBetsTotal",
    "AvgBetsPerActiveDay",
    "MaxDailyBets",
    "StdDailyBets",
    "NumberOfBetsLast30",
    "NumberOfBetsPrevious60",
    "BetsTrendRatio",
    "TurnoverTotal",
    "AvgDailyTurnover",
    "MaxDailyTurnover",
    "StdDailyTurnover",
    "TurnoverLast30",
    "TurnoverPrevious60",
    "TurnoverTrendRatio",
    "NetHoldTotal",
    "GrossLossTotal",
    "GrossWinTotal",
    "MaxDailyLoss",
    "StdDailyHold",
    "LossDays",
    "ActiveDaysLast30",
    "ActiveDaysPrevious60",
    "ActiveDaysTrendRatio",
]


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    """Metricas de discriminacion, calibracion y lift para una clase minoritaria."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    prevalence = float(y.mean())
    top_count = max(1, int(np.ceil(len(y) * 0.10)))
    top_indices = np.argsort(-p)[:top_count]
    top_rate = float(y[top_indices].mean())

    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(p, bins, right=True) - 1, 0, 9)
    calibration_error = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            calibration_error += float(mask.mean()) * abs(float(y[mask].mean() - p[mask].mean()))

    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(prevalence, 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6),
        "pr_auc": round(float(average_precision_score(y, p)), 6),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p)), 6),
        "ece_10_bins": round(calibration_error, 6),
        "lift_top_decile": round(top_rate / prevalence, 6) if prevalence else None,
    }


def select_risk_thresholds(
    y_validation: np.ndarray,
    probabilities: np.ndarray,
    *,
    medium_recall_target: float = 0.80,
    high_recall_target: float = 0.50,
) -> dict[str, float]:
    """Selecciona umbrales usando solamente validacion.

    El umbral medio captura al menos el recall objetivo entre Medio+Alto. El alto
    busca el mejor precision con recall minimo para formar una cohorte prioritaria.
    """
    y = np.asarray(y_validation, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    thresholds = np.unique(p)
    if not len(thresholds):
        raise ValueError("No hay umbrales disponibles para seleccionar")

    def choose(minimum_recall: float) -> float:
        recalls = np.array([recall_score(y, p >= threshold) for threshold in thresholds])
        precisions = np.array(
            [precision_score(y, p >= threshold, zero_division=0) for threshold in thresholds]
        )
        eligible = np.flatnonzero(recalls >= minimum_recall)
        if not len(eligible):
            return float(thresholds.min())
        best_precision = precisions[eligible].max()
        best = eligible[np.isclose(precisions[eligible], best_precision)]
        selected = float(thresholds[best].max())
        # SQL persiste 10 decimales. Cuantizar hacia abajo evita que un redondeo
        # ascendente excluya observaciones empatadas y reduzca el recall objetivo.
        quantized = float(np.floor((selected - 1e-9) * 10**10) / 10**10)
        if recall_score(y, p >= quantized) >= minimum_recall:
            return max(0.0, quantized)
        for candidate in thresholds[::-1]:
            quantized = float(np.floor((float(candidate) - 1e-9) * 10**10) / 10**10)
            if recall_score(y, p >= quantized) >= minimum_recall:
                return max(0.0, quantized)
        return 0.0

    medium = choose(medium_recall_target)
    high = choose(high_recall_target)
    high = max(high, medium)
    return {"medium": medium, "high": high}


def risk_levels(probabilities: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    """Convierte probabilidad calibrada en Bajo, Medio o Alto."""
    p = np.asarray(probabilities, dtype=float)
    return np.select(
        [p >= thresholds["high"], p >= thresholds["medium"]],
        ["Alto", "Medio"],
        default="Bajo",
    )


def threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(probabilities) >= threshold
    matrix = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "threshold": round(float(threshold), 10),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 6),
        "predicted_positive": int(pred.sum()),
        "confusion_matrix": matrix.tolist(),
    }


def tier_summary(
    y_true: np.ndarray, probabilities: np.ndarray, thresholds: dict[str, float]
) -> dict:
    frame = pd.DataFrame(
        {
            "Target": np.asarray(y_true, dtype=int),
            "Probability": np.asarray(probabilities, dtype=float),
            "RiskLevel": risk_levels(probabilities, thresholds),
        }
    )
    summary = {}
    for level in ("Bajo", "Medio", "Alto"):
        subset = frame.loc[frame["RiskLevel"] == level]
        summary[level] = {
            "rows": int(len(subset)),
            "positives": int(subset["Target"].sum()),
            "observed_event_rate": round(float(subset["Target"].mean()), 6)
            if len(subset)
            else None,
            "mean_probability": round(float(subset["Probability"].mean()), 6)
            if len(subset)
            else None,
        }
    return summary


@dataclass
class ModeloRiesgoSupervisado:
    """Entrena, selecciona y conserva un clasificador calibrado de riesgo."""

    features: list[str] | None = None
    random_state: int = 42
    calibration_folds: int = 5
    model_: object | None = None
    model_name_: str | None = None
    thresholds_: dict[str, float] | None = None
    metrics_: dict | None = None
    feature_importance_: dict[str, float] | None = None

    def __post_init__(self) -> None:
        self.features = self.features or list(RISK_FEATURES_TRANSFERABLE)

    def _candidates(self) -> dict[str, Pipeline]:
        logistic = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        gradient_boosting = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=250,
                        max_leaf_nodes=15,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        class_weight="balanced",
                        early_stopping=False,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )
        return {"logistic_regression": logistic, "hist_gradient_boosting": gradient_boosting}

    def _fit_calibrated(
        self, estimator: Pipeline, x_train: pd.DataFrame, y_train: np.ndarray, groups: np.ndarray
    ) -> CalibratedClassifierCV:
        n_splits = min(self.calibration_folds, len(np.unique(groups)))
        if n_splits < 2:
            raise ValueError("Se requieren al menos dos clientes para calibracion agrupada")
        folds = list(GroupKFold(n_splits=n_splits).split(x_train, y_train, groups))
        calibrated = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=folds,
            n_jobs=-1,
            ensemble=True,
        )
        calibrated.fit(x_train, y_train)
        return calibrated

    def fit_select_evaluate(self, abt: pd.DataFrame) -> ModeloRiesgoSupervisado:
        validation = validate_risk_abt(abt)
        missing_features = sorted(set(self.features).difference(abt.columns))
        if missing_features:
            raise ValueError(f"Faltan features del modelo: {missing_features}")

        partitions = {
            name: abt.loc[abt["SplitSet"] == name].reset_index(drop=True)
            for name in ("train", "validation", "test")
        }
        train = partitions["train"]
        val = partitions["validation"]
        test = partitions["test"]
        x_train = train[self.features]
        y_train = train[TARGET_COLUMN].astype(int).to_numpy()
        groups = train["UserID"].to_numpy()

        fitted: dict[str, CalibratedClassifierCV] = {}
        candidate_metrics = {}
        for name, estimator in self._candidates().items():
            model = self._fit_calibrated(estimator, x_train, y_train, groups)
            probabilities = model.predict_proba(val[self.features])[:, 1]
            fitted[name] = model
            candidate_metrics[name] = evaluate_probabilities(
                val[TARGET_COLUMN].astype(int).to_numpy(), probabilities
            )

        self.model_name_ = max(
            candidate_metrics,
            key=lambda name: (
                candidate_metrics[name]["pr_auc"],
                -candidate_metrics[name]["brier"],
            ),
        )
        self.model_ = fitted[self.model_name_]

        y_val = val[TARGET_COLUMN].astype(int).to_numpy()
        p_val = self.predict_proba(val)
        self.thresholds_ = select_risk_thresholds(y_val, p_val)

        y_test = test[TARGET_COLUMN].astype(int).to_numpy()
        p_test = self.predict_proba(test)
        importance = permutation_importance(
            self.model_,
            val[self.features],
            y_val,
            scoring="average_precision",
            n_repeats=8,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.feature_importance_ = dict(
            sorted(
                zip(self.features, importance.importances_mean, strict=True),
                key=lambda pair: -pair[1],
            )
        )

        self.metrics_ = {
            "model_version": MODEL_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "selected_model": self.model_name_,
            "data_validation": validation.as_dict(),
            "candidate_validation": candidate_metrics,
            "thresholds_selected_on_validation": self.thresholds_,
            "validation": {
                **evaluate_probabilities(y_val, p_val),
                "medium_or_higher": threshold_metrics(y_val, p_val, self.thresholds_["medium"]),
                "high": threshold_metrics(y_val, p_val, self.thresholds_["high"]),
                "tiers": tier_summary(y_val, p_val, self.thresholds_),
            },
            "test": {
                **evaluate_probabilities(y_test, p_test),
                "medium_or_higher": threshold_metrics(y_test, p_test, self.thresholds_["medium"]),
                "high": threshold_metrics(y_test, p_test, self.thresholds_["high"]),
                "tiers": tier_summary(y_test, p_test, self.thresholds_),
            },
            "permutation_importance_validation_pr_auc": {
                key: round(float(value), 8) for key, value in self.feature_importance_.items()
            },
        }
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("El modelo no esta entrenado")
        return self.model_.predict_proba(frame[self.features])[:, 1]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.thresholds_ is None:
            raise RuntimeError("El modelo no tiene umbrales definidos")
        probabilities = self.predict_proba(frame)
        return pd.DataFrame(
            {
                "FrameId": frame["FrameId"].to_numpy(),
                "UserID": frame["UserID"].to_numpy(),
                "CutoffDate": pd.to_datetime(frame["CutoffDate"]).dt.date,
                "SplitSet": frame["SplitSet"].to_numpy(),
                "ActualTarget": frame[TARGET_COLUMN].astype(int).to_numpy(),
                "RiskProbability": probabilities,
                "RiskLevel": risk_levels(probabilities, self.thresholds_),
            }
        )

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ModeloRiesgoSupervisado:
        return joblib.load(path)
