"""Modelo NBO calibrado con evidencia RCT y campañas semi-sintéticas.

Las tasas agregadas se anclan en Hillstrom y Criteo. La traducción a recompensas
baja/media/alta, la afinidad individual y el valor son explícitamente
semi-sintéticos porque no existe histórico real con esos tres tratamientos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from casino_ia.config import FECHA_CORTE, REWARDS
from casino_ia.models.response import FEATURES_RESPUESTA

MODEL_VERSION = "response-nbo-calibrated-v1.0.0"
SIMULATION_VERSION = "campaign-rct-anchored-v1.0.0"
FEATURE_SET_VERSION = "casino-response-v1"
REWARDS_NBO = ("baja", "media", "alta")
TREATMENTS = ("control", *REWARDS_NBO)


@dataclass(frozen=True)
class ResponseAnchors:
    control_rate: float
    low_rate: float
    medium_rate: float
    high_rate: float
    criteo_relative_lift: float

    def target_rates(self) -> dict[str, float]:
        return {
            "control": self.control_rate,
            "baja": self.low_rate,
            "media": self.medium_rate,
            "alta": self.high_rate,
        }


def build_response_anchors(evidence: pd.DataFrame) -> ResponseAnchors:
    """Construye tasas objetivo sin ocultar la correspondencia semi-sintética."""
    required = {"SourceDataset", "TreatmentLabel", "ObservedRate"}
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise ValueError(f"Faltan columnas de evidencia: {missing}")
    lookup = {
        (str(row.SourceDataset), str(row.TreatmentLabel)): float(row.ObservedRate)
        for row in evidence.itertuples()
    }
    control = lookup[("HillstromEmail", "No E-Mail")]
    mens = lookup[("HillstromEmail", "Mens E-Mail")]
    womens = lookup[("HillstromEmail", "Womens E-Mail")]
    criteo_control = lookup[("CriteoUpliftV21", "0")]
    criteo_treatment = lookup[("CriteoUpliftV21", "1")]
    relative_lift = criteo_treatment / criteo_control
    low = control * relative_lift
    anchors = ResponseAnchors(
        control_rate=control,
        low_rate=low,
        medium_rate=womens,
        high_rate=mens,
        criteo_relative_lift=relative_lift,
    )
    rates = list(anchors.target_rates().values())
    if not all(0 < rate < 1 for rate in rates):
        raise ValueError(f"Tasas objetivo invalidas: {rates}")
    if not (anchors.control_rate < anchors.low_rate < anchors.medium_rate < anchors.high_rate):
        raise ValueError(f"Las tasas no respetan control < baja < media < alta: {rates}")
    return anchors


def calibrate_mean_probability(raw_score: np.ndarray, target_rate: float) -> np.ndarray:
    """Ajusta solo el intercepto logit hasta alcanzar la media objetivo."""
    raw = np.asarray(raw_score, dtype=float)
    if not 0 < target_rate < 1:
        raise ValueError("target_rate debe estar entre 0 y 1")
    low, high = -30.0, 30.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        probability = 1.0 / (1.0 + np.exp(-(raw + midpoint)))
        if probability.mean() < target_rate:
            low = midpoint
        else:
            high = midpoint
    return 1.0 / (1.0 + np.exp(-(raw + (low + high) / 2.0)))


def assign_client_split(client_ids: pd.Series) -> pd.Series:
    """Split determinístico por cliente; todas sus campañas quedan juntas."""
    hashed = pd.util.hash_pandas_object(client_ids.astype(str), index=False).to_numpy()
    buckets = hashed % 100
    return pd.Series(
        np.select([buckets < 70, buckets < 85], ["train", "validation"], default="test"),
        index=client_ids.index,
        dtype="object",
    )


def simulate_rct_anchored_campaigns(
    features: pd.DataFrame,
    base_probability: pd.DataFrame,
    anchors: ResponseAnchors,
    *,
    campaign_count: int = 8,
    random_state: int = 42,
    interaction_strength: float = 0.8,
) -> pd.DataFrame:
    """Genera campañas balanceadas con tasas medias ancladas en RCT reales."""
    required = {
        "IdCliente",
        "ValorTeoricoCasa",
        "NroSesiones",
        "RatioTendenciaCoinIn",
        "PuntosAcumulados",
        "CompsAcumulados",
        *FEATURES_RESPUESTA,
    }
    missing = sorted(required.difference(features.columns))
    if missing:
        raise ValueError(f"Faltan variables de clientes: {missing}")
    base = features.merge(
        base_probability[["IdCliente", "ProbRespuesta"]].rename(
            columns={"ProbRespuesta": "RawBaseProbability"}
        ),
        on="IdCliente",
        validate="one_to_one",
    ).sort_values("IdCliente", ignore_index=True)

    value_rank = base["ValorTeoricoCasa"].rank(pct=True).fillna(0.5)
    points_rank = base["PuntosAcumulados"].rank(pct=True).fillna(0.5)
    comps_rank = base["CompsAcumulados"].rank(pct=True).fillna(0.5)
    middle_affinity = (1.0 - (value_rank - 0.6).abs() / 0.6).clip(0.0, 1.0)
    base["Affinity_baja"] = ((1.0 - value_rank) + points_rank) / 2.0
    base["Affinity_media"] = (middle_affinity + comps_rank) / 2.0
    base["Affinity_alta"] = (value_rank + comps_rank) / 2.0
    base["Affinity_control"] = 0.5

    raw_probability = base["RawBaseProbability"].clip(0.001, 0.999).to_numpy()
    raw_logit = np.log(raw_probability / (1.0 - raw_probability))
    probability_by_type = {}
    for treatment, target_rate in anchors.target_rates().items():
        affinity = base[f"Affinity_{treatment}"].to_numpy()
        heterogeneous_score = raw_logit + interaction_strength * (affinity - 0.5)
        probability_by_type[treatment] = calibrate_mean_probability(
            heterogeneous_score, target_rate
        )

    sessions = pd.to_numeric(base["NroSesiones"], errors="coerce").clip(lower=1)
    trend = (
        pd.to_numeric(base["RatioTendenciaCoinIn"], errors="coerce")
        .fillna(1.0)
        .clip(0.5, 2.0)
    )
    base_value = (
        pd.to_numeric(base["ValorTeoricoCasa"], errors="coerce").fillna(0.0)
        * REWARDS.factor_uplift
        * trend
    ).clip(lower=0.0)
    # Conservamos sesiones en el cálculo para controlar valores degenerados.
    base["ValueBase"] = np.where(sessions > 0, base_value, 0.0)

    rng = np.random.default_rng(random_state)
    start_date = pd.Timestamp(FECHA_CORTE) + pd.Timedelta(days=1)
    rows = []
    treatment_array = np.asarray(TREATMENTS, dtype=object)
    client_order = np.arange(len(base))
    maximum_cost = max(REWARDS.costo.values())

    for campaign_index in range(campaign_count):
        exposed = base.copy()
        exposed["RewardType"] = treatment_array[
            (client_order + campaign_index) % len(treatment_array)
        ]
        exposed["SimulationProbability"] = [
            probability_by_type[treatment][position]
            for position, treatment in enumerate(exposed["RewardType"])
        ]
        exposed["Responded"] = (
            rng.random(len(exposed)) < exposed["SimulationProbability"].to_numpy()
        ).astype(int)
        exposed["Cost"] = exposed["RewardType"].map({"control": 0.0, **REWARDS.costo})
        intensity = exposed["Cost"] / maximum_cost
        affinity = np.array(
            [
                exposed.at[index, f"Affinity_{treatment}"]
                for index, treatment in enumerate(exposed["RewardType"])
            ]
        )
        exposed["ValueIfResponse"] = np.where(
            exposed["RewardType"].eq("control"),
            0.0,
            exposed["ValueBase"] * (1.0 + 0.5 * intensity * affinity),
        )
        exposed["ActualIncrementalValue"] = np.where(
            exposed["Responded"].eq(1), exposed["ValueIfResponse"], 0.0
        )
        exposed["CampaignId"] = f"RCT-SIM-{campaign_index + 1:02d}"
        exposed["OfferDate"] = start_date + pd.Timedelta(days=14 * campaign_index)
        exposed["DataNature"] = "semi_synthetic_rct_anchored"
        rows.append(exposed)

    history = pd.concat(rows, ignore_index=True)
    history["SplitSet"] = assign_client_split(history["IdCliente"])
    validate_campaign_history(history, campaign_count)
    return history


def validate_campaign_history(history: pd.DataFrame, campaign_count: int = 8) -> None:
    required = {
        "CampaignId",
        "OfferDate",
        "IdCliente",
        "RewardType",
        "Responded",
        "SimulationProbability",
        "ValueIfResponse",
        "ActualIncrementalValue",
        "Cost",
        "SplitSet",
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"Historico incompleto: {missing}")
    if history.duplicated(["CampaignId", "IdCliente"]).any():
        raise ValueError("CampaignId + IdCliente debe ser unico")
    if set(history["RewardType"]) != set(TREATMENTS):
        raise ValueError("Faltan tratamientos en el historico")
    if not set(history["Responded"]).issubset({0, 1}):
        raise ValueError("Responded debe ser binario")
    split_counts = history.groupby("IdCliente")["SplitSet"].nunique()
    if (split_counts != 1).any():
        raise ValueError("Un cliente aparece en mas de un split")
    exposures = history.groupby("IdCliente").size()
    if not exposures.eq(campaign_count).all():
        raise ValueError("Cada cliente debe tener una exposicion por campaña")
    treatment_counts = history.groupby(["IdCliente", "RewardType"]).size().unstack(fill_value=0)
    expected_per_type = campaign_count // len(TREATMENTS)
    balanceable = campaign_count % len(TREATMENTS) == 0
    balanced = treatment_counts.eq(expected_per_type).all().all()
    if balanceable and not balanced:
        raise ValueError("La rotacion de tratamientos no esta balanceada por cliente")


def response_metrics(y_true: pd.Series | np.ndarray, probability: np.ndarray) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    prevalence = float(y.mean())
    top_count = max(1, int(np.ceil(len(y) * 0.10)))
    top_rate = float(y[np.argsort(-p)[:top_count]].mean())
    return {
        "rows": int(len(y)),
        "responses": int(y.sum()),
        "response_rate": round(prevalence, 6),
        "roc_auc": round(float(roc_auc_score(y, p)), 6) if len(np.unique(y)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y, p)), 6),
        "brier": round(float(brier_score_loss(y, p)), 6),
        "log_loss": round(float(log_loss(y, p)), 6),
        "lift_top_decile": round(top_rate / prevalence, 6) if prevalence else None,
    }


@dataclass
class ModeloRespuestaNBOCalibrado:
    features: list[str] | None = None
    random_state: int = 42
    calibration_folds: int = 5
    response_model_: object | None = None
    value_model_: object | None = None
    model_name_: str | None = None
    metrics_: dict | None = None
    medians_: pd.Series | None = None

    def __post_init__(self) -> None:
        self.features = self.features or list(FEATURES_RESPUESTA)

    def _matrix(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        numeric = frame[self.features].apply(pd.to_numeric, errors="coerce")
        if fit:
            self.medians_ = numeric.median(numeric_only=True)
        if self.medians_ is None:
            raise RuntimeError("No hay medianas de entrenamiento")
        matrix = numeric.fillna(self.medians_).fillna(0.0)
        for treatment in TREATMENTS:
            matrix[f"Reward_{treatment}"] = frame["RewardType"].eq(treatment).astype(int)
        return matrix

    def _candidate_estimators(self) -> dict[str, Pipeline]:
        return {
            "logistic_regression": Pipeline(
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
            ),
            "hist_gradient_boosting": Pipeline(
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
            ),
        }

    def _fit_calibrated(
        self, estimator: Pipeline, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray
    ) -> CalibratedClassifierCV:
        folds = list(
            GroupKFold(n_splits=self.calibration_folds).split(x, y, groups=groups)
        )
        model = CalibratedClassifierCV(
            estimator=estimator,
            method="sigmoid",
            cv=folds,
            n_jobs=-1,
            ensemble=True,
        )
        model.fit(x, y)
        return model

    def fit_select_evaluate(self, history: pd.DataFrame) -> ModeloRespuestaNBOCalibrado:
        validate_campaign_history(history, int(history["CampaignId"].nunique()))
        partitions = {
            split: history.loc[history["SplitSet"] == split].reset_index(drop=True)
            for split in ("train", "validation", "test")
        }
        train, validation, test = (
            partitions["train"],
            partitions["validation"],
            partitions["test"],
        )
        x_train = self._matrix(train, fit=True)
        x_validation = self._matrix(validation)
        y_train = train["Responded"].astype(int).to_numpy()
        y_validation = validation["Responded"].astype(int).to_numpy()

        candidates = {}
        candidate_metrics = {}
        for name, estimator in self._candidate_estimators().items():
            model = self._fit_calibrated(
                estimator, x_train, y_train, train["IdCliente"].to_numpy()
            )
            probability = model.predict_proba(x_validation)[:, 1]
            candidates[name] = model
            candidate_metrics[name] = response_metrics(y_validation, probability)

        self.model_name_ = max(
            candidate_metrics,
            key=lambda name: (
                candidate_metrics[name]["pr_auc"],
                -candidate_metrics[name]["brier"],
            ),
        )
        self.response_model_ = candidates[self.model_name_]

        reward_responders = train["Responded"].eq(1) & train["RewardType"].ne("control")
        self.value_model_ = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            loss="absolute_error",
            random_state=self.random_state,
        )
        self.value_model_.fit(
            x_train.loc[reward_responders],
            train.loc[reward_responders, "ValueIfResponse"],
        )

        self.metrics_ = {
            "model_version": MODEL_VERSION,
            "simulation_version": SIMULATION_VERSION,
            "feature_set_version": FEATURE_SET_VERSION,
            "data_nature": "semi_synthetic_rct_anchored",
            "selected_model": self.model_name_,
            "split_rows": {key: int(len(value)) for key, value in partitions.items()},
            "split_clients": {
                key: int(value["IdCliente"].nunique()) for key, value in partitions.items()
            },
            "candidate_validation": candidate_metrics,
            "validation": self._evaluate_partition(validation),
            "test": self._evaluate_partition(test),
        }
        return self

    def _evaluate_partition(self, frame: pd.DataFrame) -> dict:
        matrix = self._matrix(frame)
        probability = self.response_model_.predict_proba(matrix)[:, 1]
        by_reward = {}
        for reward, group in frame.groupby("RewardType"):
            positions = frame.index.get_indexer(group.index)
            by_reward[str(reward)] = response_metrics(
                group["Responded"], probability[positions]
            )
            by_reward[str(reward)]["predicted_mean_rate"] = round(
                float(probability[positions].mean()), 6
            )

        responders = frame["Responded"].eq(1) & frame["RewardType"].ne("control")
        predicted_value = self.value_model_.predict(matrix.loc[responders])
        actual_value = frame.loc[responders, "ValueIfResponse"]
        value_metrics = {
            "rows": int(responders.sum()),
            "mae": round(float(mean_absolute_error(actual_value, predicted_value)), 6),
            "r2": round(float(r2_score(actual_value, predicted_value)), 6),
        }
        return {
            **response_metrics(frame["Responded"], probability),
            "by_reward": by_reward,
            "value_if_response": value_metrics,
        }

    def predict_options(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.response_model_ is None or self.value_model_ is None:
            raise RuntimeError("El modelo no esta entrenado")
        control_frame = features.copy()
        control_frame["RewardType"] = "control"
        control_probability = self.response_model_.predict_proba(
            self._matrix(control_frame)
        )[:, 1]

        options = []
        for reward in REWARDS_NBO:
            candidate = features.copy()
            candidate["RewardType"] = reward
            matrix = self._matrix(candidate)
            probability = self.response_model_.predict_proba(matrix)[:, 1]
            value = np.maximum(self.value_model_.predict(matrix), 0.0)
            cost = float(REWARDS.costo[reward])
            uplift_probability = probability - control_probability
            options.append(
                pd.DataFrame(
                    {
                        "IdCliente": candidate["IdCliente"].to_numpy(),
                        "RewardType": reward,
                        "ControlProbability": control_probability,
                        "ResponseProbability": probability,
                        "UpliftProbability": uplift_probability,
                        "ValueIncremental": value,
                        "Cost": cost,
                        "ExpectedValue": probability * value - cost,
                        "IncrementalExpectedValue": uplift_probability * value - cost,
                    }
                )
            )
        result = pd.concat(options, ignore_index=True)
        component_columns = [
            "ControlProbability",
            "ResponseProbability",
            "UpliftProbability",
            "ValueIncremental",
            "Cost",
        ]
        result[component_columns] = result[component_columns].round(6)
        result["ExpectedValue"] = (
            result["ResponseProbability"] * result["ValueIncremental"] - result["Cost"]
        ).round(6)
        result["IncrementalExpectedValue"] = (
            result["UpliftProbability"] * result["ValueIncremental"] - result["Cost"]
        ).round(6)
        return result

    def predict_wide(self, features: pd.DataFrame) -> pd.DataFrame:
        options = self.predict_options(features)
        renamed = options.rename(
            columns={
                "ResponseProbability": "ProbRespuesta",
                "ValueIncremental": "ValorIncremental",
                "ExpectedValue": "ValorEsperado",
            }
        )
        wide = renamed.pivot(
            index="IdCliente",
            columns="RewardType",
            values=["ProbRespuesta", "ValorIncremental", "ValorEsperado"],
        )
        wide.columns = [f"{metric}_{reward}" for metric, reward in wide.columns]
        return wide.reset_index()

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> ModeloRespuestaNBOCalibrado:
        return joblib.load(path)

