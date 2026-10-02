"""Etapa 2: entrena, evalua y publica el modelo supervisado de riesgo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sqlalchemy import create_engine, text

from casino_ia import config
from casino_ia.models.risk_supervised import (
    FEATURE_SET_VERSION,
    MODEL_VERSION,
    ModeloRiesgoSupervisado,
)

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "05_modelo_riesgo_supervisado.sql"
MODEL_FILE = config.MODELS_STORE / "modelo_riesgo_supervisado.joblib"
METRICS_FILE = config.METRICS / "metrics_riesgo_supervisado.json"
LEVELS_FILE = config.METRICS / "riesgo_supervisado_por_nivel.csv"
FIGURE_FILE = config.FIGURES / "riesgo_supervisado_evaluacion.png"


def run_sql_contract() -> None:
    command = [
        "sqlcmd",
        "-S",
        config.DB.server,
        "-d",
        config.DB.database,
        "-E",
        "-C",
        "-b",
        "-i",
        str(SQL_FILE),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Fallo el contrato SQL ({result.returncode}): {result.stderr}")


def read_abt() -> pd.DataFrame:
    engine = create_engine(config.DB.sqlalchemy_url())
    with engine.connect() as connection:
        return pd.read_sql("SELECT * FROM ml.vw_RiskAnalyticalBaseCurrent", connection)


def plot_evaluation(
    y_test: np.ndarray,
    probabilities: np.ndarray,
    predictions: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    axes[0, 0].plot(fpr, tpr, color="#4472C4")
    axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0, 0].set_title("ROC — test")
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    axes[0, 1].plot(recall, precision, color="#C00000")
    axes[0, 1].axhline(y_test.mean(), linestyle="--", color="gray")
    axes[0, 1].set_title("Precision–Recall — test")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precision")

    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    axes[1, 0].plot(predicted, observed, marker="o", color="#70AD47")
    axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[1, 0].set_title("Calibracion — test")
    axes[1, 0].set_xlabel("Probabilidad estimada")
    axes[1, 0].set_ylabel("Frecuencia observada")

    order = ["Bajo", "Medio", "Alto"]
    tier = (
        predictions.assign(ActualTarget=y_test)
        .groupby("RiskLevel", observed=False)["ActualTarget"]
        .agg(rows="size", event_rate="mean")
        .reindex(order)
    )
    axes[1, 1].bar(order, tier["rows"].fillna(0), color=["#70AD47", "#FFC000", "#C00000"])
    axes[1, 1].set_title("Volumen y tasa observada por nivel — test")
    axes[1, 1].set_ylabel("Ventanas")
    secondary = axes[1, 1].twinx()
    secondary.plot(order, tier["event_rate"], color="black", marker="o")
    secondary.set_ylabel("Tasa de evento")
    secondary.set_ylim(0, max(0.2, float(tier["event_rate"].max()) * 1.25))

    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=140)
    plt.close(fig)


def persist_results(
    abt: pd.DataFrame,
    model: ModeloRiesgoSupervisado,
    predictions: pd.DataFrame,
) -> int:
    engine = create_engine(config.DB.sqlalchemy_url(), fast_executemany=True)
    data_run_ids = abt["RunId"].drop_duplicates().tolist()
    if len(data_run_ids) != 1:
        raise ValueError(f"Se esperaba un unico DataRunId: {data_run_ids}")
    data_run_id = int(data_run_ids[0])
    metrics_json = json.dumps(model.metrics_, ensure_ascii=False, default=str)
    split_rows = abt["SplitSet"].value_counts().to_dict()

    with engine.begin() as connection:
        existing = connection.execute(
            text(
                """
                SELECT ModelRunId FROM ml.RiskModelRun
                WHERE DataRunId=:data_run_id AND ModelVersion=:model_version
                """
            ),
            {"data_run_id": data_run_id, "model_version": MODEL_VERSION},
        ).scalar()
        if existing is None:
            model_run_id = int(
                connection.execute(
                    text(
                        """
                        INSERT INTO ml.RiskModelRun (
                            DataRunId,ModelVersion,FeatureSetVersion,SelectedModel,
                            MediumThreshold,HighThreshold,TrainRows,ValidationRows,
                            TestRows,MetricsJson,ArtifactPath,Status
                        )
                        OUTPUT INSERTED.ModelRunId
                        VALUES (
                            :data_run_id,:model_version,:feature_version,:selected_model,
                            :medium,:high,:train_rows,:validation_rows,:test_rows,
                            :metrics,:artifact,'STARTED'
                        )
                        """
                    ),
                    {
                        "data_run_id": data_run_id,
                        "model_version": MODEL_VERSION,
                        "feature_version": FEATURE_SET_VERSION,
                        "selected_model": model.model_name_,
                        "medium": model.thresholds_["medium"],
                        "high": model.thresholds_["high"],
                        "train_rows": int(split_rows["train"]),
                        "validation_rows": int(split_rows["validation"]),
                        "test_rows": int(split_rows["test"]),
                        "metrics": metrics_json,
                        "artifact": str(MODEL_FILE),
                    },
                ).scalar_one()
            )
        else:
            model_run_id = int(existing)
            connection.execute(
                text("DELETE FROM ml.RiskModelPrediction WHERE ModelRunId=:model_run_id"),
                {"model_run_id": model_run_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE ml.RiskModelRun
                    SET FeatureSetVersion=:feature_version,SelectedModel=:selected_model,
                        MediumThreshold=:medium,HighThreshold=:high,
                        TrainRows=:train_rows,ValidationRows=:validation_rows,
                        TestRows=:test_rows,MetricsJson=:metrics,ArtifactPath=:artifact,
                        Status='STARTED',StartedAt=SYSUTCDATETIME(),CompletedAt=NULL
                    WHERE ModelRunId=:model_run_id
                    """
                ),
                {
                    "model_run_id": model_run_id,
                    "feature_version": FEATURE_SET_VERSION,
                    "selected_model": model.model_name_,
                    "medium": model.thresholds_["medium"],
                    "high": model.thresholds_["high"],
                    "train_rows": int(split_rows["train"]),
                    "validation_rows": int(split_rows["validation"]),
                    "test_rows": int(split_rows["test"]),
                    "metrics": metrics_json,
                    "artifact": str(MODEL_FILE),
                },
            )

        payload = predictions.copy()
        payload.insert(0, "ModelRunId", model_run_id)
        payload["RiskProbability"] = payload["RiskProbability"].round(10)
        payload.to_sql(
            "RiskModelPrediction",
            schema="ml",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=2000,
        )
        inserted = int(
            connection.execute(
                text(
                    "SELECT COUNT_BIG(*) FROM ml.RiskModelPrediction WHERE ModelRunId=:model_run_id"
                ),
                {"model_run_id": model_run_id},
            ).scalar_one()
        )
        if inserted != len(predictions):
            raise RuntimeError(f"Predicciones esperadas={len(predictions)}, insertadas={inserted}")
        connection.execute(
            text(
                """
                UPDATE ml.RiskModelRun
                SET Status='COMPLETED',CompletedAt=SYSUTCDATETIME()
                WHERE ModelRunId=:model_run_id
                """
            ),
            {"model_run_id": model_run_id},
        )
    return model_run_id


def main() -> int:
    run_sql_contract()
    abt = read_abt()
    model = ModeloRiesgoSupervisado().fit_select_evaluate(abt)
    model.save(MODEL_FILE)

    predictions = model.predict(abt)
    model_run_id = persist_results(abt, model, predictions)

    METRICS_FILE.write_text(
        json.dumps(model.metrics_, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    levels = (
        predictions.groupby(["SplitSet", "RiskLevel"], observed=False)
        .agg(
            rows=("FrameId", "size"),
            users=("UserID", "nunique"),
            positives=("ActualTarget", "sum"),
            observed_event_rate=("ActualTarget", "mean"),
            mean_probability=("RiskProbability", "mean"),
        )
        .reset_index()
    )
    levels.to_csv(LEVELS_FILE, index=False, encoding="utf-8")

    test_mask = abt["SplitSet"].eq("test").to_numpy()
    test_predictions = predictions.loc[test_mask].reset_index(drop=True)
    plot_evaluation(
        test_predictions["ActualTarget"].to_numpy(),
        test_predictions["RiskProbability"].to_numpy(),
        test_predictions,
    )

    print("ETAPA 2 COMPLETADA")
    print(f"ModelRunId: {model_run_id}")
    print(f"Modelo seleccionado: {model.model_name_}")
    print(f"Umbrales: {model.thresholds_}")
    print(json.dumps(model.metrics_["test"], ensure_ascii=False, indent=2))
    print(f"Artefacto: {MODEL_FILE}")
    print(f"Metricas: {METRICS_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
