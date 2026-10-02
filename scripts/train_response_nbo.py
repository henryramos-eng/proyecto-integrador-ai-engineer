"""Etapa 3: entrena y publica P(respuesta | cliente, recompensa)."""

from __future__ import annotations

import hashlib
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
from casino_ia.data import cargar_features_cliente
from casino_ia.models.response import ModeloRespuesta
from casino_ia.models.response_nbo_calibrated import (
    MODEL_VERSION,
    SIMULATION_VERSION,
    ModeloRespuestaNBOCalibrado,
    build_response_anchors,
    simulate_rct_anchored_campaigns,
)

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "06_modelo_respuesta_nbo.sql"
BASE_MODEL_FILE = config.MODELS_STORE / "modelo_respuesta.joblib"
MODEL_FILE = config.MODELS_STORE / "modelo_respuesta_nbo_calibrado.joblib"
HISTORY_FILE = config.DATA_PROCESSED / "historico_campanas_rct_calibrado.parquet"
OPTIONS_FILE = config.DATA_PROCESSED / "opciones_respuesta_nbo.parquet"
METRICS_FILE = config.METRICS / "metrics_respuesta_nbo_calibrado.json"
BY_REWARD_FILE = config.METRICS / "respuesta_nbo_por_recompensa.csv"
FIGURE_FILE = config.FIGURES / "respuesta_nbo_evaluacion.png"


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
        raise RuntimeError(f"Fallo contrato SQL ({result.returncode}): {result.stderr}")


def read_evidence() -> pd.DataFrame:
    engine = create_engine(config.DB.sqlalchemy_url())
    query = """
    SELECT 'HillstromEmail' AS SourceDataset,Segment AS TreatmentLabel,
           'visit' AS OutcomeName,COUNT_BIG(*) AS SourceRows,
           SUM(CONVERT(bigint,Visit)) AS Responses,
           AVG(CONVERT(float,Visit)) AS ObservedRate
    FROM ext.HillstromEmail GROUP BY Segment
    UNION ALL
    SELECT 'CriteoUpliftV21',CONVERT(varchar(10),Treatment),'visit',COUNT_BIG(*),
           SUM(CONVERT(bigint,Visit)),AVG(CONVERT(float,Visit))
    FROM ext.CriteoUpliftV21 GROUP BY Treatment
    """
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def evidence_with_roles(evidence: pd.DataFrame) -> pd.DataFrame:
    roles = {
        ("HillstromEmail", "No E-Mail"): "Tasa control base",
        ("HillstromEmail", "Womens E-Mail"): "Tasa objetivo semi-sintetica para recompensa media",
        ("HillstromEmail", "Mens E-Mail"): "Tasa objetivo semi-sintetica para recompensa alta",
        ("CriteoUpliftV21", "0"): "Denominador del uplift relativo para recompensa baja",
        ("CriteoUpliftV21", "1"): "Numerador del uplift relativo para recompensa baja",
    }
    result = evidence.copy()
    result["TreatmentLabel"] = result["TreatmentLabel"].astype(str)
    result["RoleInSimulation"] = [
        roles[(row.SourceDataset, row.TreatmentLabel)] for row in result.itertuples()
    ]
    return result


def plot_evaluation(history: pd.DataFrame, model: ModeloRespuestaNBOCalibrado) -> None:
    test = history.loc[history["SplitSet"] == "test"].reset_index(drop=True)
    matrix = model._matrix(test)
    probability = model.response_model_.predict_proba(matrix)[:, 1]
    y = test["Responded"].astype(int).to_numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fpr, tpr, _ = roc_curve(y, probability)
    axes[0, 0].plot(fpr, tpr, color="#4472C4")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0, 0].set_title("ROC — test por clientes")

    precision, recall, _ = precision_recall_curve(y, probability)
    axes[0, 1].plot(recall, precision, color="#C00000")
    axes[0, 1].axhline(y.mean(), linestyle="--", color="gray")
    axes[0, 1].set_title("Precision–Recall — test")

    observed, predicted = calibration_curve(y, probability, n_bins=10, strategy="quantile")
    axes[1, 0].plot(predicted, observed, marker="o", color="#70AD47")
    axes[1, 0].plot([0, 1], [0, 1], "--", color="gray")
    axes[1, 0].set_title("Calibracion — test")
    axes[1, 0].set_xlabel("Probabilidad estimada")
    axes[1, 0].set_ylabel("Tasa observada")

    comparison = (
        test.assign(PredictedProbability=probability)
        .groupby("RewardType", observed=False)
        .agg(observed=("Responded", "mean"), predicted=("PredictedProbability", "mean"))
        .reindex(["control", "baja", "media", "alta"])
    )
    x = np.arange(len(comparison))
    axes[1, 1].bar(x - 0.18, comparison["observed"], 0.36, label="Observada")
    axes[1, 1].bar(x + 0.18, comparison["predicted"], 0.36, label="Predicha")
    axes[1, 1].set_xticks(x, comparison.index)
    axes[1, 1].set_title("Tasa por tipo — test")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=140)
    plt.close(fig)


def persist_results(
    evidence: pd.DataFrame,
    history: pd.DataFrame,
    options: pd.DataFrame,
    model: ModeloRespuestaNBOCalibrado,
) -> int:
    fingerprint_payload = {
        "model_version": MODEL_VERSION,
        "simulation_version": SIMULATION_VERSION,
        "evidence": evidence[
            ["SourceDataset", "TreatmentLabel", "SourceRows", "Responses", "ObservedRate"]
        ].to_dict("records"),
        "clients": int(options["IdCliente"].nunique()),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    metrics_json = json.dumps(model.metrics_, ensure_ascii=False, default=str)
    engine = create_engine(config.DB.sqlalchemy_url(), fast_executemany=True)

    with engine.begin() as connection:
        existing = connection.execute(
            text(
                """
                SELECT ResponseModelRunId FROM ml.ResponseModelRun
                WHERE ModelVersion=:version AND SourceFingerprint=:fingerprint
                """
            ),
            {"version": MODEL_VERSION, "fingerprint": fingerprint},
        ).scalar()
        if existing is None:
            run_id = int(
                connection.execute(
                    text(
                        """
                        INSERT INTO ml.ResponseModelRun(
                            ModelVersion,SimulationVersion,SourceFingerprint,
                            SelectedModel,MetricsJson,ArtifactPath,Status
                        ) OUTPUT INSERTED.ResponseModelRunId
                        VALUES(:version,:simulation,:fingerprint,:model,:metrics,:artifact,'STARTED')
                        """
                    ),
                    {
                        "version": MODEL_VERSION,
                        "simulation": SIMULATION_VERSION,
                        "fingerprint": fingerprint,
                        "model": model.model_name_,
                        "metrics": metrics_json,
                        "artifact": str(MODEL_FILE),
                    },
                ).scalar_one()
            )
        else:
            run_id = int(existing)
            connection.execute(
                text("DELETE FROM ml.ResponseOption WHERE ResponseModelRunId=:run_id"),
                {"run_id": run_id},
            )
            connection.execute(
                text("DELETE FROM ml.ResponseCampaignSynthetic WHERE ResponseModelRunId=:run_id"),
                {"run_id": run_id},
            )
            connection.execute(
                text("DELETE FROM ml.ResponseEvidenceAnchor WHERE ResponseModelRunId=:run_id"),
                {"run_id": run_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE ml.ResponseModelRun
                    SET SelectedModel=:model,MetricsJson=:metrics,ArtifactPath=:artifact,
                        Status='STARTED',StartedAt=SYSUTCDATETIME(),CompletedAt=NULL
                    WHERE ResponseModelRunId=:run_id
                    """
                ),
                {
                    "run_id": run_id,
                    "model": model.model_name_,
                    "metrics": metrics_json,
                    "artifact": str(MODEL_FILE),
                },
            )

        anchor_payload = evidence.copy()
        anchor_payload.insert(0, "ResponseModelRunId", run_id)
        anchor_payload.to_sql(
            "ResponseEvidenceAnchor",
            schema="ml",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=1000,
        )

        history_columns = [
            "CampaignId",
            "OfferDate",
            "IdCliente",
            "RewardType",
            "SplitSet",
            "Responded",
            "SimulationProbability",
            "ValueIfResponse",
            "ActualIncrementalValue",
            "Cost",
            "DataNature",
        ]
        history_payload = history[history_columns].copy()
        history_payload.insert(0, "ResponseModelRunId", run_id)
        history_payload.to_sql(
            "ResponseCampaignSynthetic",
            schema="ml",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=2000,
        )

        option_payload = options.copy()
        option_payload.insert(0, "ResponseModelRunId", run_id)
        option_payload.to_sql(
            "ResponseOption",
            schema="ml",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=2000,
        )

        campaign_rows = int(
            connection.execute(
                text(
                    "SELECT COUNT_BIG(*) FROM ml.ResponseCampaignSynthetic "
                    "WHERE ResponseModelRunId=:run_id"
                ),
                {"run_id": run_id},
            ).scalar_one()
        )
        option_rows = int(
            connection.execute(
                text("SELECT COUNT_BIG(*) FROM ml.ResponseOption WHERE ResponseModelRunId=:run_id"),
                {"run_id": run_id},
            ).scalar_one()
        )
        if campaign_rows != len(history) or option_rows != len(options):
            raise RuntimeError(
                f"Conteos invalidos: campañas={campaign_rows}/{len(history)}, "
                f"opciones={option_rows}/{len(options)}"
            )
        connection.execute(
            text(
                "UPDATE ml.ResponseModelRun SET Status='COMPLETED',CompletedAt=SYSUTCDATETIME() "
                "WHERE ResponseModelRunId=:run_id"
            ),
            {"run_id": run_id},
        )
    return run_id


def main() -> int:
    run_sql_contract()
    evidence = evidence_with_roles(read_evidence())
    anchors = build_response_anchors(evidence)
    features = cargar_features_cliente()
    base_model = ModeloRespuesta.load(BASE_MODEL_FILE)
    base_probability = base_model.predict_proba(features)
    history = simulate_rct_anchored_campaigns(features, base_probability, anchors)

    model = ModeloRespuestaNBOCalibrado().fit_select_evaluate(history)
    model.metrics_["anchors"] = {
        **anchors.target_rates(),
        "criteo_relative_lift": round(anchors.criteo_relative_lift, 6),
        "mapping_note": (
            "baja/media/alta es una correspondencia semi-sintetica; "
            "no son tratamientos historicos reales"
        ),
    }
    model.save(MODEL_FILE)
    options = model.predict_options(features)
    run_id = persist_results(evidence, history, options, model)

    history.to_parquet(HISTORY_FILE, index=False)
    options.to_parquet(OPTIONS_FILE, index=False)
    METRICS_FILE.write_text(
        json.dumps(model.metrics_, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    summary = (
        options.groupby("RewardType", observed=False)
        .agg(
            clients=("IdCliente", "nunique"),
            mean_probability=("ResponseProbability", "mean"),
            mean_uplift=("UpliftProbability", "mean"),
            mean_value=("ValueIncremental", "mean"),
            positive_expected_value=("ExpectedValue", lambda values: int((values > 0).sum())),
            mean_expected_value=("ExpectedValue", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(BY_REWARD_FILE, index=False, encoding="utf-8")
    plot_evaluation(history, model)

    print("ETAPA 3 COMPLETADA")
    print(f"ResponseModelRunId: {run_id}")
    print(f"Modelo seleccionado: {model.model_name_}")
    print(json.dumps(model.metrics_["test"], ensure_ascii=False, indent=2))
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
