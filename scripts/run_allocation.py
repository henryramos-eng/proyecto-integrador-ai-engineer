"""Etapa 4: ejecuta, valida y publica el optimizador greedy de recompensas."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from sqlalchemy import create_engine, text

from casino_ia import config
from casino_ia.data import cargar_features_cliente
from casino_ia.models import ModeloRespuesta, ModeloRiesgo
from casino_ia.optimization.allocate import (
    MAX_FRACCION_ALTAS,
    MAX_FRACCION_SEGMENTO,
    MIN_SESIONES_90D,
    asignar_recompensas,
)

OPTIMIZER_VERSION = "greedy-nbo-v2.0.0"
RISK_SOURCE = "modelo-riesgo-comportamental-casino-v1"
SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "07_optimizador_recompensas.sql"
RISK_MODEL_FILE = config.MODELS_STORE / "modelo_riesgo.joblib"
BASE_RESPONSE_MODEL_FILE = config.MODELS_STORE / "modelo_respuesta.joblib"
PLAN_FILE = config.METRICS / "plan_asignacion.csv"
METRICS_FILE = config.METRICS / "metrics_optimizador_recompensas.json"
REWARDS = ("baja", "media", "alta")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_response_options() -> pd.DataFrame:
    query = """
    SELECT ResponseModelRunId,IdCliente,RewardType,ControlProbability,
           ResponseProbability,UpliftProbability,ValueIncremental,Cost,
           ExpectedValue,IncrementalExpectedValue
    FROM ml.vw_ResponseOptionCurrent
    """
    engine = create_engine(config.DB.sqlalchemy_url())
    with engine.connect() as connection:
        return pd.read_sql(query, connection)


def validate_response_options(options: pd.DataFrame, client_ids: set[int]) -> None:
    required = {
        "ResponseModelRunId",
        "IdCliente",
        "RewardType",
        "ResponseProbability",
        "ValueIncremental",
        "Cost",
        "ExpectedValue",
    }
    missing = sorted(required.difference(options.columns))
    if missing:
        raise ValueError(f"Faltan columnas de opciones NBO: {missing}")
    if options.empty:
        raise ValueError("ml.vw_ResponseOptionCurrent no contiene opciones")
    if options.duplicated(["IdCliente", "RewardType"]).any():
        raise ValueError("Hay opciones duplicadas por cliente y recompensa")
    run_ids = options["ResponseModelRunId"].drop_duplicates()
    if len(run_ids) != 1:
        raise ValueError(f"Se esperaba una sola corrida de respuesta: {run_ids.tolist()}")
    if set(options["IdCliente"].astype(int)) != client_ids:
        raise ValueError("Las opciones NBO y la cartera de clientes no coinciden")
    reward_sets = options.groupby("IdCliente")["RewardType"].agg(set)
    if not reward_sets.map(lambda values: values == set(REWARDS)).all():
        raise ValueError("Cada cliente debe tener opciones baja, media y alta")
    recomputed = options["ResponseProbability"] * options["ValueIncremental"] - options["Cost"]
    if (recomputed - options["ExpectedValue"]).abs().max() > 0.00001:
        raise ValueError("ExpectedValue no coincide con probabilidad, valor y costo")


def reward_costs(options: pd.DataFrame) -> dict[str, float]:
    distinct = options.groupby("RewardType")["Cost"].nunique()
    if not distinct.eq(1).all():
        raise ValueError("El costo debe ser único por tipo de recompensa")
    costs = options.groupby("RewardType")["Cost"].first().astype(float).to_dict()
    if set(costs) != set(REWARDS):
        raise ValueError(f"Tipos de recompensa incompletos: {sorted(costs)}")
    return costs


def build_scoring(features: pd.DataFrame, options: pd.DataFrame) -> pd.DataFrame:
    if features.empty or features["IdCliente"].duplicated().any():
        raise ValueError("La tabla de features debe contener un cliente único por fila")

    risk_model = ModeloRiesgo.load(RISK_MODEL_FILE)
    response_v1 = ModeloRespuesta.load(BASE_RESPONSE_MODEL_FILE)
    risk = risk_model.predict(features)[["IdCliente", "NivelRiesgo", "RiesgoScore"]]
    base_probability = response_v1.predict_proba(features)[["IdCliente", "ProbRespuesta"]]

    wide = options.pivot(
        index="IdCliente",
        columns="RewardType",
        values=["ResponseProbability", "ValueIncremental"],
    )
    names = {
        "ResponseProbability": "ProbRespuesta",
        "ValueIncremental": "ValorIncremental",
    }
    wide.columns = [f"{names[metric]}_{reward}" for metric, reward in wide.columns]
    wide = wide.reset_index()

    columns = [
        "IdCliente",
        "Segmento",
        "NroSesiones",
        "ValorTeoricoCasa",
        "RatioTendenciaCoinIn",
    ]
    scoring = (
        features[columns]
        .merge(risk, on="IdCliente", validate="one_to_one")
        .merge(base_probability, on="IdCliente", validate="one_to_one")
        .merge(wide, on="IdCliente", validate="one_to_one")
    )
    # Los 15 días disponibles están completamente contenidos en la ventana de 90 días.
    scoring["SesionesUltimos90d"] = scoring["NroSesiones"].astype(int)
    return scoring


def v1_baseline_scoring(scoring: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("ProbRespuesta_", "ValorIncremental_")
    return scoring.drop(columns=[column for column in scoring if column.startswith(prefixes)])


def allocation_summary(decisions: pd.DataFrame, budget: float) -> dict:
    assigned = decisions.loc[decisions["Asignada"]].copy()
    segment_spend = assigned.groupby("Segmento", dropna=False)["Costo"].sum()
    high_spend = assigned.loc[assigned["Recompensa"].eq("alta"), "Costo"].sum()
    total_spend = float(assigned["Costo"].sum())
    return {
        "clients": int(len(decisions)),
        "positive_candidates": int(decisions["RecompensaSugerida"].notna().sum()),
        "assigned_clients": int(len(assigned)),
        "total_spend": round(total_spend, 6),
        "budget_utilization": round(total_spend / budget, 6) if budget else 0.0,
        "total_expected_value": round(float(assigned["ValorEsperado"].sum()), 6),
        "high_reward_spend": round(float(high_spend), 6),
        "high_reward_budget_share": round(float(high_spend) / budget, 6) if budget else 0.0,
        "max_segment_budget_share": (
            round(float(segment_spend.max()) / budget, 6)
            if budget and not segment_spend.empty
            else 0.0
        ),
        "assigned_by_reward": {
            str(key): int(value)
            for key, value in assigned["Recompensa"].value_counts().sort_index().items()
        },
        "assigned_by_risk": {
            str(key): int(value)
            for key, value in assigned["NivelRiesgo"].value_counts().sort_index().items()
        },
        "decisions_by_reason": {
            str(key): int(value)
            for key, value in decisions["MotivoDecision"].value_counts().items()
        },
    }


def validate_decisions(
    decisions: pd.DataFrame,
    input_clients: int,
    budget: float,
    max_high_fraction: float,
    max_segment_fraction: float,
) -> None:
    if len(decisions) != input_clients or not decisions["IdCliente"].is_unique:
        raise ValueError("Debe existir exactamente una decisión por cliente")
    assigned = decisions.loc[decisions["Asignada"]]
    if (assigned["ValorEsperado"] <= 0).any():
        raise ValueError("Se asignó una recompensa con valor esperado no positivo")
    if assigned["Costo"].sum() > budget + 0.000001:
        raise ValueError("La asignación supera el presupuesto")
    if decisions.loc[decisions["NivelRiesgo"].eq("Alto"), "Asignada"].any():
        raise ValueError("Se asignó una recompensa a un cliente de riesgo alto")
    invalid_medium = assigned["NivelRiesgo"].eq("Medio") & ~assigned["Recompensa"].eq("baja")
    if invalid_medium.any():
        raise ValueError("Un cliente de riesgo medio recibió una recompensa distinta de baja")
    if decisions.loc[decisions["SesionesUltimos90d"] < MIN_SESIONES_90D, "Asignada"].any():
        raise ValueError("Se asignó una recompensa sin cumplir la elegibilidad mínima")
    high_spend = assigned.loc[assigned["Recompensa"].eq("alta"), "Costo"].sum()
    if high_spend > budget * max_high_fraction + 0.000001:
        raise ValueError("Se superó el tope de recompensas altas")
    segment_spend = assigned.groupby("Segmento", dropna=False)["Costo"].sum()
    if not segment_spend.empty and segment_spend.max() > budget * max_segment_fraction + 0.000001:
        raise ValueError("Se superó el tope por segmento")
    if decisions["MotivoDecision"].fillna("").eq("").any():
        raise ValueError("Hay decisiones sin trazabilidad")


def combine_with_baseline(decisions: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = ["IdCliente", "Recompensa", "Costo", "ValorEsperado", "Asignada"]
    renamed = baseline[columns].rename(
        columns={
            "Recompensa": "BaselineRecompensa",
            "Costo": "BaselineCosto",
            "ValorEsperado": "BaselineValorEsperado",
            "Asignada": "BaselineAsignada",
        }
    )
    return decisions.merge(renamed, on="IdCliente", how="left", validate="one_to_one")


def persist_results(
    decisions: pd.DataFrame,
    response_run_id: int,
    budget: float,
    metrics: dict,
) -> int:
    risk_hash = file_sha256(RISK_MODEL_FILE)
    baseline_hash = file_sha256(BASE_RESPONSE_MODEL_FILE)
    fingerprint_data = {
        "optimizer_version": OPTIMIZER_VERSION,
        "response_run_id": response_run_id,
        "risk_hash": risk_hash,
        "baseline_hash": baseline_hash,
        "budget": budget,
        "max_high_fraction": MAX_FRACCION_ALTAS,
        "max_segment_fraction": MAX_FRACCION_SEGMENTO,
        "min_sessions": MIN_SESIONES_90D,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")
    ).hexdigest()
    optimized = metrics["optimized_v2"]
    baseline = metrics["baseline_v1"]
    metrics_json = json.dumps(metrics, ensure_ascii=False, default=str)
    engine = create_engine(config.DB.sqlalchemy_url(), fast_executemany=True)

    with engine.begin() as connection:
        existing = connection.execute(
            text(
                """
                SELECT AllocationRunId FROM ml.RewardAllocationRun
                WHERE OptimizerVersion=:version AND SourceFingerprint=:fingerprint
                """
            ),
            {"version": OPTIMIZER_VERSION, "fingerprint": fingerprint},
        ).scalar()
        values = {
            "response_run_id": response_run_id,
            "version": OPTIMIZER_VERSION,
            "fingerprint": fingerprint,
            "risk_source": RISK_SOURCE,
            "risk_hash": risk_hash,
            "baseline_hash": baseline_hash,
            "budget": budget,
            "max_high": MAX_FRACCION_ALTAS,
            "max_segment": MAX_FRACCION_SEGMENTO,
            "min_sessions": MIN_SESIONES_90D,
            "input_clients": optimized["clients"],
            "candidates": optimized["positive_candidates"],
            "assigned": optimized["assigned_clients"],
            "spend": optimized["total_spend"],
            "expected": optimized["total_expected_value"],
            "baseline_assigned": baseline["assigned_clients"],
            "baseline_spend": baseline["total_spend"],
            "baseline_expected": baseline["total_expected_value"],
            "metrics": metrics_json,
        }
        if existing is None:
            run_id = int(
                connection.execute(
                    text(
                        """
                        INSERT INTO ml.RewardAllocationRun(
                            ResponseModelRunId,OptimizerVersion,SourceFingerprint,
                            RiskSource,RiskArtifactHash,BaselineArtifactHash,Budget,
                            MaxHighFraction,MaxSegmentFraction,MinSessions90Days,
                            InputClients,PositiveCandidates,AssignedClients,TotalSpend,
                            TotalExpectedValue,BaselineAssignedClients,BaselineSpend,
                            BaselineExpectedValue,MetricsJson,Status
                        ) OUTPUT INSERTED.AllocationRunId
                        VALUES(
                            :response_run_id,:version,:fingerprint,:risk_source,
                            :risk_hash,:baseline_hash,:budget,:max_high,:max_segment,
                            :min_sessions,:input_clients,:candidates,:assigned,:spend,
                            :expected,:baseline_assigned,:baseline_spend,
                            :baseline_expected,:metrics,'STARTED'
                        )
                        """
                    ),
                    values,
                ).scalar_one()
            )
        else:
            run_id = int(existing)
            connection.execute(
                text("DELETE FROM ml.RewardAllocationDecision WHERE AllocationRunId=:run_id"),
                {"run_id": run_id},
            )
            values["run_id"] = run_id
            connection.execute(
                text(
                    """
                    UPDATE ml.RewardAllocationRun SET
                        ResponseModelRunId=:response_run_id,RiskSource=:risk_source,
                        RiskArtifactHash=:risk_hash,BaselineArtifactHash=:baseline_hash,
                        Budget=:budget,MaxHighFraction=:max_high,
                        MaxSegmentFraction=:max_segment,MinSessions90Days=:min_sessions,
                        InputClients=:input_clients,PositiveCandidates=:candidates,
                        AssignedClients=:assigned,TotalSpend=:spend,
                        TotalExpectedValue=:expected,
                        BaselineAssignedClients=:baseline_assigned,
                        BaselineSpend=:baseline_spend,
                        BaselineExpectedValue=:baseline_expected,MetricsJson=:metrics,
                        Status='STARTED',StartedAt=SYSUTCDATETIME(),CompletedAt=NULL
                    WHERE AllocationRunId=:run_id
                    """
                ),
                values,
            )

        payload = decisions.rename(
            columns={
                "Segmento": "Segment",
                "NivelRiesgo": "RiskLevel",
                "RiesgoScore": "RiskScore",
                "SesionesUltimos90d": "SessionsLast90Days",
                "RecompensaSugerida": "SuggestedReward",
                "CostoSugerido": "SuggestedCost",
                "Recompensa": "AssignedReward",
                "Costo": "Cost",
                "ProbRespuesta": "ResponseProbability",
                "ValorIncremental": "IncrementalValue",
                "ValorEsperado": "ExpectedValue",
                "Eficiencia": "Efficiency",
                "GastoAcumulado": "CumulativeSpend",
                "Asignada": "Assigned",
                "MotivoDecision": "DecisionReason",
                "BaselineRecompensa": "BaselineAssignedReward",
                "BaselineCosto": "BaselineCost",
                "BaselineValorEsperado": "BaselineExpectedValue",
                "BaselineAsignada": "BaselineAssigned",
            }
        )
        destination_columns = [
            "IdCliente",
            "Segment",
            "RiskLevel",
            "RiskScore",
            "SessionsLast90Days",
            "SuggestedReward",
            "SuggestedCost",
            "AssignedReward",
            "Cost",
            "ResponseProbability",
            "IncrementalValue",
            "ExpectedValue",
            "Efficiency",
            "CumulativeSpend",
            "Assigned",
            "DecisionReason",
            "BaselineAssignedReward",
            "BaselineCost",
            "BaselineExpectedValue",
            "BaselineAssigned",
        ]
        payload = payload[destination_columns].copy()
        payload.insert(0, "AllocationRunId", run_id)
        payload.to_sql(
            "RewardAllocationDecision",
            schema="ml",
            con=connection,
            if_exists="append",
            index=False,
            chunksize=1000,
        )
        inserted = int(
            connection.execute(
                text(
                    "SELECT COUNT_BIG(*) FROM ml.RewardAllocationDecision "
                    "WHERE AllocationRunId=:run_id"
                ),
                {"run_id": run_id},
            ).scalar_one()
        )
        if inserted != len(decisions):
            raise RuntimeError(f"Decisiones esperadas={len(decisions)}, insertadas={inserted}")
        connection.execute(
            text(
                "UPDATE ml.RewardAllocationRun SET Status='COMPLETED',"
                "CompletedAt=SYSUTCDATETIME() WHERE AllocationRunId=:run_id"
            ),
            {"run_id": run_id},
        )
    return run_id


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presupuesto", type=float, default=config.REWARDS.presupuesto)
    args = parser.parse_args()
    if args.presupuesto <= 0:
        parser.error("--presupuesto debe ser mayor que cero")

    run_sql_contract()
    features = cargar_features_cliente()
    options = read_response_options()
    validate_response_options(options, set(features["IdCliente"].astype(int)))
    costs = reward_costs(options)
    response_run_id = int(options["ResponseModelRunId"].iloc[0])
    scoring = build_scoring(features, options)

    optimized = asignar_recompensas(scoring, presupuesto=args.presupuesto, costos=costs)
    baseline = asignar_recompensas(
        v1_baseline_scoring(scoring),
        presupuesto=args.presupuesto,
        costos=costs,
    )
    validate_decisions(
        optimized,
        len(scoring),
        args.presupuesto,
        MAX_FRACCION_ALTAS,
        MAX_FRACCION_SEGMENTO,
    )
    validate_decisions(
        baseline,
        len(scoring),
        args.presupuesto,
        MAX_FRACCION_ALTAS,
        MAX_FRACCION_SEGMENTO,
    )

    optimized_summary = allocation_summary(optimized, args.presupuesto)
    baseline_summary = allocation_summary(baseline, args.presupuesto)
    expected_delta = (
        optimized_summary["total_expected_value"] - baseline_summary["total_expected_value"]
    )
    metrics = {
        "optimizer_version": OPTIMIZER_VERSION,
        "response_model_run_id": response_run_id,
        "risk_source": RISK_SOURCE,
        "session_window_note": (
            "NroSesiones usa los 15 dias disponibles, contenidos en la ventana de 90 dias"
        ),
        "guardrails": {
            "minimum_sessions_90_days": MIN_SESIONES_90D,
            "maximum_high_reward_budget_fraction": MAX_FRACCION_ALTAS,
            "maximum_segment_budget_fraction": MAX_FRACCION_SEGMENTO,
            "high_risk": "sin recompensa",
            "medium_risk": "solo recompensa baja",
            "positive_expected_value_only": True,
        },
        "optimized_v2": optimized_summary,
        "baseline_v1": baseline_summary,
        "comparison": {
            "expected_value_delta": round(expected_delta, 6),
            "expected_value_improvement": (
                round(expected_delta / baseline_summary["total_expected_value"], 6)
                if baseline_summary["total_expected_value"]
                else None
            ),
        },
    }
    decisions = combine_with_baseline(optimized, baseline)
    decisions = decisions.merge(
        scoring[["IdCliente", "RiesgoScore"]], on="IdCliente", validate="one_to_one"
    )
    run_id = persist_results(decisions, response_run_id, args.presupuesto, metrics)

    decisions.to_csv(PLAN_FILE, index=False, encoding="utf-8")
    METRICS_FILE.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("ETAPA 4 COMPLETADA")
    print(f"AllocationRunId: {run_id}")
    print(f"ResponseModelRunId: {response_run_id}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
