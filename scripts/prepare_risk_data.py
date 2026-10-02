"""Ejecuta y publica la etapa 1 del modelo supervisado de riesgo.

No entrena modelos. Materializa la ABT en SQL Server, valida su contrato,
publica Parquet y genera artefactos de calidad/EDA.
"""

from __future__ import annotations

import argparse
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
from sqlalchemy import create_engine

from casino_ia import config
from casino_ia.features.risk_supervised import (
    RISK_FEATURES_V1,
    TARGET_COLUMN,
    validate_risk_abt,
)

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "04_preparar_abt_riesgo_supervisado.sql"
PARQUET_FILE = config.DATA_PROCESSED / "abt_riesgo_supervisado.parquet"
QUALITY_FILE = config.METRICS / "riesgo_etapa1_calidad.json"
SUMMARY_FILE = config.METRICS / "riesgo_etapa1_resumen.csv"
CORRELATION_FILE = config.METRICS / "riesgo_etapa1_correlaciones.csv"
FIGURE_FILE = config.FIGURES / "riesgo_etapa1_eda.png"


def run_sql_preparation() -> None:
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
        raise RuntimeError(f"Fallo sqlcmd ({result.returncode}): {result.stderr}")


def read_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    engine = create_engine(config.DB.sqlalchemy_url())
    with engine.connect() as connection:
        abt = pd.read_sql("SELECT * FROM ml.vw_RiskAnalyticalBaseCurrent", connection)
        quality = pd.read_sql(
            "SELECT * FROM ml.vw_RiskDataQualityCurrent ORDER BY MetricName", connection
        )
        rejections = pd.read_sql(
            """
            SELECT ReasonCode, COUNT_BIG(*) AS RejectedRows,
                   COUNT(DISTINCT UserID) AS Users
            FROM ml.RiskDataRejection
            WHERE RunId=(SELECT MAX(RunId) FROM ml.RiskDataPreparationRun WHERE Status='COMPLETED')
            GROUP BY ReasonCode ORDER BY ReasonCode
            """,
            connection,
        )
    return abt, quality, rejections


def generate_eda(abt: pd.DataFrame) -> dict:
    numeric = abt[RISK_FEATURES_V1].apply(pd.to_numeric, errors="coerce")
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
    summary["missing"] = numeric.isna().sum()
    summary["missing_rate"] = numeric.isna().mean()
    summary.to_csv(SUMMARY_FILE, encoding="utf-8")

    corr_input = numeric.copy()
    corr_input[TARGET_COLUMN] = pd.to_numeric(abt[TARGET_COLUMN])
    correlations = (
        corr_input.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(key=lambda values: values.abs(), ascending=False)
        .rename("correlation_with_target")
        .to_frame()
    )
    correlations.to_csv(CORRELATION_FILE, encoding="utf-8")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    abt[TARGET_COLUMN].value_counts().sort_index().plot.bar(
        ax=axes[0, 0], color=["#4472C4", "#C00000"]
    )
    axes[0, 0].set_title("Etiqueta RG en los siguientes 30 dias")
    axes[0, 0].set_xlabel("TargetRGEvent")
    axes[0, 0].set_ylabel("Ventanas")

    split_target = pd.crosstab(abt["SplitSet"], abt[TARGET_COLUMN]).reindex(
        ["train", "validation", "test"]
    )
    split_target.plot.bar(stacked=True, ax=axes[0, 1], color=["#70AD47", "#C00000"])
    axes[0, 1].set_title("Distribucion por split y etiqueta")
    axes[0, 1].set_xlabel("")

    for target, color in [(0, "#4472C4"), (1, "#C00000")]:
        values = pd.to_numeric(
            abt.loc[abt[TARGET_COLUMN] == target, "TurnoverTotal"], errors="coerce"
        ).dropna()
        axes[1, 0].hist(
            np.log1p(values.clip(lower=0)), bins=40, alpha=0.55, label=str(target), color=color
        )
    axes[1, 0].set_title("log(1 + turnover 90d) por etiqueta")
    axes[1, 0].legend(title="Target")

    box_data = [
        pd.to_numeric(abt.loc[abt[TARGET_COLUMN] == target, "ActiveDays"], errors="coerce").dropna()
        for target in (0, 1)
    ]
    axes[1, 1].boxplot(box_data, tick_labels=["0", "1"], showfliers=False)
    axes[1, 1].set_title("Dias activos en 90d por etiqueta")
    axes[1, 1].set_xlabel("TargetRGEvent")
    fig.tight_layout()
    fig.savefig(FIGURE_FILE, dpi=140)
    plt.close(fig)

    return {
        "top_absolute_correlations": {
            str(k): round(float(v), 6)
            for k, v in correlations.head(10)["correlation_with_target"].items()
            if pd.notna(v)
        },
        "features_with_missing": {
            str(k): round(float(v), 6)
            for k, v in summary.loc[summary["missing_rate"] > 0, "missing_rate"]
            .sort_values(ascending=False)
            .items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-sql", action="store_true", help="Usa la ABT ya materializada en SQL Server"
    )
    args = parser.parse_args()

    if not args.skip_sql:
        run_sql_preparation()

    abt, quality, rejections = read_outputs()
    validation = validate_risk_abt(abt)
    abt.to_parquet(PARQUET_FILE, index=False)
    eda = generate_eda(abt)

    report = {
        "stage": "1_preparacion_datos_riesgo_supervisado",
        "status": "COMPLETED",
        "methodology": {
            "unit": "cliente_fecha_corte",
            "observation_days": 90,
            "prediction_horizon_days": 30,
            "target": "primer evento RG durante los 30 dias posteriores",
            "split": "hash deterministico por cliente; sin clientes compartidos",
            "excluded_from_features": [
                "fechas y tipos de intervencion RG",
                "RGCase global",
                "atributos de contexto CountryName/LanguageName/Gender/AgeAtCutoff",
            ],
        },
        "validation": validation.as_dict(),
        "sql_quality": quality.to_dict("records"),
        "rejections": rejections.to_dict("records"),
        "eda": eda,
        "artifacts": {
            "parquet": str(PARQUET_FILE),
            "summary": str(SUMMARY_FILE),
            "correlations": str(CORRELATION_FILE),
            "figure": str(FIGURE_FILE),
        },
    }
    QUALITY_FILE.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    print("ETAPA 1 COMPLETADA")
    print(json.dumps(validation.as_dict(), ensure_ascii=False, indent=2))
    print("ABT SQL: ml.vw_RiskAnalyticalBaseCurrent")
    print(f"Parquet: {PARQUET_FILE}")
    print(f"Calidad: {QUALITY_FILE}")
    print(f"EDA: {FIGURE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
