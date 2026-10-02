"""Contrato y controles de la ABT supervisada de riesgo.

La transformación pesada vive en SQL para conservar trazabilidad cerca del dato.
Este módulo valida el contrato antes de publicar Parquet o entrenar modelos.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "TargetRGEvent"
KEY_COLUMNS = ["RunId", "UserID", "CutoffDate"]
TEMPORAL_COLUMNS = ["ObservationStartDate", "CutoffDate", "HorizonEndDate"]
CONTEXT_COLUMNS = [
    "CountryName",
    "LanguageName",
    "Gender",
    "AgeAtCutoff",
    "DaysSinceRegistration",
    "DaysSinceFirstDeposit",
]

# Lista blanca inicial. Evita que metadatos, atributos sensibles o campos de la
# intervencion entren accidentalmente al modelo de la etapa 2.
RISK_FEATURES_V1 = [
    "ActivityRows",
    "ActiveDays",
    "ProductCount",
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
    "FixedOddsRows",
    "LiveActionRows",
    "PokerRows",
    "LiveActionShare",
    "PokerShare",
    "FixedToLiveTurnoverRatio",
]

FORBIDDEN_LEAKAGE_TOKENS = (
    "rgfirst",
    "rglast",
    "eventtype",
    "interventiontype",
    "rgsumevents",
    "originalrgcase",
)


class RiskDataContractError(ValueError):
    """La tabla analitica incumple una condicion que invalida el modelado."""


@dataclass(frozen=True)
class RiskDataValidation:
    rows: int
    users: int
    positives: int
    positive_rate: float
    split_rows: dict[str, int]
    split_users: dict[str, int]
    missing_cells: int

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "users": self.users,
            "positives": self.positives,
            "positive_rate": self.positive_rate,
            "split_rows": self.split_rows,
            "split_users": self.split_users,
            "missing_cells": self.missing_cells,
        }


def validate_risk_abt(frame: pd.DataFrame) -> RiskDataValidation:
    """Valida claves, ventanas, etiqueta, splits y ausencia de fuga explicita."""
    required = set(KEY_COLUMNS + TEMPORAL_COLUMNS + ["SplitSet", TARGET_COLUMN])
    required.update(RISK_FEATURES_V1)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RiskDataContractError(f"Faltan columnas requeridas: {missing}")
    if frame.empty:
        raise RiskDataContractError("La ABT esta vacia")

    leaked = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in FORBIDDEN_LEAKAGE_TOKENS)
    ]
    if leaked:
        raise RiskDataContractError(f"Columnas con fuga potencial: {leaked}")

    duplicated = frame.duplicated(["RunId", "UserID", "CutoffDate"]).sum()
    if duplicated:
        raise RiskDataContractError(f"Claves duplicadas: {duplicated}")

    targets = set(pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").dropna().astype(int))
    if targets != {0, 1}:
        raise RiskDataContractError(f"La etiqueta debe contener ambas clases 0/1: {targets}")

    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(frame["SplitSet"].dropna().astype(str))
    if observed_splits != allowed_splits:
        raise RiskDataContractError(f"Splits incompletos o invalidos: {observed_splits}")

    split_count_per_user = frame.groupby("UserID")["SplitSet"].nunique()
    if (split_count_per_user > 1).any():
        raise RiskDataContractError("Un cliente aparece en mas de un split")

    observation = pd.to_datetime(frame["ObservationStartDate"])
    cutoff = pd.to_datetime(frame["CutoffDate"])
    horizon = pd.to_datetime(frame["HorizonEndDate"])
    if not ((cutoff - observation).dt.days == 89).all():
        raise RiskDataContractError("La ventana de observacion no es de 90 dias inclusivos")
    if not ((horizon - cutoff).dt.days == 30).all():
        raise RiskDataContractError("El horizonte objetivo no es de 30 dias")

    nonnegative = [
        "ActivityRows",
        "ActiveDays",
        "ProductCount",
        "MonetaryObservedRows",
        "MissingBetsRows",
        "NumberOfBetsTotal",
        "MaxDailyBets",
        "LossDays",
    ]
    if (frame[nonnegative].apply(pd.to_numeric, errors="coerce") < 0).any().any():
        raise RiskDataContractError("Hay conteos negativos en la ABT")
    if not pd.to_numeric(frame["MissingMonetaryPct"], errors="coerce").between(0, 1).all():
        raise RiskDataContractError("MissingMonetaryPct esta fuera de [0,1]")

    feature_frame = frame[RISK_FEATURES_V1]
    return RiskDataValidation(
        rows=int(len(frame)),
        users=int(frame["UserID"].nunique()),
        positives=int(pd.to_numeric(frame[TARGET_COLUMN]).sum()),
        positive_rate=round(float(pd.to_numeric(frame[TARGET_COLUMN]).mean()), 6),
        split_rows={str(k): int(v) for k, v in frame["SplitSet"].value_counts().items()},
        split_users={
            str(k): int(v) for k, v in frame.groupby("SplitSet")["UserID"].nunique().items()
        },
        missing_cells=int(feature_frame.isna().sum().sum()),
    )
