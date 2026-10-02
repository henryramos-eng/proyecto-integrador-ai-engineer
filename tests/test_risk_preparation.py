"""Pruebas del contrato de datos para riesgo supervisado."""

from __future__ import annotations

import pandas as pd
import pytest

from casino_ia.features.risk_supervised import (
    CONTEXT_COLUMNS,
    RISK_FEATURES_V1,
    RiskDataContractError,
    validate_risk_abt,
)


def sample_abt() -> pd.DataFrame:
    rows = []
    splits = {101: "train", 102: "validation", 103: "test", 104: "train"}
    for index, (user_id, split) in enumerate(splits.items()):
        row = {
            "RunId": 1,
            "UserID": user_id,
            "CutoffDate": pd.Timestamp("2009-01-01") + pd.Timedelta(days=index * 30),
            "ObservationStartDate": pd.Timestamp("2008-10-04") + pd.Timedelta(days=index * 30),
            "HorizonEndDate": pd.Timestamp("2009-01-31") + pd.Timedelta(days=index * 30),
            "SplitSet": split,
            "TargetRGEvent": index % 2,
        }
        row.update({column: None for column in CONTEXT_COLUMNS})
        row.update({column: float(index + 1) for column in RISK_FEATURES_V1})
        row["MissingMonetaryPct"] = 0.25
        rows.append(row)
    return pd.DataFrame(rows)


def test_valid_abt_contract():
    result = validate_risk_abt(sample_abt())
    assert result.rows == 4
    assert result.users == 4
    assert result.positives == 2


def test_rejects_explicit_target_leakage():
    frame = sample_abt()
    frame["RGFirstDate"] = pd.Timestamp("2009-02-01")
    with pytest.raises(RiskDataContractError, match="fuga"):
        validate_risk_abt(frame)


def test_rejects_client_shared_between_splits():
    frame = pd.concat([sample_abt(), sample_abt().iloc[[0]]], ignore_index=True)
    frame.loc[len(frame) - 1, "CutoffDate"] += pd.Timedelta(days=30)
    frame.loc[len(frame) - 1, "ObservationStartDate"] += pd.Timedelta(days=30)
    frame.loc[len(frame) - 1, "HorizonEndDate"] += pd.Timedelta(days=30)
    frame.loc[len(frame) - 1, "SplitSet"] = "test"
    with pytest.raises(RiskDataContractError, match="mas de un split"):
        validate_risk_abt(frame)


def test_rejects_wrong_temporal_window():
    frame = sample_abt()
    frame.loc[0, "ObservationStartDate"] = frame.loc[0, "CutoffDate"]
    with pytest.raises(RiskDataContractError, match="90 dias"):
        validate_risk_abt(frame)
