"""Pruebas de la etapa 3 de respuesta por recompensa."""

from __future__ import annotations

import numpy as np
import pandas as pd

from casino_ia.models.response_nbo_calibrated import (
    TREATMENTS,
    assign_client_split,
    build_response_anchors,
    calibrate_mean_probability,
)


def evidence_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("HillstromEmail", "No E-Mail", 0.10),
            ("HillstromEmail", "Womens E-Mail", 0.15),
            ("HillstromEmail", "Mens E-Mail", 0.18),
            ("CriteoUpliftV21", "0", 0.04),
            ("CriteoUpliftV21", "1", 0.052),
        ],
        columns=["SourceDataset", "TreatmentLabel", "ObservedRate"],
    )


def test_anchor_mapping_is_ordered_and_explicit():
    anchors = build_response_anchors(evidence_frame())
    assert anchors.control_rate == 0.10
    assert np.isclose(anchors.low_rate, 0.13)
    assert anchors.medium_rate == 0.15
    assert anchors.high_rate == 0.18


def test_intercept_calibration_reaches_target_mean():
    raw = np.linspace(-4, 3, 500)
    probability = calibrate_mean_probability(raw, 0.17)
    assert np.isclose(probability.mean(), 0.17, atol=1e-8)
    assert probability.min() > 0
    assert probability.max() < 1


def test_client_split_is_stable_and_exclusive():
    clients = pd.Series([1, 1, 1, 2, 2, 3, 3, 3])
    split = assign_client_split(clients)
    frame = pd.DataFrame({"client": clients, "split": split})
    assert frame.groupby("client")["split"].nunique().eq(1).all()
    assert split.equals(assign_client_split(clients))


def test_treatment_contract_contains_control_and_three_rewards():
    assert TREATMENTS == ("control", "baja", "media", "alta")
