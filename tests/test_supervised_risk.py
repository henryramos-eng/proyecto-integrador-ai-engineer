"""Pruebas de funciones puras del modelo supervisado de riesgo."""

from __future__ import annotations

import numpy as np

from casino_ia.models.risk_supervised import (
    RISK_FEATURES_TRANSFERABLE,
    evaluate_probabilities,
    risk_levels,
    select_risk_thresholds,
    threshold_metrics,
)


def test_thresholds_are_ordered_and_use_validation_recall():
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    probability = np.array([0.02, 0.08, 0.12, 0.20, 0.35, 0.18, 0.40, 0.55, 0.75, 0.90])
    thresholds = select_risk_thresholds(
        y, probability, medium_recall_target=0.80, high_recall_target=0.50
    )
    assert 0 <= thresholds["medium"] <= thresholds["high"] <= 1
    assert threshold_metrics(y, probability, thresholds["medium"])["recall"] >= 0.80
    assert threshold_metrics(y, probability, thresholds["high"])["recall"] >= 0.50


def test_levels_respect_thresholds():
    levels = risk_levels(np.array([0.05, 0.30, 0.80]), {"medium": 0.20, "high": 0.60})
    assert levels.tolist() == ["Bajo", "Medio", "Alto"]


def test_probability_metrics_for_perfect_ranking():
    metrics = evaluate_probabilities(np.array([0, 0, 1, 1]), np.array([0.01, 0.10, 0.80, 0.95]))
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["brier"] < 0.03


def test_transferable_features_exclude_external_product_types():
    forbidden = {"ProductCount", "FixedOddsRows", "LiveActionRows", "PokerRows"}
    assert forbidden.isdisjoint(RISK_FEATURES_TRANSFERABLE)
