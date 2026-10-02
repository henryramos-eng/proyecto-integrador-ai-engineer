# Etapa 1 — Preparación de datos para riesgo supervisado

## Objetivo

Construir una tabla analítica reproducible para predecir si un cliente tendrá su
primer evento de juego responsable durante los próximos 30 días, utilizando
exclusivamente sus 90 días previos de actividad.

## Unidad y ventanas

- Unidad: `UserID + CutoffDate`.
- Observación: desde `CutoffDate - 89 días` hasta `CutoffDate`, inclusivo.
- Predicción: desde `CutoffDate + 1` hasta `CutoffDate + 30`.
- Cortes: 14 fechas separadas por 30 días, desde 2008-11-01.
- Etiqueta: `TargetRGEvent`, igual a 1 cuando el primer evento RG ocurre dentro
  del horizonte de predicción.

Esta construcción produce un panel temporal. Un cliente puede tener más de una
ventana, pero siempre pertenece a un único split (`train`, `validation` o `test`).

## Prevención de fuga

Las fechas, tipos y resultados de las intervenciones RG se usan únicamente para
crear la etiqueta. No forman parte de la lista blanca de predictores. Tampoco se
usan el indicador global `RGCase` ni actividad posterior a la fecha de corte.

Los atributos demográficos se conservan para auditoría de representatividad y
equidad, pero se excluyen de la lista de predictores V1.

## Exclusiones y nulos

- `EventTypeFirst=2`: corresponde a una apelación de una intervención previa;
  su fecha no representa un inicio fiable. Se excluye y se registra el motivo.
- Ventanas sin actividad en los 90 días: se rechazan explícitamente; no se inventa
  actividad ni se reemplaza con ceros.
- `Turnover` y `Hold` faltantes: permanecen como `NULL`. El dataset documenta un
  problema de transferencia en productos de terceros. Se añade
  `MissingMonetaryPct` para representar esta ausencia.
- Claves diarias repetidas: se agregan de forma determinística; el conteo original
  queda registrado como advertencia de calidad.

## Artefactos

- SQL: `ml.RiskAnalyticalBase` y vista `ml.vw_RiskAnalyticalBaseCurrent`.
- Rechazos: `ml.RiskDataRejection`.
- Auditoría: `ml.RiskDataPreparationRun` y `ml.RiskDataQualitySummary`.
- Parquet: `data/processed/abt_riesgo_supervisado.parquet`.
- Calidad: `reports/metrics/riesgo_etapa1_calidad.json`.
- Resumen y correlaciones: `reports/metrics/riesgo_etapa1_*.csv`.
- EDA: `reports/figures/riesgo_etapa1_eda.png`.

La etapa no imputa, escala, balancea ni entrena modelos. Esas transformaciones se
ajustarán exclusivamente con el conjunto de entrenamiento en la etapa 2.

