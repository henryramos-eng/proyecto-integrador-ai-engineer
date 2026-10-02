# Etapa 3 — Modelo de respuesta por recompensa

## Objetivo

Estimar `P(respuesta | cliente, recompensa)` para recompensas `baja`, `media` y
`alta`, además de valor incremental, costo y valor esperado.

## Naturaleza de los datos

Casino Palacio Real no dispone todavía de campañas históricas con los tres tipos
de recompensa y resultado observado. Por autorización del proyecto, se utiliza
una campaña **semi-sintética calibrada con evidencia real**:

- Control: tasa de visita del grupo `No E-Mail` de Hillstrom.
- Baja: uplift relativo de visita del tratamiento Criteo aplicado al control.
- Media: tasa de visita de `Womens E-Mail` de Hillstrom.
- Alta: tasa de visita de `Mens E-Mail` de Hillstrom.

La correspondencia baja/media/alta es una hipótesis de simulación. Los tratamientos
originales no son recompensas monetarias equivalentes y nunca se presentan como
si lo fueran.

## Diseño

- Ocho campañas, una exposición por cliente y campaña.
- Rotación balanceada: cada cliente aparece dos veces en control, baja, media y alta.
- La respuesta se genera con semilla 42 y probabilidades medias ancladas en los RCT.
- La heterogeneidad usa únicamente atributos reales del cliente ya disponibles.
- El valor económico se deriva de `ValorTeoricoCasa`, tendencia y la configuración
  existente `factor_uplift`/costos.
- Train, validación y test se separan por cliente; un cliente nunca cruza splits.

## Modelado

- Baseline: regresión logística calibrada.
- Candidato: HistGradientBoosting calibrado.
- Selección: PR-AUC de validación y Brier como desempate.
- Modelo adicional de valor si responde: HistGradientBoostingRegressor.

## Salida por cliente–recompensa

- `ControlProbability`
- `ResponseProbability`
- `UpliftProbability`
- `ValueIncremental`
- `Cost`
- `ExpectedValue = ResponseProbability × ValueIncremental − Cost`
- `IncrementalExpectedValue = UpliftProbability × ValueIncremental − Cost`

## Artefactos

- SQL: `ml.ResponseModelRun`, `ml.ResponseEvidenceAnchor`,
  `ml.ResponseCampaignSynthetic`, `ml.ResponseOption` y vistas actuales.
- Modelo: `models_store/modelo_respuesta_nbo_calibrado.joblib`.
- Histórico: `data/processed/historico_campanas_rct_calibrado.parquet`.
- Opciones: `data/processed/opciones_respuesta_nbo.parquet`.
- Métricas: `reports/metrics/metrics_respuesta_nbo_calibrado.json`.
- EDA: `reports/figures/respuesta_nbo_evaluacion.png`.

Este modelo es adecuado para demostrar y probar el optimizador. Debe reemplazarse
o recalibrarse cuando existan campañas reales del casino con tratamiento, costo,
respuesta y ventana de atribución documentados.

