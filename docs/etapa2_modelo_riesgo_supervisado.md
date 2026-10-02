# Etapa 2 — Modelo supervisado de riesgo

## Objetivo

Estimar una probabilidad calibrada de que el cliente tenga su primer evento de
juego responsable durante los 30 días posteriores a la fecha de corte y convertir
esa probabilidad en niveles `Bajo`, `Medio` o `Alto`.

## Diseño experimental

- Fuente: `ml.vw_RiskAnalyticalBaseCurrent` de la etapa 1.
- Entrenamiento, validación y test están separados por cliente.
- La imputación y el escalado viven dentro de pipelines de scikit-learn y se
  ajustan solamente con entrenamiento.
- La calibración utiliza `GroupKFold` por `UserID`; un cliente no aparece en ambos
  lados de un fold.
- La selección del modelo y los umbrales utiliza validación.
- Test se consulta una sola vez después de seleccionar modelo y umbrales.

## Modelos

1. Regresión logística balanceada: baseline interpretable.
2. HistGradientBoosting balanceado: candidato no lineal.

Ambos se calibran con Platt scaling (`sigmoid`). La métrica principal de selección
es PR-AUC por el desbalance de la etiqueta; Brier se usa como criterio secundario.

## Umbrales

- `Medio o superior`: umbral que busca al menos 80% de recall en validación.
- `Alto`: umbral que busca al menos 50% de recall en validación con mayor precisión.
- `Bajo`: probabilidad inferior al umbral medio.

Los umbrales no fuerzan porcentajes fijos de clientes. Las tasas resultantes son
una salida del modelo y pueden revisarse según el costo de falsos negativos.

## Transferibilidad

La lista de variables excluye país, idioma, género, edad y tipos de producto
específicos del operador externo. Se usan únicamente señales de actividad,
frecuencia, apuestas, turnover, pérdidas y tendencias que pueden reconstruirse
con el histórico del casino.

Esto reduce, pero no elimina, el cambio de dominio. Antes de usar el score para
decisiones reales debe validarse con etiquetas propias de Casino Palacio Real.

## Salidas

- Artefacto: `models_store/modelo_riesgo_supervisado.joblib`.
- Métricas: `reports/metrics/metrics_riesgo_supervisado.json`.
- Resultado por nivel: `reports/metrics/riesgo_supervisado_por_nivel.csv`.
- Evaluación visual: `reports/figures/riesgo_supervisado_evaluacion.png`.
- SQL: `ml.RiskModelRun`, `ml.RiskModelPrediction` y
  `ml.vw_RiskModelPredictionCurrent`.

