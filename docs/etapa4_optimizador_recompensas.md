# Etapa 4 — Optimizador de recompensas

## Objetivo

Seleccionar como máximo una recompensa por cliente maximizando valor esperado,
respetando riesgo, elegibilidad, presupuesto y topes de concentración.

## Entradas

- Cartera real de 888 clientes: `dbo.vw_FeaturesCliente`.
- Riesgo aplicable a clientes del casino: artefacto `modelo_riesgo.joblib`.
- Baseline V1: `modelo_respuesta.joblib`, con una probabilidad por cliente.
- NBO V2: `ml.vw_ResponseOptionCurrent`, con probabilidad y valor específicos
  para recompensa baja, media y alta.

El modelo supervisado de riesgo de la etapa 2 fue entrenado con usuarios de una
fuente externa. Sus `UserID` no se unen con `IdCliente`; por ello no se realiza una
correspondencia falsa entre identidades. Antes de usarlo directamente en esta
cartera se requiere reconstruir el mismo contrato de variables y validar el cambio
de dominio con etiquetas propias del casino.

## Método

Se conserva el greedy existente como algoritmo de optimización:

1. Para cada cliente se consideran únicamente las recompensas permitidas por riesgo.
2. Se elige la opción individual de mayor valor esperado positivo.
3. Los candidatos se ordenan por `ValorEsperado / Costo`.
4. Se recorre toda la lista aplicando presupuesto y topes; rechazar un candidato
   no detiene la evaluación de los siguientes.

El mismo greedy y los mismos guardrails se ejecutan con V1 para obtener un baseline
comparable. La única diferencia entre ambos escenarios es la información de respuesta:
V1 usa `P(respuesta | cliente)` y V2 usa
`P(respuesta | cliente, recompensa)`.

## Guardrails conservados

- Riesgo alto: sin recompensa.
- Riesgo medio: únicamente recompensa baja.
- Riesgo bajo: recompensa baja, media o alta.
- Mínimo tres sesiones en los últimos 90 días.
- Solo opciones con valor esperado positivo.
- Una decisión y como máximo una recompensa por cliente.
- Gasto total no superior al presupuesto.
- Máximo 25% del presupuesto en recompensas altas.
- Máximo 40% del presupuesto por segmento.

El histórico del casino cubre 15 días. `NroSesiones` representa las sesiones
observadas dentro de ese periodo, que está completamente contenido en 90 días.

## Persistencia y trazabilidad

- `ml.RewardAllocationRun`: configuración, hashes, métricas y comparación V2/V1.
- `ml.RewardAllocationDecision`: una fila por cliente, opción sugerida, asignación
  final, valor económico, baseline y `DecisionReason`.
- `ml.vw_RewardAllocationCurrent`: última decisión completa publicada.
- `ml.vw_RewardAllocationSummaryCurrent`: resumen de la última corrida.
- `reports/metrics/plan_asignacion.csv`: exportación auditable.
- `reports/metrics/metrics_optimizador_recompensas.json`: métricas de la corrida.

La publicación es transaccional e idempotente para la misma combinación de modelos,
presupuesto y parámetros del optimizador.
