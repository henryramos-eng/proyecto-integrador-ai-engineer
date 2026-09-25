# Guía de asignación de recompensas (recomendación de siguiente mejor oferta)

> Documento de referencia para el asistente. Simulado, fines académicos.

## Cómo decide el sistema qué recompensa recomendar

Para cada cliente elegible y cada tipo de recompensa se calcula el **valor esperado**:

    valor_esperado = probabilidad_de_respuesta * valor_incremental_estimado - costo_de_la_recompensa

- **probabilidad_de_respuesta**: la entrega el modelo de propensión (0 a 1).
- **valor_incremental_estimado**: fracción del valor teórico de la casa del cliente
  por sesión, ajustada por su tendencia de actividad reciente.
- **costo_de_la_recompensa**: alta = S/ 40, media = S/ 15, baja = S/ 5.

Se asigna a cada cliente la recompensa con mayor valor esperado (si es positivo) y
luego se ordena la lista por **eficiencia** = valor_esperado / costo. Se recorre esa
lista asignando hasta agotar el presupuesto de la campaña.

## Cuándo recomendar cada tipo

| Situación del cliente | Recomendación sugerida |
|---|---|
| Riesgo bajo, propensión alta (decil 8-10), alto valor histórico | Recompensa alta |
| Riesgo bajo, propensión media (decil 5-7) | Recompensa media |
| Propensión baja (decil 1-4) o valor incremental chico | Recompensa baja o solo nutrición de marca |
| Riesgo medio | Solo recompensa baja |
| Riesgo alto | Ninguna recompensa. Derivar a juego responsable |

## Qué NO hacer

- No recomendar por segmento solamente (VIP/Estándar); el segmento es un dato más,
  no la regla.
- No asignar una recompensa cuyo valor esperado sea negativo, aunque quede
  presupuesto.
- No superar los topes: máximo 25 % del presupuesto en recompensas altas y máximo
  40 % concentrado en un mismo segmento.

## Versión adaptativa (contextual bandit)

Cuando haya resultados de campañas anteriores, la política de recomendación se
vuelve un *contextual bandit* (Thompson Sampling): aprende en línea qué recompensa
funciona mejor para cada perfil, equilibrando explotar lo conocido y explorar
alternativas.
