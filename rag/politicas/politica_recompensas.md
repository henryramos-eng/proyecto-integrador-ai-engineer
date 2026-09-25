# Política de asignación de recompensas — Casino Palacio Real

> Documento simulado para fines académicos (Proyecto Integrador).

## 1. Objetivo

Asignar el presupuesto promocional a los clientes con mayor retorno esperado,
de forma trazable y respetando el marco de juego responsable.

## 2. Tipos de recompensa

| Tipo | Contenido | Costo referencial |
|---|---|---|
| Alta | Bono de juego + cena + estacionamiento | S/ 40 |
| Media | Bono de juego + bebida de cortesía | S/ 15 |
| Baja | Puntos extra de fidelidad | S/ 5 |

## 3. Elegibilidad

- Solo clientes con al menos 3 sesiones registradas en los últimos 90 días.
- Un cliente recibe **como máximo una** recompensa por campaña.
- Los clientes clasificados como **riesgo alto** NO son elegibles para ninguna
  recompensa y se derivan al protocolo de juego responsable.
- Los clientes de **riesgo medio** solo pueden recibir recompensa baja.

## 4. Criterio de priorización

Se ordena a los clientes elegibles por eficiencia:

    eficiencia = (probabilidad_respuesta * valor_incremental_estimado - costo) / costo

Se asignan recompensas en ese orden hasta agotar el presupuesto de la campaña.

## 5. Topes

- Máximo 25 % del presupuesto en recompensas de tipo alto.
- Máximo 40 % del presupuesto concentrado en un mismo segmento.

## 6. Registro

Cada asignación queda registrada con: cliente, recompensa, costo, score de riesgo,
probabilidad de respuesta y fecha. El registro se conserva para evaluar el ROI de
la campaña frente al baseline.
