"""Optimización de la asignación de recompensas.

Dado el scoring por cliente (riesgo + probabilidad de respuesta + valor teórico),
elige qué recompensa asignar a cada cliente maximizando el retorno esperado de la
campaña sujeto a un presupuesto.

    valor_esperado(c, r) = P(respuesta | c, r) * valor_incremental(c, r) - costo(r)

Restricciones:
  * presupuesto total de la campaña
  * NivelRiesgo == 'Alto'  -> no elegible (guardrail de juego responsable)
  * NivelRiesgo == 'Medio' -> solo recompensa 'baja'
  * al menos 3 sesiones registradas en los últimos 90 días
  * máximo 25 % del presupuesto en recompensas altas
  * máximo 40 % del presupuesto concentrado en un mismo segmento

Método: ranking por eficiencia (valor_esperado / costo) y selección greedy tipo
mochila hasta agotar el presupuesto. El baseline de comparación es la asignación
por reglas de segmento.
"""

from __future__ import annotations

import pandas as pd

from casino_ia.config import REWARDS

RECOMPENSAS = ("alta", "media", "baja")
MIN_SESIONES_90D = 3
MAX_FRACCION_ALTAS = 0.25
MAX_FRACCION_SEGMENTO = 0.40
_PERMITIDAS_POR_RIESGO = {
    "Bajo": ("alta", "media", "baja"),
    "Medio": ("baja",),
    "Alto": (),
}

_COLUMNAS_DECISION = [
    "IdCliente",
    "Segmento",
    "NivelRiesgo",
    "ProbRespuesta",
    "SesionesUltimos90d",
    "RecompensaSugerida",
    "CostoSugerido",
    "Recompensa",
    "Costo",
    "ValorIncremental",
    "ValorEsperado",
    "Eficiencia",
    "GastoAcumulado",
    "Asignada",
    "MotivoDecision",
]


def _uplift_valor(fila: pd.Series) -> float:
    """Valor incremental esperado si el cliente responde.

    Se aproxima como una fracción del valor teórico de la casa por sesión,
    ajustada por la tendencia reciente de actividad.
    """
    base = max(float(fila.get("ValorTeoricoCasa", 0.0)), 0.0)
    por_sesion = base / max(float(fila.get("NroSesiones", 1)), 1.0)
    tendencia = fila.get("RatioTendenciaCoinIn", 1.0)
    tendencia = 1.0 if pd.isna(tendencia) or not tendencia else float(tendencia)
    return por_sesion * REWARDS.factor_uplift * min(max(tendencia, 0.5), 2.0)


def _sesiones_ultimos_90d(fila: pd.Series) -> int:
    """Obtiene la ventana de 90 días o usa el periodo disponible del prototipo."""
    valor = fila.get("SesionesUltimos90d", fila.get("NroSesiones", 0))
    return 0 if pd.isna(valor) else int(valor)


def evaluar_recompensas(
    scoring: pd.DataFrame,
    costos: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Evalúa cada recompensa permitida con datos V2 o el cálculo V1 de respaldo.

    V2 se reconoce por las columnas ``ProbRespuesta_<tipo>`` y
    ``ValorIncremental_<tipo>``. Si no existen, se conserva el comportamiento V1
    con una probabilidad única y el uplift derivado de las features del cliente.
    """

    costos = costos or REWARDS.costo
    filas = []
    for _, cliente in scoring.iterrows():
        permitidas = _PERMITIDAS_POR_RIESGO.get(str(cliente["NivelRiesgo"]), ())
        for recompensa in permitidas:
            columna_prob = f"ProbRespuesta_{recompensa}"
            columna_valor = f"ValorIncremental_{recompensa}"
            prob = float(cliente.get(columna_prob, cliente.get("ProbRespuesta", 0.0)))
            valor_incremental = float(
                cliente.get(columna_valor, _uplift_valor(cliente))
            )
            costo = float(costos[recompensa])
            valor_esperado = prob * valor_incremental - costo
            filas.append(
                {
                    "IdCliente": cliente["IdCliente"],
                    "Recompensa": recompensa,
                    "ProbRespuesta": round(prob, 4),
                    "ValorIncremental": round(valor_incremental, 4),
                    "Costo": costo,
                    "ValorEsperado": round(valor_esperado, 2),
                    "Eficiencia": round(valor_esperado / costo, 3),
                }
            )
    return pd.DataFrame(
        filas,
        columns=[
            "IdCliente",
            "Recompensa",
            "ProbRespuesta",
            "ValorIncremental",
            "Costo",
            "ValorEsperado",
            "Eficiencia",
        ],
    )


def asignar_recompensas(
    scoring: pd.DataFrame,
    presupuesto: float | None = None,
    costos: dict[str, float] | None = None,
    max_fraccion_altas: float = MAX_FRACCION_ALTAS,
    max_fraccion_segmento: float = MAX_FRACCION_SEGMENTO,
) -> pd.DataFrame:
    """Devuelve una decisión trazable por cliente usando el baseline greedy.

    `scoring` debe tener las variables V1 y, para V2, ProbRespuesta_<tipo> y
    ValorIncremental_<tipo>. La probabilidad V1 se conserva para el baseline.

    Si existe SesionesUltimos90d, se usa para la elegibilidad. En el prototipo,
    cuyo histórico abarca menos de 90 días, NroSesiones representa esa ventana.
    """
    presupuesto = REWARDS.presupuesto if presupuesto is None else presupuesto
    costos = costos or REWARDS.costo
    opciones = evaluar_recompensas(scoring, costos)
    opciones_por_cliente = {
        id_cliente: grupo for id_cliente, grupo in opciones.groupby("IdCliente")
    }

    decisiones = []
    for _, fila in scoring.iterrows():
        nivel = str(fila["NivelRiesgo"])
        sesiones_90d = _sesiones_ultimos_90d(fila)
        decision = {
            "IdCliente": fila["IdCliente"],
            "Segmento": fila.get("Segmento"),
            "NivelRiesgo": nivel,
            "ProbRespuesta": round(float(fila.get("ProbRespuesta", 0.0)), 4),
            "SesionesUltimos90d": sesiones_90d,
            "RecompensaSugerida": None,
            "CostoSugerido": 0.0,
            "Recompensa": None,
            "Costo": 0.0,
            "ValorIncremental": None,
            "ValorEsperado": None,
            "Eficiencia": None,
            "GastoAcumulado": None,
            "Asignada": False,
            "MotivoDecision": "",
        }

        if nivel == "Alto":
            decision["MotivoDecision"] = "Sin recompensa: riesgo alto"
            decisiones.append(decision)
            continue
        if sesiones_90d < MIN_SESIONES_90D:
            decision["MotivoDecision"] = "No elegible: menos de 3 sesiones en 90 días"
            decisiones.append(decision)
            continue

        permitidas = _PERMITIDAS_POR_RIESGO.get(nivel, ())
        if not permitidas:
            decision["MotivoDecision"] = "No elegible: nivel de riesgo no reconocido"
            decisiones.append(decision)
            continue

        opciones_cliente = opciones_por_cliente.get(fila["IdCliente"])
        if opciones_cliente is None or opciones_cliente.empty:
            decision["MotivoDecision"] = "No elegible: sin recompensas permitidas"
            decisiones.append(decision)
            continue
        mejor = opciones_cliente.loc[opciones_cliente["ValorEsperado"].idxmax()]
        decision["ProbRespuesta"] = mejor["ProbRespuesta"]
        decision["ValorIncremental"] = mejor["ValorIncremental"]
        decision["ValorEsperado"] = mejor["ValorEsperado"]
        decision["Eficiencia"] = mejor["Eficiencia"]
        if mejor["ValorEsperado"] <= 0:
            decision["MotivoDecision"] = "Sin recompensa: valor esperado no positivo"
        else:
            decision["RecompensaSugerida"] = mejor["Recompensa"]
            decision["CostoSugerido"] = mejor["Costo"]
            decision["MotivoDecision"] = "Pendiente de evaluación greedy"
        decisiones.append(decision)

    if not decisiones:
        return pd.DataFrame(columns=_COLUMNAS_DECISION)

    cand = pd.DataFrame(decisiones, columns=_COLUMNAS_DECISION)
    candidatas = cand["RecompensaSugerida"].notna()
    cand = pd.concat(
        [
            cand[candidatas].sort_values("Eficiencia", ascending=False),
            cand[~candidatas],
        ],
        ignore_index=True,
    )

    # Selección greedy con topes. Un rechazo no corta el recorrido: un candidato
    # posterior de menor costo todavía puede entrar en el presupuesto disponible.
    gasto_total = 0.0
    gasto_altas = 0.0
    gasto_segmento: dict[str, float] = {}
    limite_altas = presupuesto * max_fraccion_altas
    limite_segmento = presupuesto * max_fraccion_segmento

    for idx in cand.index[cand["RecompensaSugerida"].notna()]:
        costo = float(cand.at[idx, "CostoSugerido"])
        recompensa = cand.at[idx, "RecompensaSugerida"]
        segmento = str(cand.at[idx, "Segmento"])

        if gasto_total + costo > presupuesto:
            cand.at[idx, "GastoAcumulado"] = gasto_total
            cand.at[idx, "MotivoDecision"] = "No asignada: presupuesto insuficiente"
            continue
        if recompensa == "alta" and gasto_altas + costo > limite_altas:
            cand.at[idx, "GastoAcumulado"] = gasto_total
            cand.at[idx, "MotivoDecision"] = "No asignada: tope 25% en recompensas altas"
            continue
        if gasto_segmento.get(segmento, 0.0) + costo > limite_segmento:
            cand.at[idx, "GastoAcumulado"] = gasto_total
            cand.at[idx, "MotivoDecision"] = "No asignada: tope 40% por segmento"
            continue

        gasto_total += costo
        gasto_segmento[segmento] = gasto_segmento.get(segmento, 0.0) + costo
        if recompensa == "alta":
            gasto_altas += costo
        cand.at[idx, "Recompensa"] = recompensa
        cand.at[idx, "Costo"] = costo
        cand.at[idx, "GastoAcumulado"] = gasto_total
        cand.at[idx, "Asignada"] = True
        cand.at[idx, "MotivoDecision"] = "Asignada"

    asignadas = cand[cand["Asignada"]]
    cand.attrs["presupuesto"] = presupuesto
    cand.attrs["gasto_total"] = float(asignadas["Costo"].sum())
    cand.attrs["valor_esperado_total"] = float(asignadas["ValorEsperado"].sum())
    cand.attrs["clientes_asignados"] = int(len(asignadas))
    cand.attrs["candidatos_totales"] = int(candidatas.sum())
    cand.attrs["decisiones_totales"] = int(len(cand))
    return cand


def baseline_reglas(scoring: pd.DataFrame, costos: dict[str, float] | None = None) -> pd.DataFrame:
    """Asignación por reglas de segmento (para comparar contra el optimizador)."""
    costos = costos or REWARDS.costo
    regla = {"VIP": "alta", "Alto": "media", "Medio": "baja", "Estandar": "baja"}
    df = scoring.copy()
    df["SesionesUltimos90d"] = df.apply(_sesiones_ultimos_90d, axis=1)
    df = df[
        (df["NivelRiesgo"] != "Alto")
        & (df["SesionesUltimos90d"] >= MIN_SESIONES_90D)
    ]
    df["Recompensa"] = df["Segmento"].map(regla).fillna("baja")
    df.loc[df["NivelRiesgo"] == "Medio", "Recompensa"] = "baja"
    df["Costo"] = df["Recompensa"].map(costos)
    df["ValorEsperado"] = df.apply(
        lambda f: float(f["ProbRespuesta"]) * _uplift_valor(f) - f["Costo"], axis=1
    ).round(2)
    return df
