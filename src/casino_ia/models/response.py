"""Modelo de probabilidad de respuesta a una recompensa.

Clasificación binaria calibrada. En el prototipo la etiqueta es un *proxy*:
"el cliente muestra momentum positivo de actividad" (su gasto de la última semana
supera lo esperado si su ritmo fuera constante). En producción se reemplaza por
el resultado real de campañas (con grupo de control ⇒ modelo de uplift).
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from casino_ia.config import FECHA_CORTE, REWARDS

# Se excluyen las variables de ventana reciente para no filtrar la etiqueta.
FEATURES_RESPUESTA = [
    "DiasDesdeUltimaSesion",
    "NroSesiones",
    "DiasActivos",
    "CoinInTotal",
    "CoinInPromedioSesion",
    "ValorTeoricoCasa",
    "CompsAcumulados",
    "PuntosAcumulados",
    "ApuestaMediaPromedio",
    "HorasJugadas",
    "AntiguedadDias",
]

# El periodo simulado tiene 15 días; la ventana "últimos 7" vs "8 previos"
# implica ratio esperado ~7/8 si el ritmo fuese constante.
UMBRAL_MOMENTUM = 7 / 8

RECOMPENSAS_NBO = ("baja", "media", "alta")
TIPOS_CAMPANA = ("control", *RECOMPENSAS_NBO)
N_CAMPANAS_SIMULADAS = 8
SEMILLA_SIMULACION = 42


def etiqueta_proxy(feats: pd.DataFrame) -> pd.Series:
    ratio = feats["RatioTendenciaCoinIn"].fillna(0.0)
    return (ratio >= UMBRAL_MOMENTUM).astype(int)


def simular_historico_campanas(
    feats: pd.DataFrame,
    riesgo: pd.DataFrame,
    prob_base: pd.DataFrame,
    n_campanas: int = N_CAMPANAS_SIMULADAS,
    random_state: int = SEMILLA_SIMULACION,
) -> pd.DataFrame:
    """Crea exposiciones semi-sintéticas sin inventar atributos de clientes.

    Las características, el valor previo y la elegibilidad proceden del histórico
    disponible. Solo se simulan la exposición y su respuesta, porque el proyecto
    no contiene campañas observadas. La asignación rota por campaña para que cada
    tratamiento quede representado sin vulnerar los guardrails de riesgo.
    """

    requeridas = {
        "IdCliente",
        "ValorTeoricoCasa",
        "NroSesiones",
        "RatioTendenciaCoinIn",
        "PuntosAcumulados",
        "CompsAcumulados",
        *FEATURES_RESPUESTA,
    }
    faltantes = sorted(requeridas.difference(feats.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas para simular campañas: {faltantes}")

    base = (
        feats.copy()
        .merge(
            riesgo[["IdCliente", "NivelRiesgo"]],
            on="IdCliente",
            how="left",
            validate="one_to_one",
        )
        .merge(
            prob_base[["IdCliente", "ProbRespuesta"]].rename(
                columns={"ProbRespuesta": "ProbRespuestaBaseV1"}
            ),
            on="IdCliente",
            how="left",
            validate="one_to_one",
        )
        .sort_values("IdCliente")
        .reset_index(drop=True)
    )
    if base[["NivelRiesgo", "ProbRespuestaBaseV1"]].isna().any().any():
        raise ValueError("Riesgo y probabilidad V1 deben cubrir todos los clientes")

    valor_pct = base["ValorTeoricoCasa"].rank(pct=True).fillna(0.5)
    puntos_pct = base["PuntosAcumulados"].rank(pct=True).fillna(0.5)
    comps_pct = base["CompsAcumulados"].rank(pct=True).fillna(0.5)
    ajuste_medio = (1.0 - (valor_pct - 0.6).abs() / 0.6).clip(0.0, 1.0)
    base["AfinidadBaja"] = ((1.0 - valor_pct) + puntos_pct) / 2.0
    base["AfinidadMedia"] = (ajuste_medio + comps_pct) / 2.0
    base["AfinidadAlta"] = (valor_pct + comps_pct) / 2.0

    sesiones = pd.to_numeric(base["NroSesiones"], errors="coerce").clip(lower=1)
    valor_sesion = (
        pd.to_numeric(base["ValorTeoricoCasa"], errors="coerce").fillna(0.0)
        / sesiones
    ).clip(lower=0.0)
    tendencia = (
        pd.to_numeric(base["RatioTendenciaCoinIn"], errors="coerce")
        .fillna(1.0)
        .clip(0.5, 2.0)
    )
    base["ValorPreCampana"] = valor_sesion * tendencia

    fecha_observada = pd.to_datetime(
        base.get("UltimaSesion", pd.Series(dtype="datetime64[ns]")),
        errors="coerce",
    ).max()
    fecha_inicio = (
        fecha_observada.normalize() + pd.Timedelta(days=1)
        if pd.notna(fecha_observada)
        else pd.Timestamp(FECHA_CORTE) + pd.Timedelta(days=1)
    )
    rng = np.random.default_rng(random_state)
    max_costo = max(REWARDS.costo.values())
    campanas = []

    for nro in range(n_campanas):
        exp = base[base["NivelRiesgo"].isin(["Bajo", "Medio"])].copy()
        orden = np.arange(len(exp))
        tipo = np.full(len(exp), "control", dtype=object)
        es_bajo = exp["NivelRiesgo"].eq("Bajo").to_numpy()
        es_medio = exp["NivelRiesgo"].eq("Medio").to_numpy()
        tipo[es_bajo] = np.asarray(TIPOS_CAMPANA, dtype=object)[
            (orden[es_bajo] + nro) % len(TIPOS_CAMPANA)
        ]
        tipo[es_medio] = np.asarray(("control", "baja"), dtype=object)[
            (orden[es_medio] + nro) % 2
        ]
        exp["TipoRecompensa"] = tipo

        afinidad = np.select(
            [
                exp["TipoRecompensa"].eq("baja"),
                exp["TipoRecompensa"].eq("media"),
                exp["TipoRecompensa"].eq("alta"),
            ],
            [exp["AfinidadBaja"], exp["AfinidadMedia"], exp["AfinidadAlta"]],
            default=0.5,
        )
        costo = exp["TipoRecompensa"].map({"control": 0.0, **REWARDS.costo})
        intensidad = costo / max_costo
        p_base = exp["ProbRespuestaBaseV1"].clip(0.001, 0.999)
        logit_base = np.log(p_base / (1.0 - p_base))
        p_respuesta = 1.0 / (
            1.0 + np.exp(-(logit_base + 1.5 * intensidad * (2.0 * afinidad - 0.5)))
        )
        respondio = rng.random(len(exp)) < p_respuesta
        valor_si_responde = exp["ValorPreCampana"] * (
            1.0 + 1.5 * intensidad * afinidad
        )
        valor_incremental = np.where(respondio, valor_si_responde, 0.0)

        fecha_oferta = fecha_inicio + pd.Timedelta(days=14 * nro)
        dias_respuesta = rng.integers(1, 8, size=len(exp))
        fecha_respuesta = pd.Series(pd.NaT, index=exp.index, dtype="datetime64[ns]")
        fecha_respuesta.loc[respondio] = fecha_oferta + pd.to_timedelta(
            dias_respuesta[respondio], unit="D"
        )

        exp["IdCampana"] = f"SIM-{nro + 1:02d}"
        exp["FechaOferta"] = fecha_oferta
        exp["Respondio"] = respondio.astype(int)
        exp["FechaRespuesta"] = fecha_respuesta
        exp["Costo"] = costo.to_numpy()
        exp["ProbRespuestaSimulada"] = p_respuesta
        exp["ValorIncrementalSiResponde"] = valor_si_responde
        exp["ValorIncremental"] = valor_incremental
        exp["ValorPostCampana"] = exp["ValorPreCampana"] + valor_incremental
        exp["FuenteDatos"] = "semi_sintetico_desde_features_cliente"
        campanas.append(exp)

    historico = pd.concat(campanas, ignore_index=True)
    _validar_historico_simulado(historico)
    return historico


def _validar_historico_simulado(historico: pd.DataFrame) -> None:
    requeridas = {
        "IdCampana",
        "FechaOferta",
        "IdCliente",
        "TipoRecompensa",
        "Respondio",
        "ValorIncremental",
        "NivelRiesgo",
    }
    faltantes = sorted(requeridas.difference(historico.columns))
    if faltantes:
        raise ValueError(f"Histórico simulado incompleto: {faltantes}")
    if historico[list(requeridas)].isna().any().any():
        raise ValueError("El histórico simulado contiene nulos en campos obligatorios")
    if historico.duplicated(["IdCampana", "IdCliente"]).any():
        raise ValueError("IdCampana + IdCliente debe ser una clave única")
    if not set(historico["TipoRecompensa"]).issubset(TIPOS_CAMPANA):
        raise ValueError("TipoRecompensa fuera del dominio permitido")
    if not set(historico["Respondio"]).issubset({0, 1}):
        raise ValueError("Respondio debe ser binario")
    if historico["NivelRiesgo"].eq("Alto").any():
        raise ValueError("El histórico no debe exponer clientes de riesgo alto")
    medios_invalidos = historico["NivelRiesgo"].eq("Medio") & ~historico[
        "TipoRecompensa"
    ].isin(["control", "baja"])
    if medios_invalidos.any():
        raise ValueError("Riesgo medio solo admite control o recompensa baja")


@dataclass
class ModeloRespuesta:
    features: list[str] = None
    _scaler: StandardScaler = None
    _model: CalibratedClassifierCV = None
    metrics_: dict = None

    def __post_init__(self):
        self.features = self.features or FEATURES_RESPUESTA

    def fit(self, feats: pd.DataFrame) -> ModeloRespuesta:
        y = etiqueta_proxy(feats)
        x = feats[self.features].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True))

        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, random_state=42, stratify=y
        )
        self._scaler = StandardScaler().fit(x_tr)

        base = GradientBoostingClassifier(random_state=42)
        self._model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._model.fit(self._scaler.transform(x_tr), y_tr)

        p_te = self._model.predict_proba(self._scaler.transform(x_te))[:, 1]
        baseline = LogisticRegression(max_iter=1000).fit(
            self._scaler.transform(x_tr), y_tr
        )
        p_base = baseline.predict_proba(self._scaler.transform(x_te))[:, 1]

        self.metrics_ = {
            "tasa_positivos": round(float(y.mean()), 3),
            "roc_auc": round(float(roc_auc_score(y_te, p_te)), 3),
            "pr_auc": round(float(average_precision_score(y_te, p_te)), 3),
            "brier": round(float(brier_score_loss(y_te, p_te)), 3),
            "roc_auc_baseline_logistica": round(float(roc_auc_score(y_te, p_base)), 3),
            "lift_top_decil": round(float(_lift_top_decil(y_te, p_te)), 2),
        }
        return self

    def predict_proba(self, feats: pd.DataFrame) -> pd.DataFrame:
        x = feats[self.features].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True))
        p = self._model.predict_proba(self._scaler.transform(x))[:, 1]
        return pd.DataFrame(
            {
                "IdCliente": feats["IdCliente"].to_numpy(),
                "ProbRespuesta": p.round(4),
                "DecilPropension": pd.qcut(
                    pd.Series(p).rank(method="first"), 10, labels=range(1, 11)
                ).astype(int),
            }
        )

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> ModeloRespuesta:
        return joblib.load(path)


@dataclass
class ModeloRespuestaNBO:
    """V2: estima respuesta y valor para cada par cliente-recompensa."""

    features: list[str] = None
    _scaler: StandardScaler = None
    _model: CalibratedClassifierCV = None
    _value_model: GradientBoostingRegressor = None
    _medianas: pd.Series = None
    metrics_: dict = None

    def __post_init__(self):
        self.features = self.features or FEATURES_RESPUESTA

    def fit(self, historico: pd.DataFrame) -> ModeloRespuestaNBO:
        _validar_historico_simulado(historico)
        faltantes = sorted(set(self.features).difference(historico.columns))
        if faltantes:
            raise ValueError(f"Faltan variables para entrenar NBO: {faltantes}")

        campanas = sorted(historico["IdCampana"].unique())
        if len(campanas) < 4:
            raise ValueError("NBO requiere al menos cuatro campañas para corte temporal")
        corte = max(1, int(len(campanas) * 0.75))
        train_ids, test_ids = campanas[:corte], campanas[corte:]
        train = historico[historico["IdCampana"].isin(train_ids)].copy()
        test = historico[historico["IdCampana"].isin(test_ids)].copy()

        self._medianas = train[self.features].apply(
            pd.to_numeric, errors="coerce"
        ).median(numeric_only=True)
        x_train = self._matriz(train)
        x_test = self._matriz(test)
        y_train = train["Respondio"].astype(int)
        y_test = test["Respondio"].astype(int)

        self._scaler = StandardScaler().fit(x_train)
        base = GradientBoostingClassifier(random_state=SEMILLA_SIMULACION)
        self._model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self._model.fit(self._scaler.transform(x_train), y_train)

        respondieron = train["Respondio"].eq(1)
        if respondieron.sum() < 10:
            raise ValueError("No hay suficientes respuestas para modelar valor incremental")
        self._value_model = GradientBoostingRegressor(
            random_state=SEMILLA_SIMULACION,
            loss="huber",
        )
        self._value_model.fit(
            self._scaler.transform(x_train.loc[respondieron]),
            train.loc[respondieron, "ValorIncremental"],
        )

        p_test = self._model.predict_proba(self._scaler.transform(x_test))[:, 1]
        respondieron_test = test["Respondio"].eq(1)
        pred_valor = self._value_model.predict(
            self._scaler.transform(x_test.loc[respondieron_test])
        )
        por_recompensa = {}
        for recompensa, grupo in test.groupby("TipoRecompensa"):
            indices = test.index.get_indexer(grupo.index)
            por_recompensa[recompensa] = _metricas_clasificacion(
                grupo["Respondio"], p_test[indices]
            )

        self.metrics_ = {
            "fuente": "semi_sintetico_desde_features_cliente",
            "campanas_entrenamiento": len(train_ids),
            "campanas_prueba": len(test_ids),
            "filas_entrenamiento": int(len(train)),
            "filas_prueba": int(len(test)),
            "respuesta": _metricas_clasificacion(y_test, p_test),
            "respuesta_por_tipo": por_recompensa,
            "valor_incremental": {
                "mae": round(
                    float(
                        mean_absolute_error(
                            test.loc[respondieron_test, "ValorIncremental"], pred_valor
                        )
                    ),
                    3,
                ),
                "r2": round(
                    float(
                        r2_score(
                            test.loc[respondieron_test, "ValorIncremental"], pred_valor
                        )
                    ),
                    3,
                ),
            },
        }
        return self

    def _matriz(self, datos: pd.DataFrame) -> pd.DataFrame:
        x = datos[self.features].apply(pd.to_numeric, errors="coerce")
        x = x.fillna(self._medianas).fillna(0.0)
        for tipo in TIPOS_CAMPANA:
            x[f"Recompensa_{tipo}"] = datos["TipoRecompensa"].eq(tipo).astype(int)
        return x

    def predict_opciones(self, feats: pd.DataFrame) -> pd.DataFrame:
        opciones = []
        for recompensa in RECOMPENSAS_NBO:
            candidatos = feats.copy()
            candidatos["TipoRecompensa"] = recompensa
            x = self._matriz(candidatos)
            x_scaled = self._scaler.transform(x)
            prob = self._model.predict_proba(x_scaled)[:, 1]
            valor = np.maximum(self._value_model.predict(x_scaled), 0.0)
            costo = float(REWARDS.costo[recompensa])
            opciones.append(
                pd.DataFrame(
                    {
                        "IdCliente": candidatos["IdCliente"].to_numpy(),
                        "Recompensa": recompensa,
                        "ProbRespuesta": prob,
                        "ValorIncremental": valor,
                        "Costo": costo,
                        "ValorEsperado": prob * valor - costo,
                    }
                )
            )
        resultado = pd.concat(opciones, ignore_index=True)
        columnas = ["ProbRespuesta", "ValorIncremental", "Costo", "ValorEsperado"]
        resultado[columnas] = resultado[columnas].round(4)
        return resultado

    def predict_wide(self, feats: pd.DataFrame) -> pd.DataFrame:
        opciones = self.predict_opciones(feats)
        wide = opciones.pivot(
            index="IdCliente",
            columns="Recompensa",
            values=["ProbRespuesta", "ValorIncremental", "Costo", "ValorEsperado"],
        )
        wide.columns = [f"{metrica}_{recompensa}" for metrica, recompensa in wide.columns]
        return wide.reset_index()

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> ModeloRespuestaNBO:
        return joblib.load(path)


def _metricas_clasificacion(y_true: pd.Series, p: np.ndarray) -> dict:
    metricas = {
        "tasa_positivos": round(float(np.asarray(y_true).mean()), 3),
        "pr_auc": round(float(average_precision_score(y_true, p)), 3),
        "brier": round(float(brier_score_loss(y_true, p)), 3),
        "lift_top_decil": round(float(_lift_top_decil(y_true, p)), 2),
    }
    metricas["roc_auc"] = (
        round(float(roc_auc_score(y_true, p)), 3)
        if pd.Series(y_true).nunique() > 1
        else None
    )
    return metricas


def _lift_top_decil(y_true: pd.Series, p: np.ndarray) -> float:
    d = pd.DataFrame({"y": np.asarray(y_true), "p": p})
    corte = d["p"].quantile(0.9)
    top = d[d["p"] >= corte]
    base = d["y"].mean()
    return (top["y"].mean() / base) if base > 0 else float("nan")
