"""Pruebas del optimizador de asignación de recompensas."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from casino_ia.genai.explainer import _plantilla, explicar_cliente
from casino_ia.optimization.allocate import asignar_recompensas


def _scoring():
    return pd.DataFrame(
        {
            "IdCliente": range(1, 11),
            "Segmento": ["VIP", "Alto", "Medio", "Estandar"] * 2 + ["VIP", "Medio"],
            "NivelRiesgo": [
                "Bajo",
                "Bajo",
                "Medio",
                "Alto",
                "Bajo",
                "Alto",
                "Medio",
                "Bajo",
                "Bajo",
                "Medio",
            ],
            "ProbRespuesta": [0.9, 0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.85, 0.3, 0.55],
            "ValorTeoricoCasa": [5000, 4000, 3000, 9000, 2500, 8000, 1500, 6000, 1000, 2000],
            "NroSesiones": [50] * 10,
            "RatioTendenciaCoinIn": [1.1] * 10,
        }
    )


def test_excluye_riesgo_alto():
    cand = asignar_recompensas(_scoring(), presupuesto=10_000)
    assert not cand.empty
    altos = cand[cand["NivelRiesgo"] == "Alto"]
    assert not altos["Asignada"].any()
    assert altos["Recompensa"].isna().all()
    assert altos["MotivoDecision"].str.contains("riesgo alto").all()


def test_respeta_presupuesto():
    cand = asignar_recompensas(_scoring(), presupuesto=30)
    asignadas = cand[cand["Asignada"]]
    assert asignadas["Costo"].sum() <= 30


def test_riesgo_medio_solo_recompensa_baja():
    cand = asignar_recompensas(_scoring(), presupuesto=10_000)
    medios = cand[(cand["NivelRiesgo"] == "Medio") & cand["Asignada"]]
    assert not medios.empty
    assert (medios["Recompensa"] == "baja").all()


def test_un_registro_por_cliente():
    cand = asignar_recompensas(_scoring(), presupuesto=10_000)
    assert cand["IdCliente"].is_unique


def test_minimo_tres_sesiones_en_90_dias():
    scoring = _scoring()
    scoring.loc[scoring["IdCliente"] == 1, "SesionesUltimos90d"] = 2
    cand = asignar_recompensas(scoring, presupuesto=10_000)
    decision = cand[cand["IdCliente"] == 1].iloc[0]
    assert not decision["Asignada"]
    assert pd.isna(decision["Recompensa"])
    assert "menos de 3 sesiones" in decision["MotivoDecision"]


def test_solo_asigna_valor_esperado_positivo_y_deja_motivo():
    cand = asignar_recompensas(_scoring(), presupuesto=10_000)
    asignadas = cand[cand["Asignada"]]
    assert (asignadas["ValorEsperado"] > 0).all()
    assert cand["MotivoDecision"].str.len().gt(0).all()


def test_tope_25_por_ciento_en_recompensas_altas():
    scoring = pd.DataFrame(
        {
            "IdCliente": range(1, 7),
            "Segmento": [f"S{i}" for i in range(1, 7)],
            "NivelRiesgo": ["Bajo"] * 6,
            "ProbRespuesta": [1.0] * 6,
            "ValorTeoricoCasa": [10_000] * 6,
            "NroSesiones": [3] * 6,
            "RatioTendenciaCoinIn": [1.0] * 6,
        }
    )
    cand = asignar_recompensas(
        scoring,
        presupuesto=100,
        costos={"alta": 10, "media": 100, "baja": 100},
        max_fraccion_segmento=1.0,
    )
    altas = cand[cand["Asignada"] & (cand["Recompensa"] == "alta")]
    assert altas["Costo"].sum() <= 25
    assert cand["MotivoDecision"].str.contains("tope 25%").any()


def test_tope_40_por_ciento_del_presupuesto_por_segmento():
    scoring = pd.DataFrame(
        {
            "IdCliente": range(1, 11),
            "Segmento": ["VIP"] * 10,
            "NivelRiesgo": ["Bajo"] * 10,
            "ProbRespuesta": [1.0] * 10,
            "ValorTeoricoCasa": [10_000] * 10,
            "NroSesiones": [3] * 10,
            "RatioTendenciaCoinIn": [1.0] * 10,
        }
    )
    cand = asignar_recompensas(
        scoring,
        presupuesto=50,
        costos={"alta": 100, "media": 100, "baja": 5},
        max_fraccion_altas=1.0,
    )
    asignadas = cand[cand["Asignada"]]
    assert asignadas.groupby("Segmento")["Costo"].sum().max() <= 20
    assert cand["MotivoDecision"].str.contains("tope 40%").any()


def test_continua_despues_de_candidato_que_no_entra_por_presupuesto():
    scoring = pd.DataFrame(
        {
            "IdCliente": [1, 2, 3],
            "Segmento": ["A", "B", "C"],
            "NivelRiesgo": ["Medio", "Medio", "Bajo"],
            "ProbRespuesta": [1.0, 1.0, 1.0],
            "ValorTeoricoCasa": [10_000, 9_000, 500],
            "NroSesiones": [3, 3, 3],
            "RatioTendenciaCoinIn": [1.0, 1.0, 1.0],
        }
    )
    cand = asignar_recompensas(
        scoring,
        presupuesto=25,
        costos={"alta": 100, "media": 5, "baja": 20},
        max_fraccion_altas=1.0,
        max_fraccion_segmento=1.0,
    ).set_index("IdCliente")
    assert cand.at[1, "Asignada"]
    assert not cand.at[2, "Asignada"]
    assert "presupuesto insuficiente" in cand.at[2, "MotivoDecision"]
    assert cand.at[3, "Asignada"]


def test_genai_no_inventa_oferta_sin_asignacion():
    ficha = {
        "IdCliente": 1,
        "NivelRiesgo": "Bajo",
        "ProbRespuesta": 0.8,
        "Asignada": False,
        "Recompensa": None,
        "MotivoDecision": "No asignada: presupuesto insuficiente",
    }
    texto = explicar_cliente(ficha)
    assert texto["oferta"] == "No aplica. No hay recompensa asignada."
    assert "No generar" in texto["mensaje"]


def test_plantilla_usa_recompensa_real():
    ficha = {
        "IdCliente": 1,
        "NivelRiesgo": "Bajo",
        "ProbRespuesta": 0.8,
        "Asignada": True,
        "Recompensa": "baja",
    }
    texto = _plantilla(ficha)
    assert "Recompensa baja" in texto["oferta"]
    assert "puntos extra" in texto["oferta"]


def test_v2_selecciona_recompensa_con_mayor_valor_esperado():
    scoring = pd.DataFrame(
        {
            "IdCliente": [1],
            "Segmento": ["VIP"],
            "NivelRiesgo": ["Bajo"],
            "ProbRespuesta": [0.5],
            "ProbRespuesta_baja": [0.4],
            "ProbRespuesta_media": [0.8],
            "ProbRespuesta_alta": [0.9],
            "ValorIncremental_baja": [30.0],
            "ValorIncremental_media": [100.0],
            "ValorIncremental_alta": [50.0],
            "ValorTeoricoCasa": [1_000],
            "NroSesiones": [10],
            "RatioTendenciaCoinIn": [1.0],
        }
    )
    decision = asignar_recompensas(
        scoring,
        presupuesto=100,
        max_fraccion_altas=1.0,
        max_fraccion_segmento=1.0,
    ).iloc[0]
    assert decision["Recompensa"] == "media"
    assert decision["ProbRespuesta"] == 0.8
    assert decision["ValorIncremental"] == 100.0
    assert decision["ValorEsperado"] == 65.0


def test_v2_riesgo_medio_ignora_opciones_media_y_alta():
    scoring = pd.DataFrame(
        {
            "IdCliente": [1],
            "Segmento": ["Medio"],
            "NivelRiesgo": ["Medio"],
            "ProbRespuesta": [0.5],
            "ProbRespuesta_baja": [0.8],
            "ProbRespuesta_media": [1.0],
            "ProbRespuesta_alta": [1.0],
            "ValorIncremental_baja": [50.0],
            "ValorIncremental_media": [10_000.0],
            "ValorIncremental_alta": [10_000.0],
            "ValorTeoricoCasa": [1_000],
            "NroSesiones": [10],
            "RatioTendenciaCoinIn": [1.0],
        }
    )
    decision = asignar_recompensas(
        scoring,
        presupuesto=100,
        max_fraccion_segmento=1.0,
    ).iloc[0]
    assert decision["Recompensa"] == "baja"
    assert decision["ValorEsperado"] == 35.0
