"""App de demo: tablero de cartera, ficha por cliente y asistente RAG."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import streamlit as st

from casino_ia import config
from casino_ia.data import cargar_features_cliente
from casino_ia.genai import AsistentePoliticas, explicar_cliente
from casino_ia.models import ModeloRespuesta, ModeloRespuestaNBO, ModeloRiesgo
from casino_ia.optimization.allocate import asignar_recompensas

st.set_page_config(page_title="Palacio Real · Recompensas", layout="wide")


@st.cache_data
def _data():
    return cargar_features_cliente()


@st.cache_resource
def _modelos():
    return (
        ModeloRiesgo.load(config.MODELS_STORE / "modelo_riesgo.joblib"),
        ModeloRespuesta.load(config.MODELS_STORE / "modelo_respuesta.joblib"),
        ModeloRespuestaNBO.load(config.MODELS_STORE / "modelo_respuesta_nbo.joblib"),
    )


feats = _data()
riesgo, respuesta, respuesta_nbo = _modelos()
pred = (
    riesgo.predict(feats)[["IdCliente", "NivelRiesgo", "RiesgoScore"]]
    .merge(respuesta.predict_proba(feats), on="IdCliente")
    .merge(respuesta_nbo.predict_wide(feats), on="IdCliente", validate="one_to_one")
    .merge(feats, on="IdCliente")
)

st.title("Casino Palacio Real — asignación de recompensas")
tab_cartera, tab_cliente, tab_chat = st.tabs(["Cartera", "Cliente", "Asistente"])

with tab_cartera:
    presupuesto = st.slider(
        "Presupuesto de la campaña (S/)",
        200,
        40000,
        int(config.REWARDS.presupuesto),
        200,
    )
    cand = asignar_recompensas(pred, presupuesto=presupuesto)
    if len(cand) and "Asignada" not in cand.columns:  # módulo desactualizado en la sesión
        st.error(
            "Reinicia la app (Ctrl+C y volver a lanzar): "
            "hay una versión vieja del optimizador en memoria."
        )
        st.stop()
    asignadas = cand[cand["Asignada"]] if len(cand) else cand
    candidatas = cand[cand["RecompensaSugerida"].notna()] if len(cand) else cand

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes en cartera", len(pred))
    c2.metric("Riesgo alto (excluidos)", int((pred["NivelRiesgo"] == "Alto").sum()))
    c3.metric("Recompensas asignadas", f"{len(asignadas)} / {len(candidatas)}")
    gasto = asignadas["Costo"].sum() if len(asignadas) else 0
    c4.metric("Gasto / presupuesto", f"{gasto:,.0f} / {presupuesto:,.0f}")

    st.caption(
        "El optimizador rankea a los clientes elegibles por **eficiencia** "
        "(valor esperado por sol) "
        "y evalúa toda la lista aplicando presupuesto, topes y guardrails. "
        "**MotivoDecision** explica el resultado de cada cliente."
    )
    st.bar_chart(pred["NivelRiesgo"].value_counts())
    if len(cand):
        cols = [
            c
            for c in [
                "IdCliente",
                "Segmento",
                "NivelRiesgo",
                "SesionesUltimos90d",
                "ProbRespuesta",
                "Recompensa",
                "Costo",
                "ValorIncremental",
                "ValorEsperado",
                "Eficiencia",
                "GastoAcumulado",
                "Asignada",
                "MotivoDecision",
            ]
            if c in cand.columns
        ]
        st.dataframe(cand[cols], width="stretch", hide_index=True)
    else:
        st.info("Ningún cliente tiene valor esperado positivo con estos parámetros.")

with tab_cliente:
    columnas_decision = [
        "IdCliente",
        "Recompensa",
        "Costo",
        "ProbRespuesta",
        "ValorIncremental",
        "ValorEsperado",
        "Asignada",
        "MotivoDecision",
    ]
    fichas = pred.drop(columns="ProbRespuesta").merge(
        cand[columnas_decision], on="IdCliente", how="left"
    )
    cid = st.selectbox("Cliente", fichas["IdCliente"].tolist())
    ficha = fichas[fichas["IdCliente"] == cid].iloc[0].to_dict()
    campos = (
        "NivelRiesgo",
        "RiesgoScore",
        "ProbRespuesta",
        "Segmento",
        "CoinInTotal",
        "Recompensa",
        "Costo",
        "Asignada",
        "MotivoDecision",
    )
    st.json({k: ficha.get(k) for k in campos})
    textos = explicar_cliente(ficha)
    st.write("**Explicación**", textos["explicacion"])
    st.write("**Oferta**", textos["oferta"])
    st.write("**Mensaje**", textos["mensaje"])

with tab_chat:
    st.caption(
        "Chatbot para el analista. Responde con RAG sobre las políticas y la guía de "
        "asignación de recompensas; si algo no está en los documentos, lo dice."
    )

    @st.cache_resource
    def _asistente():
        return AsistentePoliticas()

    asistente = _asistente()
    ejemplos = [
        "¿Un cliente de riesgo medio puede recibir recompensa alta?",
        "¿Qué pasa con los clientes de riesgo alto?",
        "¿Cómo se elige a quién premiar si el presupuesto no alcanza?",
        "¿Qué incluye la recompensa media?",
    ]
    st.write("Ejemplos: " + " · ".join(f"`{e}`" for e in ejemplos))

    if "chat" not in st.session_state:
        st.session_state.chat = []
    for m in st.session_state.chat:
        st.chat_message(m["role"]).write(m["content"])

    pregunta = st.chat_input("Escribe tu pregunta")
    if pregunta:
        st.session_state.chat.append({"role": "user", "content": pregunta})
        st.chat_message("user").write(pregunta)
        r = asistente.responder(pregunta)
        texto = r["respuesta"]
        if r["fuentes"]:
            texto += "\n\n_Fuentes: " + ", ".join(sorted(set(r["fuentes"]))) + "_"
        st.session_state.chat.append({"role": "assistant", "content": texto})
        st.chat_message("assistant").write(texto)
