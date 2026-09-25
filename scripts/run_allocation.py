"""Genera el plan de asignación de recompensas y lo compara con el baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from casino_ia import config
from casino_ia.data import cargar_features_cliente
from casino_ia.models import ModeloRespuesta, ModeloRespuestaNBO, ModeloRiesgo
from casino_ia.optimization.allocate import asignar_recompensas, baseline_reglas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--presupuesto", type=float, default=config.REWARDS.presupuesto)
    args = ap.parse_args()

    feats = cargar_features_cliente()
    riesgo = ModeloRiesgo.load(config.MODELS_STORE / "modelo_riesgo.joblib")
    respuesta = ModeloRespuesta.load(config.MODELS_STORE / "modelo_respuesta.joblib")
    respuesta_nbo = ModeloRespuestaNBO.load(
        config.MODELS_STORE / "modelo_respuesta_nbo.joblib"
    )

    scoring = (
        feats[["IdCliente", "Segmento", "NroSesiones", "ValorTeoricoCasa", "RatioTendenciaCoinIn"]]
        .merge(riesgo.predict(feats)[["IdCliente", "NivelRiesgo"]], on="IdCliente")
        .merge(respuesta.predict_proba(feats)[["IdCliente", "ProbRespuesta"]], on="IdCliente")
        .merge(respuesta_nbo.predict_wide(feats), on="IdCliente", validate="one_to_one")
    )

    candidatos = asignar_recompensas(scoring, presupuesto=args.presupuesto)
    plan = candidatos[candidatos["Asignada"]]
    total_candidatos = candidatos.attrs.get(
        "candidatos_totales",
        int(candidatos["RecompensaSugerida"].notna().sum()),
    )
    base = baseline_reglas(scoring)
    candidatos.to_csv(config.METRICS / "plan_asignacion.csv", index=False)

    print(f"Presupuesto: {args.presupuesto:,.0f}")
    print(
        f"Optimizador -> candidatos con valor positivo: {total_candidatos}  |  "
        f"asignados: {len(plan)}  gasto: {plan['Costo'].sum():,.0f}  "
        f"valor esperado: {plan['ValorEsperado'].sum():,.0f}"
    )
    print(f"Baseline    -> clientes: {len(base)}  gasto: {base['Costo'].sum():,.0f}  "
          f"valor esperado: {base['ValorEsperado'].sum():,.0f}")
    if base["ValorEsperado"].sum() > 0:
        mejora = plan["ValorEsperado"].sum() / base["ValorEsperado"].sum() - 1
        print(f"Mejora en valor esperado: {mejora:+.1%}")
    print(pd.Series(plan["Recompensa"]).value_counts().rename("recompensas asignadas"))


if __name__ == "__main__":
    main()
