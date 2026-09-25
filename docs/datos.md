# Datos

## Fuente

`data/raw/playersession_ficticio_100k.csv` — muestra **simulada** de la tabla de
hechos `FctPlayerSession` de un data warehouse de casino.

- 100 000 sesiones de juego
- 888 clientes · 740 máquinas · 2 salas
- Periodo: 15–29 de julio de 2026 (15 días)
- 45 columnas · sin datos personales reales

## Columnas relevantes de `FctPlayerSession`

| Columna | Tipo | Descripción |
|---|---|---|
| `TransID` | bigint | ID único de la sesión |
| `IdCliente` | int | Cliente (jugador) |
| `Mnum` | int | Máquina de juego |
| `IdSala` | int | Sala |
| `StartTime` / `EndTime` | datetime | Inicio / fin de la sesión |
| `TimePlayed` | int | Segundos jugados |
| `Hora` | int | Hora de inicio (0–23) |
| `CoinIn` / `CoinOut` | decimal | Dinero apostado / devuelto por la máquina |
| `Games` | int | Nº de partidas |
| `AverageBet` | decimal | Apuesta promedio |
| `Win` | decimal | Resultado del jugador (negativo = pérdida) |
| `TheoWin` | decimal | Ganancia teórica de la casa (valor del cliente) |
| `CompEarned` | decimal | *Comps* (beneficios) ganados |
| `PointsEarned` | int | Puntos de fidelidad ganados |
| `TripNumber` | int | Nº de visita |
| `IdEmpresa`, `IdMoneda`, `IdCalendario`, `IdNegocio`, `SessionTypeID` | int | Claves de dimensión |

## Tabla analítica por cliente — `vw_FeaturesCliente`

Una fila por cliente (888). Se construye en SQL (`sql/03_features_cliente_scoring.sql`)
y se replica en pandas como *fallback* (`src/casino_ia/features/build.py`).

| Grupo | Variables | Usa |
|---|---|---|
| Identificación | `IdCliente`, `Segmento`, `Ciudad`, `FechaAlta`, `AntiguedadDias` | — |
| Actividad / frecuencia | `NroSesiones`, `DiasActivos`, `NroVisitas`, `DiasDesdeUltimaSesion` | riesgo + respuesta |
| Valor / monetario | `CoinInTotal`, `CoinInPromedioSesion`, `ValorTeoricoCasa`, `ResultadoClienteTotal`, `PerdidasAcumuladas`, `CompsAcumulados`, `PuntosAcumulados` | riesgo + respuesta |
| Intensidad de juego | `HorasJugadas`, `DuracionPromedioMin`, `DuracionMaximaMin`, `CoinInPorHora`, `ApuestaMediaPromedio`, `ApuestaMaxima` | riesgo |
| Patrones de riesgo | `PctSesionesLargas`, `PctJuegoMadrugada`, `PctSesionesChasing`, `VolatilidadResultado` | riesgo |
| Tendencia reciente | `CoinIn_Ultimos7d`, `CoinIn_Previo`, `RatioTendenciaCoinIn`, `Sesiones_Ultimos7d`, `Sesiones_Previo` | respuesta |
| Estructura | `NroSalas` | riesgo |

## Etiquetas

No hay etiquetas reales de "respondió a la campaña" ni de "juego problemático".

- **Prototipo:** *proxies* por percentiles / reglas (`vw_ClientesScoring`) y
  métodos no supervisados.
- **Producción:** histórico de campañas con grupo de control (respuesta / uplift)
  y marcas del área de juego responsable (riesgo).

### Histórico semi-sintético para V2 NBO

Como la fuente no contiene campañas observadas, V2 genera
`data/processed/historico_campanas_simulado.parquet` de forma determinística
(semilla 42). Conserva los 888 perfiles y sus valores reales dentro de la muestra;
solo simula exposición, tipo de recompensa y respuesta en ocho campañas. Incluye
grupo `control`, fecha de oferta/respuesta, costo, valor previo/posterior y valor
incremental. Riesgo alto queda excluido y riesgo medio solo recibe `control` o
recompensa `baja`.

Este conjunto es **semi-sintético y no constituye evidencia histórica de campañas**.
Sirve únicamente para validar el flujo `P(respuesta | cliente, recompensa)` hasta
que exista un registro real con exposición, resultado y ventana de atribución.

## Limitaciones de la muestra simulada

- Distribución uniforme en el tiempo: todos los clientes juegan casi todos los días
  ⇒ `DiasActivos`, `DiasDesdeUltimaSesion` y `TripNumber` tienen poca varianza.
- `CoinIn` con cola muy pesada por sesiones VIP puntuales.
- Las señales de riesgo se calibran en términos **relativos** (percentiles), no
  absolutos, para que el prototipo produzca una segmentación útil.
