# Base de datos — `CasinoPalacioReal`

Ejecutar en **SQL Server 2022+** (SSMS), en orden:

| # | Script | Crea |
|---|---|---|
| 1 | `01_cargar_FctPlayerSession.sql` | BD `CasinoPalacioReal` + tabla `dbo.FctPlayerSession` y carga del CSV |
| 2 | `02_crear_dimensiones_y_vista.sql` | Modelo estrella (`DimCliente`, `DimMaquina`, `DimSala`, `DimEmpresa`, `DimUbicacion`, `DimMoneda`, `DimNegocio`, `DimTipoSesion`, `DimCalendario`) + claves foráneas + vistas `vw_SesionesDetalle`, `vw_ResumenCliente`, `vw_ResumenMaquina`, `vw_ResumenDiario` |
| 3 | `03_features_cliente_scoring.sql` | `vw_FeaturesCliente` (tabla analítica, 1 fila por cliente) y `vw_ClientesScoring` (baseline por reglas: `NivelRiesgo`, `DecilPropension`, `AccionRecomendada`) |
| 4 | `04_preparar_abt_riesgo_supervisado.sql` | ABT temporal supervisada de riesgo y controles de calidad |
| 5 | `05_modelo_riesgo_supervisado.sql` | Corridas, predicciones y vista vigente del modelo de riesgo |
| 6 | `06_modelo_respuesta_nbo.sql` | Evidencia, campañas calibradas y opciones cliente-recompensa |
| 7 | `07_optimizador_recompensas.sql` | Corridas, decisiones trazables y vistas vigentes del optimizador |

## Antes de empezar

1. Copiar `../data/raw/playersession_ficticio_100k.csv` a `C:\Data\` (el servicio
   de SQL Server no lee dentro de OneDrive).
2. Verificar la versión: `SELECT @@VERSION;` — debe ser 16.x o superior.

## Compartir la base ya cargada

Se distribuye un backup `CasinoPalacioReal.bak` por Google Drive / OneDrive
(no se versiona en git). Instrucciones de restauración en
`../entregables/LEEME_restaurar_base.md`.

## Notas de los datos

- El CSV es UTF-8 con saltos de línea LF ⇒ `BULK INSERT ... ROWTERMINATOR='0x0a'`.
- `ManualEdit` llega como texto `'True'/'False'` ⇒ se carga como `VARCHAR(5)`.
- `PointMultiplier` viene siempre vacío ⇒ `NULL`.
- `TripNumber` y algunas columnas de puntos son constantes en la muestra simulada.
- Los atributos de `DimCliente` / `DimMaquina` (nombres, ciudades, fabricantes)
  son ficticios pero **deterministas**: re-ejecutar da el mismo resultado.
