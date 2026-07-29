# Diseño — Maduración del pipeline MLOps

**Fecha:** 2026-07-28
**Estado:** Aprobado (pendiente de plan de implementación)

## Objetivo

Evolucionar el pipeline actual (un workflow monolítico `ingest → train → predict`
que corre a diario) hacia una arquitectura MLOps madura con:

1. Separación de responsabilidades en **entrenamiento** e **inferencia**.
2. Un **registro de modelos** con concepto de "modelo de producción".
3. Un **gate de promoción** que solo asciende un modelo nuevo si mejora al actual,
   con aprobación humana (human-in-the-loop).

## Decisiones acordadas

| Decisión | Elección |
|---|---|
| Registro de modelos | **DVC/GCS, sin W&B** (autocontenido, sin secrets) |
| Cadencia | **Inferencia diaria**, **entrenamiento semanal** |
| Promoción | **Manual** (human-in-the-loop) |
| Orquestación | **Enfoque A**: DVC versiona artefactos; scripts en `src/` orquestados por workflows |

## Arquitectura

Se pasa de un workflow a **tres**, más un registro en DVC/GCS.

```
training.yml (semanal + manual)
    ingest → train (candidato) → evaluate (candidato vs producción)
                                      → metrics/comparison.json (reporte)

promote.yml (solo manual)
    promote: candidato → producción

inference.yml (diario + manual)
    ingest → predict (con modelo de producción)
```

### Registro de modelos (DVC/GCS)

| Ruta | Qué es | Quién la escribe |
|---|---|---|
| `models/candidate/lgbm_model.txt` | Último modelo entrenado (no aprobado) | `training.yml` |
| `models/production/lgbm_model.txt` | Modelo en producción (aprobado) | `promote.yml` |
| `metrics/candidate.json` | RMSE y metadata del candidato | `training.yml` |
| `metrics/production.json` | RMSE de referencia de producción | `promote.yml` |
| `metrics/comparison.json` | Candidato vs producción + recomendación | `training.yml` |

Los binarios de modelo se versionan con DVC (cache en GCS); los JSON de métricas
se versionan en git y son legibles con `dvc metrics show` / `dvc metrics diff`.

### Flujo típico

1. Semanalmente corre `training.yml`: entrena un candidato, lo compara contra
   producción en la **misma ventana de validación**, y publica un reporte.
2. El usuario revisa el reporte (artefacto del run + log legible).
3. Si aprueba, dispara `promote.yml` manualmente → el candidato pasa a producción.
4. La inferencia diaria (`inference.yml`) siempre usa `models/production/`, sin
   verse afectada por candidatos no aprobados.

## Componentes (módulos en `src/`)

| Módulo | Estado | Responsabilidad |
|---|---|---|
| `features.py` | existe | Fuente compartida de features (sin cambios) |
| `ingest_xm.py` | existe | Descarga y agrega datos de XM (sin cambios) |
| `train.py` | modificar | Entrena candidato → `models/candidate/` + `metrics/candidate.json`. Se le quita todo W&B. |
| `evaluate.py` | **nuevo** | Compara candidato vs producción en la misma ventana → `metrics/comparison.json` |
| `promote.py` | **nuevo** | Copia candidato→producción y actualiza `metrics/production.json` |
| `predict.py` | modificar | Carga desde `models/production/` (hoy `models/lgbm_model.txt`) |
| `lineage.py` | reutilizar | Se quita `log_lineage_to_wandb`; se conservan `get_git_commit`/`get_dvc_hash` para estampar trazabilidad en los JSON de métricas |

### Contratos (una responsabilidad por módulo)

- `train.py`: datos + params → modelo candidato + `candidate.json`. No conoce producción.
- `evaluate.py`: candidato + producción + datos → `comparison.json`. No entrena ni promueve.
- `promote.py`: candidato → producción. No evalúa.
- `predict.py`: modelo de producción + datos → predicción. No conoce candidatos.

### Retiro de W&B

- Eliminar `wandb` de `requirements.txt`.
- Quitar de `train.py`: `wandb.init`, logging de config/métricas y registro del artefacto.
- La trazabilidad (RMSE, git_sha, hiperparámetros, hash de datos) pasa a los JSON
  de métricas versionados.

## Metodología del gate (comparación justa)

El `metrics/production.json` guarda el RMSE de producción medido en *su* ventana de
validación histórica. Comparar contra el RMSE del candidato en una ventana nueva
sería injusto. Por eso `evaluate.py` **re-evalúa ambos modelos en la misma ventana
de validación actual**:

1. Ingesta datos frescos, construye features, split cronológico (últimos `validation_days`).
2. Candidato → predice sobre validación → `rmse_candidate`.
3. Producción → predice sobre **la misma** validación → `rmse_production`.
4. Recomendación: `PROMOTE` si `rmse_candidate < rmse_production * (1 - margin)`,
   de lo contrario `KEEP`.

El `margin` (mejora mínima relativa, ej. 1%) se configura en `params.yaml` bajo
`promotion.margin`, para evitar promover por ruido estadístico.

### Ejemplo de `metrics/comparison.json`

```json
{
  "rmse_candidate": 85.1,
  "rmse_production": 87.5,
  "delta_pct": -2.74,
  "margin_pct": 1.0,
  "recommendation": "PROMOTE",
  "evaluated_at": "2026-08-03",
  "git_sha": "abc123"
}
```

`training.yml` sube este archivo como artefacto del run y lo imprime legible en el log.

## Bootstrap (primera vez, sin modelo de producción)

- `evaluate.py`: si no existe `models/production/`, `rmse_production = null` y
  recomendación = `PROMOTE (bootstrap)`.
- `predict.py`: si no existe producción, **falla ruidosamente** (ya lo hace). La
  inferencia diaria no corre hasta que se promueva el primer modelo.
- Secuencia inicial: `training.yml` → `promote.yml` (una vez) → producción sembrada
  → arranca la inferencia diaria. Documentado en README.

## Manejo de errores

- `evaluate.py` y `promote.py` fallan con código ≠ 0 ante entradas faltantes
  (candidato inexistente, datos vacíos).
- `evaluate.py` **no** falla el build si el candidato es peor — es informativo
  (`recommendation: KEEP`).
- `promote.py` valida que exista un candidato antes de sobrescribir producción.

## Testing

Reutiliza la infraestructura de pytest existente, con foco en la lógica del gate.

| Archivo | Qué cubre |
|---|---|
| `tests/test_evaluate.py` | Lógica de recomendación: mejor por más del margen → `PROMOTE`; mejor dentro del margen → `KEEP`; peor → `KEEP`; sin producción → `PROMOTE (bootstrap)` |
| `tests/test_promote.py` | `promote` copia candidato→producción y actualiza `production.json`; falla sin candidato |
| `tests/test_params.py` | Ampliar: existe `promotion.margin` con `0 < margin < 1` |

Diseño para testeabilidad: la decisión del gate se aísla en una función pura:

```python
def decide(rmse_candidate, rmse_production, margin) -> str:
    # "PROMOTE" | "KEEP"  (bootstrap si rmse_production is None)
```

Los tests cubren la lógica sin entrenar modelos ni tocar disco (I/O se prueba con
`tmp_path`). No se harán tests end-to-end que entrenen LightGBM real; la confianza
viene de tests unitarios + la corrida real de CI.

## Cambios en configuración

- `params.yaml`: agregar sección `promotion` con `margin: 0.01`.
- `requirements.txt`: eliminar `wandb`.
- Workflows: reemplazar `daily_pipeline.yml` por `inference.yml`, `training.yml`,
  `promote.yml`.

## Fuera de alcance (para más adelante)

- Validación de datos en `ingest` (rangos, nulos, continuidad de fechas).
- Features enriquecidas (calendario, hidrología) y baseline ingenuo.
- Reentrenamiento disparado por drift de datos.
