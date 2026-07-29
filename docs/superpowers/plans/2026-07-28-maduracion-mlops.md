# Maduración MLOps — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolucionar el pipeline monolítico hacia tres workflows (inference/training/promote) con un registro de modelos en DVC/GCS y un gate de promoción manual.

**Architecture:** Enfoque A — DVC versiona artefactos (datos y modelos); la lógica vive en módulos `src/` con una responsabilidad cada uno, orquestados por workflows de GitHub Actions. El "modelo de producción" es un artefacto en `models/production/` promovido manualmente; el gate compara candidato vs producción reevaluando ambos en la misma ventana de validación.

**Tech Stack:** Python 3.11, LightGBM, pandas, DVC (`dvc[gs]`) + GCS, GitHub Actions, pytest.

## Global Constraints

- **Python de producción: 3.11**; el venv local de tests es **3.9** → NO usar sintaxis PEP 604 (`X | Y`) en firmas evaluadas en runtime; usar `from __future__ import annotations` cuando se necesite.
- **Sin W&B** — retirar por completo; trazabilidad en JSON versionados.
- Los scripts fallan con **código ≠ 0** ante entradas faltantes (no silenciar errores).
- Fuente única de features: `src/features.py`. Fuente única de hiperparámetros: `params.yaml`.
- Modelos (binarios) → DVC/GCS; métricas (`metrics/*.json`) → git.
- Verificación local de tests en el venv de `C:\Users\carlo\AppData\Local\Temp\testvenv` (pandas 2.x / pytest); workflows se validan con una corrida real de CI vía `workflow_dispatch`.
- Commits sin trailer de co-autor (config `includeCoAuthoredBy: false` ya activa).

## File Structure

- `params.yaml` — MODIFICAR: nueva sección `promotion.margin`.
- `src/features.py` — MODIFICAR: agregar `make_supervised`, `chronological_split`.
- `src/registry.py` — CREAR: constantes de rutas + helpers JSON (fuente única de rutas del registro).
- `src/evaluate.py` — CREAR: `decide`, `build_comparison`, `evaluate_models`.
- `src/promote.py` — CREAR: `promote`.
- `src/train.py` — MODIFICAR: quitar W&B; guardar candidato; escribir `candidate.json`.
- `src/predict.py` — MODIFICAR: cargar desde `models/production/`.
- `src/lineage.py` — MODIFICAR: quitar `log_lineage_to_wandb`.
- `requirements.txt` — MODIFICAR: quitar `wandb`.
- `dvc.yaml` — MODIFICAR: `train` produce candidato; quitar stage `predict`.
- `.gitignore` — MODIFICAR: permitir punteros `.dvc` de modelos.
- `.github/workflows/training.yml` — CREAR.
- `.github/workflows/promote.yml` — CREAR.
- `.github/workflows/inference.yml` — CREAR (reemplaza `daily_pipeline.yml`).
- `.github/workflows/daily_pipeline.yml` — ELIMINAR.
- `tests/test_params.py`, `tests/test_features.py` — MODIFICAR.
- `tests/test_evaluate.py`, `tests/test_promote.py`, `tests/test_registry.py` — CREAR.
- `README.md` — MODIFICAR.

**Nota de rutas (Windows):** el binario de python del venv de tests es
`C:\Users\carlo\AppData\Local\Temp\testvenv\Scripts\python.exe`. En los comandos
`pytest` de abajo se asume que se invoca con ese intérprete desde la raíz del repo.

---

### Task 1: Config — `promotion.margin` en params.yaml

**Files:**
- Modify: `params.yaml`
- Test: `tests/test_params.py`

**Interfaces:**
- Produces: `params.yaml` con clave `promotion.margin` (float, 0<margin<1).

- [ ] **Step 1: Escribir el test que falla**

Agregar a `tests/test_params.py`:

```python
def test_promotion_margin_present_and_sane():
    p = load()
    assert "promotion" in p, "Falta la sección 'promotion' en params.yaml"
    margin = p["promotion"]["margin"]
    assert 0 < margin < 1, "promotion.margin debe estar entre 0 y 1"
```

- [ ] **Step 2: Correr el test y verificar que falla**

Run: `pytest tests/test_params.py::test_promotion_margin_present_and_sane -v`
Expected: FAIL (KeyError 'promotion').

- [ ] **Step 3: Agregar la sección a params.yaml**

Al final de `params.yaml`:

```yaml
promotion:
  margin: 0.01            # mejora relativa mínima de RMSE para recomendar PROMOTE (1%)
```

- [ ] **Step 4: Correr el test y verificar que pasa**

Run: `pytest tests/test_params.py -v`
Expected: PASS (todos).

- [ ] **Step 5: Commit**

```bash
git add params.yaml tests/test_params.py
git commit -m "feat: agregar promotion.margin a params.yaml"
```

---

### Task 2: Helpers de dataset compartidos en features.py

**Files:**
- Modify: `src/features.py`
- Modify: `src/train.py` (usar los nuevos helpers)
- Test: `tests/test_features.py`

**Interfaces:**
- Produces:
  - `make_supervised(df, lookback) -> (X: DataFrame, y: Series)` — aplica `build_lag_features`, `dropna`, y separa X (columnas de lag) e y (`precio_target`).
  - `chronological_split(X, y, validation_days) -> (X_train, X_val, y_train, y_val)` — últimos `validation_days` como validación.
- Consumes: `build_lag_features`, `feature_columns` (ya existen en features.py).

- [ ] **Step 1: Escribir los tests que fallan**

Agregar a `tests/test_features.py`:

```python
from src.features import make_supervised, chronological_split


def test_make_supervised_shapes_and_columns():
    prices = _sample_prices()  # 30 valores
    df = pd.DataFrame({"PrecioPromedio": prices})
    X, y = make_supervised(df, LOOKBACK)
    # dropna quita las primeras (LOOKBACK-1) filas y la última (target NaN)
    assert list(X.columns) == feature_columns(LOOKBACK)
    assert len(X) == len(y) == len(prices) - (LOOKBACK - 1) - 1
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_chronological_split_sizes_and_order():
    prices = _sample_prices()
    df = pd.DataFrame({"PrecioPromedio": prices})
    X, y = make_supervised(df, LOOKBACK)
    X_tr, X_val, y_tr, y_val = chronological_split(X, y, validation_days=5)
    assert len(X_val) == 5 and len(y_val) == 5
    assert len(X_tr) == len(X) - 5
    # el split respeta el orden temporal (validación es el tramo final)
    assert X_val.index[0] == X.index[len(X) - 5]
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_features.py -v`
Expected: FAIL (ImportError: make_supervised).

- [ ] **Step 3: Implementar en features.py**

Agregar al final de `src/features.py`:

```python
def make_supervised(df, lookback):
    """DataFrame de precios -> (X, y) listos para entrenar/evaluar.

    Aplica los lags, elimina NaN y separa features y objetivo.
    """
    df = build_lag_features(df, lookback).dropna()
    X = df[feature_columns(lookback)]
    y = df["precio_target"]
    return X, y


def chronological_split(X, y, validation_days):
    """Separa (X, y) dejando los últimos `validation_days` como validación."""
    split = len(X) - validation_days
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]
```

- [ ] **Step 4: Refactorizar train.py para usarlos**

En `src/train.py`, reemplazar el bloque que hoy hace `build_lag_features` + `dropna` + arma `X`/`y` + split manual por:

```python
    lookback = p["lookback_days"]
    X, y = make_supervised(df, lookback)
    X_train, X_val, y_train, y_val = chronological_split(X, y, p["validation_days"])

    print(f"🎯 Entrenamiento: {len(X_train)} muestras")
    print(f"🎯 Validación: {len(X_val)} muestras")
```

Y agregar el import al inicio de `train.py`:

```python
from src.features import make_supervised, chronological_split
```

(quitar el import previo `from src.features import build_lag_features, feature_columns` si ya no se usan directamente en train.py.)

- [ ] **Step 5: Correr los tests y verificar que pasan**

Run: `pytest tests/test_features.py -v`
Expected: PASS (incluyendo los existentes de alineamiento).

- [ ] **Step 6: Verificar sintaxis de train.py**

Run: `python -m py_compile src/train.py`
Expected: sin salida (OK).

- [ ] **Step 7: Commit**

```bash
git add src/features.py src/train.py tests/test_features.py
git commit -m "refactor: helpers make_supervised/chronological_split compartidos"
```

---

### Task 3: registry.py — rutas del registro y helpers JSON

**Files:**
- Create: `src/registry.py`
- Test: `tests/test_registry.py`

**Interfaces:**
- Produces:
  - Constantes `Path`: `CANDIDATE_MODEL`, `PRODUCTION_MODEL`, `CANDIDATE_METRICS`, `PRODUCTION_METRICS`, `COMPARISON_METRICS`.
  - `write_json(path, data: dict) -> None` — crea directorios padre; escribe con indent=2.
  - `read_json(path) -> dict`.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_registry.py`:

```python
from pathlib import Path

from src import registry


def test_path_constants():
    assert registry.CANDIDATE_MODEL == Path("models/candidate/lgbm_model.txt")
    assert registry.PRODUCTION_MODEL == Path("models/production/lgbm_model.txt")
    assert registry.CANDIDATE_METRICS == Path("metrics/candidate.json")
    assert registry.PRODUCTION_METRICS == Path("metrics/production.json")
    assert registry.COMPARISON_METRICS == Path("metrics/comparison.json")


def test_write_read_json_roundtrip(tmp_path):
    p = tmp_path / "sub" / "m.json"          # el subdir no existe aún
    data = {"rmse": 12.5, "recommendation": "PROMOTE"}
    registry.write_json(p, data)
    assert p.exists()
    assert registry.read_json(p) == data
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL (ModuleNotFoundError: src.registry).

- [ ] **Step 3: Implementar registry.py**

Crear `src/registry.py`:

```python
"""Rutas del registro de modelos y helpers de lectura/escritura de métricas.

Fuente única de verdad de dónde viven los artefactos, para que train, evaluate,
promote y predict no dupliquen rutas.
"""
import json
from pathlib import Path

MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")

CANDIDATE_MODEL = MODELS_DIR / "candidate" / "lgbm_model.txt"
PRODUCTION_MODEL = MODELS_DIR / "production" / "lgbm_model.txt"

CANDIDATE_METRICS = METRICS_DIR / "candidate.json"
PRODUCTION_METRICS = METRICS_DIR / "production.json"
COMPARISON_METRICS = METRICS_DIR / "comparison.json"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(Path(path), "r") as f:
        return json.load(f)
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/registry.py tests/test_registry.py
git commit -m "feat: registry.py con rutas del registro y helpers JSON"
```

---

### Task 4: Gate — función pura `decide`

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `decide(rmse_candidate: float, rmse_production, margin: float) -> str` — devuelve `"PROMOTE"` o `"KEEP"`. Si `rmse_production is None` (bootstrap) → `"PROMOTE"`.
- Nota: `evaluate.py` inicia con `from __future__ import annotations` (compatibilidad 3.9 del venv de tests).

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_evaluate.py`:

```python
from src.evaluate import decide


def test_decide_promote_when_better_beyond_margin():
    assert decide(rmse_candidate=85.0, rmse_production=100.0, margin=0.01) == "PROMOTE"


def test_decide_keep_when_better_within_margin():
    # 99.5 < 100 pero la mejora (0.5%) no supera el margen de 1%
    assert decide(rmse_candidate=99.5, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_keep_when_worse():
    assert decide(rmse_candidate=110.0, rmse_production=100.0, margin=0.01) == "KEEP"


def test_decide_bootstrap_when_no_production():
    assert decide(rmse_candidate=85.0, rmse_production=None, margin=0.01) == "PROMOTE"
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL (ModuleNotFoundError: src.evaluate).

- [ ] **Step 3: Implementar decide en evaluate.py**

Crear `src/evaluate.py`:

```python
"""Evaluación del candidato contra el modelo de producción (gate de promoción)."""
from __future__ import annotations


def decide(rmse_candidate, rmse_production, margin):
    """Decide si promover el candidato.

    Devuelve "PROMOTE" o "KEEP". Sin modelo de producción (bootstrap) -> PROMOTE.
    Promueve solo si el candidato mejora el RMSE por más del margen relativo.
    """
    if rmse_production is None:
        return "PROMOTE"
    if rmse_candidate < rmse_production * (1 - margin):
        return "PROMOTE"
    return "KEEP"
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: gate de promocion (funcion pura decide)"
```

---

### Task 5: Evaluate — `build_comparison` y orquestación `evaluate_models`

**Files:**
- Modify: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `decide` (Task 4); `registry` (Task 3); `make_supervised`, `chronological_split` (Task 2).
- Produces:
  - `build_comparison(rmse_candidate, rmse_production, margin, git_sha, evaluated_at) -> dict` — arma el dict de `comparison.json` (incluye `delta_pct`, `margin_pct`, `recommendation`). `delta_pct` es `None` en bootstrap.
  - `evaluate_models() -> dict` — orquesta: carga datos, arma validación, evalúa candidato y (si existe) producción, arma la comparación, la escribe en `registry.COMPARISON_METRICS`, la imprime y la retorna.

- [ ] **Step 1: Escribir los tests de build_comparison (falla)**

Agregar a `tests/test_evaluate.py`:

```python
from src.evaluate import build_comparison


def test_build_comparison_promote():
    c = build_comparison(rmse_candidate=85.0, rmse_production=100.0,
                         margin=0.01, git_sha="abc", evaluated_at="2026-08-03")
    assert c["recommendation"] == "PROMOTE"
    assert c["rmse_candidate"] == 85.0
    assert c["rmse_production"] == 100.0
    assert round(c["delta_pct"], 2) == -15.0     # (85-100)/100*100
    assert c["margin_pct"] == 1.0
    assert c["git_sha"] == "abc"
    assert c["evaluated_at"] == "2026-08-03"


def test_build_comparison_bootstrap():
    c = build_comparison(rmse_candidate=85.0, rmse_production=None,
                         margin=0.01, git_sha="abc", evaluated_at="2026-08-03")
    assert c["recommendation"] == "PROMOTE"
    assert c["rmse_production"] is None
    assert c["delta_pct"] is None
```

- [ ] **Step 2: Correr y verificar que falla**

Run: `pytest tests/test_evaluate.py::test_build_comparison_promote -v`
Expected: FAIL (ImportError: build_comparison).

- [ ] **Step 3: Implementar build_comparison y evaluate_models**

Agregar a `src/evaluate.py` (tras `decide`):

```python
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

from src import registry
from src.features import make_supervised, chronological_split
from src.lineage import get_git_commit


def build_comparison(rmse_candidate, rmse_production, margin, git_sha, evaluated_at):
    """Arma el dict de comparison.json a partir de los RMSE y el margen."""
    if rmse_production is None:
        delta_pct = None
    else:
        delta_pct = (rmse_candidate - rmse_production) / rmse_production * 100
    return {
        "rmse_candidate": rmse_candidate,
        "rmse_production": rmse_production,
        "delta_pct": delta_pct,
        "margin_pct": margin * 100,
        "recommendation": decide(rmse_candidate, rmse_production, margin),
        "evaluated_at": evaluated_at,
        "git_sha": git_sha,
    }


def _load_params():
    import yaml
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def _rmse_on_validation(model_path, X_val, y_val):
    model = lgb.Booster(model_file=str(model_path))
    preds = model.predict(X_val)
    return float(np.sqrt(mean_squared_error(y_val, preds)))


def evaluate_models():
    """Evalúa candidato vs producción en la misma ventana y escribe comparison.json."""
    if not registry.CANDIDATE_MODEL.exists():
        raise FileNotFoundError(
            f"No hay modelo candidato en {registry.CANDIDATE_MODEL}. ¿Corrió 'train'?"
        )

    params = _load_params()
    lookback = params["train"]["lookback_days"]
    validation_days = params["train"]["validation_days"]
    margin = params["promotion"]["margin"]

    df = pd.read_csv("data/raw/price_xm_daily.csv")
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)
    X, y = make_supervised(df, lookback)
    _, X_val, _, y_val = chronological_split(X, y, validation_days)

    rmse_candidate = _rmse_on_validation(registry.CANDIDATE_MODEL, X_val, y_val)
    rmse_production = (
        _rmse_on_validation(registry.PRODUCTION_MODEL, X_val, y_val)
        if registry.PRODUCTION_MODEL.exists() else None
    )

    comparison = build_comparison(
        rmse_candidate, rmse_production, margin,
        git_sha=get_git_commit(), evaluated_at=date.today().isoformat(),
    )
    registry.write_json(registry.COMPARISON_METRICS, comparison)

    print("=" * 48)
    print(f"  RMSE candidato:   {rmse_candidate:.2f} $/MWh")
    if rmse_production is None:
        print("  RMSE producción:  (no existe — bootstrap)")
    else:
        print(f"  RMSE producción:  {rmse_production:.2f} $/MWh"
              f"  (delta {comparison['delta_pct']:+.2f}%)")
    print(f"  Recomendación:    {comparison['recommendation']}")
    print("=" * 48)
    return comparison


if __name__ == "__main__":
    evaluate_models()
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (6 tests). `evaluate_models` no se testea unitariamente (requiere modelos LightGBM; se valida en CI).

- [ ] **Step 5: Verificar sintaxis**

Run: `python -m py_compile src/evaluate.py`
Expected: sin salida.

- [ ] **Step 6: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluate_models compara candidato vs produccion en la misma ventana"
```

---

### Task 6: train.py — quitar W&B, guardar candidato, escribir candidate.json

**Files:**
- Modify: `src/train.py`
- Test: (sin unit test nuevo; se valida por CI y por tests de features)

**Interfaces:**
- Consumes: `registry.CANDIDATE_MODEL`, `registry.CANDIDATE_METRICS`, `registry.write_json`; `get_git_commit`, `get_dvc_hash`.
- Produces: `models/candidate/lgbm_model.txt` y `metrics/candidate.json` con claves: `rmse`, `trained_at`, `git_sha`, `data_hash`, `n_train`, `n_val`, `params`.

- [ ] **Step 1: Quitar imports de W&B y agregar los nuevos**

En `src/train.py`, eliminar:

```python
import wandb
from src.lineage import log_lineage_to_wandb
```

Asegurar estos imports (agregar los que falten):

```python
from datetime import date
from src import registry
from src.lineage import get_git_commit, get_dvc_hash
```

- [ ] **Step 2: Guardar el modelo en la ruta del candidato**

Reemplazar el bloque `# 6. Guardar modelo localmente` por:

```python
    # 6. Guardar el modelo candidato
    registry.CANDIDATE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(registry.CANDIDATE_MODEL))
    print(f"💾 Modelo candidato guardado en: {registry.CANDIDATE_MODEL}")
```

- [ ] **Step 3: Reemplazar todo el bloque de W&B por escritura de candidate.json**

Eliminar el bloque `# 7. Registrar en W&B` completo (desde `run = wandb.init(` hasta
`print("✅ Entrenamiento completado y registrado en W&B")`) y reemplazarlo por:

```python
    # 7. Registrar métricas y trazabilidad del candidato
    metrics = {
        "rmse": rmse,
        "trained_at": date.today().isoformat(),
        "git_sha": get_git_commit(),
        "data_hash": get_dvc_hash(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "params": p,
    }
    registry.write_json(registry.CANDIDATE_METRICS, metrics)
    print(f"📝 Métricas del candidato guardadas en: {registry.CANDIDATE_METRICS}")
    print("✅ Entrenamiento completado.")
```

- [ ] **Step 4: Verificar sintaxis**

Run: `python -m py_compile src/train.py`
Expected: sin salida.

- [ ] **Step 5: Verificar que no queda ninguna referencia a wandb**

Run: `grep -rn "wandb" src/train.py`
Expected: sin resultados.

- [ ] **Step 6: Correr toda la suite (no debe romperse nada)**

Run: `pytest -v`
Expected: PASS (todos).

- [ ] **Step 7: Commit**

```bash
git add src/train.py
git commit -m "feat: train guarda candidato y candidate.json; se retira W&B de train"
```

---

### Task 7: promote.py — candidato → producción

**Files:**
- Create: `src/promote.py`
- Test: `tests/test_promote.py`

**Interfaces:**
- Consumes: `registry` (constantes + `read_json`/`write_json`).
- Produces: `promote(candidate_model=..., production_model=..., candidate_metrics=..., production_metrics=...) -> None` — copia el modelo candidato a producción y escribe `production.json` (= candidate.json + `promoted_at`). Lanza `FileNotFoundError` si no hay candidato. Los parámetros tienen default a las constantes de `registry` para permitir inyección de rutas en tests.

- [ ] **Step 1: Escribir los tests que fallan**

Crear `tests/test_promote.py`:

```python
import json
from pathlib import Path

import pytest

from src.promote import promote


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_promote_copies_model_and_writes_production_metrics(tmp_path):
    cand_model = tmp_path / "models/candidate/lgbm_model.txt"
    prod_model = tmp_path / "models/production/lgbm_model.txt"
    cand_metrics = tmp_path / "metrics/candidate.json"
    prod_metrics = tmp_path / "metrics/production.json"
    _write(cand_model, "MODELO-CANDIDATO")
    _write(cand_metrics, json.dumps({"rmse": 85.0, "git_sha": "abc"}))

    promote(cand_model, prod_model, cand_metrics, prod_metrics)

    assert prod_model.read_text() == "MODELO-CANDIDATO"
    prod = json.loads(prod_metrics.read_text())
    assert prod["rmse"] == 85.0
    assert prod["git_sha"] == "abc"
    assert "promoted_at" in prod


def test_promote_fails_without_candidate(tmp_path):
    with pytest.raises(FileNotFoundError):
        promote(
            tmp_path / "models/candidate/lgbm_model.txt",
            tmp_path / "models/production/lgbm_model.txt",
            tmp_path / "metrics/candidate.json",
            tmp_path / "metrics/production.json",
        )
```

- [ ] **Step 2: Correr y verificar que fallan**

Run: `pytest tests/test_promote.py -v`
Expected: FAIL (ModuleNotFoundError: src.promote).

- [ ] **Step 3: Implementar promote.py**

Crear `src/promote.py`:

```python
"""Promoción manual: mueve el modelo candidato a producción."""
import shutil
from datetime import date
from pathlib import Path

from src import registry


def promote(candidate_model=registry.CANDIDATE_MODEL,
            production_model=registry.PRODUCTION_MODEL,
            candidate_metrics=registry.CANDIDATE_METRICS,
            production_metrics=registry.PRODUCTION_METRICS):
    """Copia el modelo candidato a producción y actualiza production.json."""
    candidate_model = Path(candidate_model)
    if not candidate_model.exists():
        raise FileNotFoundError(
            f"No hay candidato para promover en {candidate_model}."
        )

    production_model = Path(production_model)
    production_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(candidate_model, production_model)

    metrics = registry.read_json(candidate_metrics)
    metrics["promoted_at"] = date.today().isoformat()
    registry.write_json(production_metrics, metrics)

    print(f"🚀 Modelo promovido a producción: {production_model}")
    print(f"📝 Métricas de producción actualizadas: {production_metrics}")


if __name__ == "__main__":
    promote()
```

- [ ] **Step 4: Correr y verificar que pasan**

Run: `pytest tests/test_promote.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/promote.py tests/test_promote.py
git commit -m "feat: promote.py mueve candidato a produccion"
```

---

### Task 8: predict.py — usar modelo de producción

**Files:**
- Modify: `src/predict.py`

**Interfaces:**
- Consumes: `registry.PRODUCTION_MODEL`.

- [ ] **Step 1: Cambiar la ruta del modelo a producción**

En `src/predict.py`, reemplazar:

```python
    model_path = Path("models/lgbm_model.txt")
```

por:

```python
    from src import registry
    model_path = registry.PRODUCTION_MODEL
```

Y actualizar el mensaje del `FileNotFoundError` para reflejar producción:

```python
    if not model_path.exists():
        raise FileNotFoundError(
            f"No hay modelo de producción en {model_path}. "
            "¿Se promovió algún modelo con promote.yml?"
        )
```

- [ ] **Step 2: Verificar sintaxis**

Run: `python -m py_compile src/predict.py`
Expected: sin salida.

- [ ] **Step 3: Correr la suite**

Run: `pytest -v`
Expected: PASS (todos).

- [ ] **Step 4: Commit**

```bash
git add src/predict.py
git commit -m "feat: predict usa el modelo de produccion"
```

---

### Task 9: lineage.py sin W&B + quitar wandb de requirements

**Files:**
- Modify: `src/lineage.py`
- Modify: `requirements.txt`
- Test: `tests/test_lineage.py` (ya cubre las funciones que quedan)

**Interfaces:**
- Produces: `lineage.py` con `get_git_commit`, `get_dvc_hash` (sin `log_lineage_to_wandb`).

- [ ] **Step 1: Eliminar la función de W&B**

En `src/lineage.py`, eliminar por completo la función `log_lineage_to_wandb(run)`
(desde su `def` hasta su último `print`). Conservar `get_git_commit` y `get_dvc_hash`.

- [ ] **Step 2: Quitar wandb de requirements.txt**

Eliminar la línea `wandb==0.28.1` de `requirements.txt`.

- [ ] **Step 3: Verificar que no queda ninguna referencia a wandb en src/**

Run: `grep -rn "wandb" src/`
Expected: sin resultados.

- [ ] **Step 4: Correr los tests de linaje**

Run: `pytest tests/test_lineage.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/lineage.py requirements.txt
git commit -m "chore: retirar W&B (lineage y requirements)"
```

---

### Task 10: DVC — candidato como output de train; quitar stage predict; .gitignore

**Files:**
- Modify: `dvc.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `dvc.yaml` con stages `ingest` y `train` (train produce el candidato y `metrics/candidate.json`). Sin stage `predict`.

- [ ] **Step 1: Reescribir dvc.yaml**

Contenido completo de `dvc.yaml`:

```yaml
stages:
  ingest:
    cmd: python src/ingest_xm.py
    deps:
      - src/ingest_xm.py
    outs:
      - data/raw/price_xm_daily.csv:
          cache: true
      - data/raw/price_xm_hourly.csv:
          cache: true

  train:
    cmd: python -m src.train
    deps:
      - src/train.py
      - src/features.py
      - data/raw/price_xm_daily.csv
    params:
      - train
    outs:
      - models/candidate/lgbm_model.txt:
          cache: true
    metrics:
      - metrics/candidate.json:
          cache: false
```

- [ ] **Step 2: Ajustar .gitignore para permitir punteros .dvc de modelos**

En `.gitignore`, reemplazar la línea `models/` por:

```
# Modelos: binarios versionados con DVC (se rastrean los .dvc, no los .txt)
models/**/*.txt
```

Y asegurar que `metrics/` NO esté ignorado (los JSON van a git). Si existe alguna
regla que ignore `metrics/`, quitarla.

- [ ] **Step 3: Verificar YAML válido**

Run: `python -c "import yaml; yaml.safe_load(open('dvc.yaml'))"`
Expected: sin error.

- [ ] **Step 4: Commit**

```bash
git add dvc.yaml .gitignore
git commit -m "chore: dvc.yaml produce candidato; predict sale del DAG"
```

**Nota:** `dvc.lock` se regenerará en la primera corrida de CI (`dvc repro`). La
promoción del modelo de producción se versiona con `dvc add models/production/...`
dentro de `promote.yml` (Task 12), no como stage.

---

### Task 11: Workflow training.yml (semanal + manual)

**Files:**
- Create: `.github/workflows/training.yml`

- [ ] **Step 1: Crear el workflow**

Crear `.github/workflows/training.yml`:

```yaml
name: Training Pipeline

on:
  schedule:
    - cron: '0 3 * * 1'   # lunes 03:00 UTC
  workflow_dispatch:

permissions:
  id-token: write
  contents: write

jobs:
  training:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest -v

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: 'projects/705685420706/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
          service_account: 'github-actions-sa@xm-energy-mlops.iam.gserviceaccount.com'

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Pull data and models from GCS
        run: dvc pull || echo "Nada que traer (primera corrida)"

      - name: Train candidate (ingest + train)
        run: dvc repro

      - name: Evaluate candidate vs production
        run: python -m src.evaluate

      - name: Upload comparison report
        uses: actions/upload-artifact@v4
        with:
          name: comparison-report
          path: metrics/comparison.json

      - name: Push artifacts to GCS
        run: dvc push

      - name: Commit metrics and DVC metadata
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "chore: candidato entrenado y evaluado" || echo "Sin cambios"
          git push || echo "Sin cambios para push"
```

- [ ] **Step 2: Verificar YAML válido**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/training.yml'))"`
Expected: sin error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/training.yml
git commit -m "ci: workflow de entrenamiento semanal con evaluacion"
```

---

### Task 12: Workflow promote.yml (solo manual)

**Files:**
- Create: `.github/workflows/promote.yml`

- [ ] **Step 1: Crear el workflow**

Crear `.github/workflows/promote.yml`:

```yaml
name: Promote Model

on:
  workflow_dispatch:

permissions:
  id-token: write
  contents: write

jobs:
  promote:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: 'projects/705685420706/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
          service_account: 'github-actions-sa@xm-energy-mlops.iam.gserviceaccount.com'

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Pull candidate from GCS
        run: dvc pull

      - name: Promote candidate to production
        run: python -m src.promote

      - name: Track production model with DVC and push
        run: |
          dvc add models/production/lgbm_model.txt
          dvc push

      - name: Commit production model and metrics
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "chore: promover modelo a produccion" || echo "Sin cambios"
          git push || echo "Sin cambios para push"
```

- [ ] **Step 2: Verificar YAML válido**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/promote.yml'))"`
Expected: sin error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/promote.yml
git commit -m "ci: workflow manual de promocion a produccion"
```

---

### Task 13: Workflow inference.yml (diario + manual) y borrar daily_pipeline.yml

**Files:**
- Create: `.github/workflows/inference.yml`
- Delete: `.github/workflows/daily_pipeline.yml`

- [ ] **Step 1: Crear el workflow de inferencia**

Crear `.github/workflows/inference.yml`:

```yaml
name: Inference Pipeline

on:
  schedule:
    - cron: '0 2 * * *'   # diario 02:00 UTC
  workflow_dispatch:

permissions:
  id-token: write
  contents: write

jobs:
  inference:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-dev.txt

      - name: Run tests
        run: pytest -v

      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: 'projects/705685420706/locations/global/workloadIdentityPools/github-pool/providers/github-provider'
          service_account: 'github-actions-sa@xm-energy-mlops.iam.gserviceaccount.com'

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Pull production model from GCS
        run: dvc pull

      - name: Ingest fresh data
        run: dvc repro ingest

      - name: Predict with production model
        run: python -m src.predict

      - name: Push data to GCS
        run: dvc push

      - name: Commit DVC metadata
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add -A
          git commit -m "chore: prediccion diaria" || echo "Sin cambios"
          git push || echo "Sin cambios para push"
```

- [ ] **Step 2: Borrar el workflow monolítico**

Run: `git rm .github/workflows/daily_pipeline.yml`

- [ ] **Step 3: Verificar YAML válido**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/inference.yml'))"`
Expected: sin error.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/inference.yml
git commit -m "ci: workflow de inferencia diaria; eliminar daily_pipeline.yml"
```

---

### Task 14: Documentación — actualizar README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Actualizar arquitectura, workflows, estructura y roadmap**

En `README.md`:
- Reemplazar el diagrama Mermaid y la tabla de etapas por la arquitectura de 3
  workflows (training/promote/inference) y el registro `models/candidate` +
  `models/production`.
- En "Stack": quitar la línea de **Weights & Biases**.
- En "Estructura del proyecto": agregar `src/evaluate.py`, `src/promote.py`,
  `src/registry.py`, `metrics/`, y `models/candidate` + `models/production`.
- En "Automatización": describir los tres workflows y sus cadencias.
- Agregar una sección **"Ciclo de vida del modelo"** documentando el bootstrap:
  `training.yml` → revisar `comparison.json` → `promote.yml` (manual) → la
  inferencia diaria arranca una vez exista un modelo de producción.
- En "Roadmap": marcar como hechas ✅ "Separar entrenamiento y despliegue" y
  "Gate de promoción a producción"; eliminar "Activar W&B online" (descartado).

- [ ] **Step 2: Verificar que no queda mención a W&B/wandb como componente activo**

Run: `grep -ni "w&b\|wandb\|weights" README.md`
Expected: sin resultados (o solo en contexto histórico si se decide dejarlo).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README refleja la arquitectura de 3 workflows y el registro"
```

---

### Task 15: Validación end-to-end en CI

**Files:** (ninguno — validación)

- [ ] **Step 1: Push de todo a main**

Run: `git push`
Expected: OK.

- [ ] **Step 2: Disparar training.yml y verificar verde**

Disparar `training.yml` vía `workflow_dispatch` (API o pestaña Actions) y confirmar
que todos los pasos quedan en `success`, incluyendo `Run tests`, `dvc repro`
(ingest+train) y `Evaluate candidate vs production` (en bootstrap: recomendación
`PROMOTE`, `rmse_production = null`).

- [ ] **Step 3: Disparar promote.yml y verificar que siembra producción**

Disparar `promote.yml`; confirmar `success` y que `models/production/lgbm_model.txt.dvc`
y `metrics/production.json` quedaron commiteados.

- [ ] **Step 4: Disparar inference.yml y verificar la predicción**

Disparar `inference.yml`; confirmar `success` y que `data/processed/prediction.csv`
se generó con el modelo de producción.

- [ ] **Step 5: Marcar el plan como completado**

Actualizar el estado del spec a "Implementado" y cerrar.

---

## Self-Review

**Spec coverage:**
- Separación train/inference → Tasks 11, 13. ✅
- Registro DVC/GCS (candidate/production + metrics) → Tasks 3, 6, 7, 10, 12. ✅
- Gate de promoción manual con comparación justa → Tasks 4, 5, 11 (reporte), 12 (promote). ✅
- Retiro de W&B → Tasks 6, 9. ✅
- Módulos nuevos evaluate/promote + ajustes train/predict/lineage → Tasks 5, 6, 7, 8, 9. ✅
- Bootstrap → Tasks 4/5 (decide/evaluate), 8 (predict falla), 14 (doc), 15 (validación). ✅
- Manejo de errores (fallos ruidosos) → Tasks 5, 7. ✅
- Testing (evaluate/promote/params) → Tasks 1, 4, 5, 7. ✅
- Config (promotion.margin, quitar wandb, workflows) → Tasks 1, 9, 11–13. ✅

**Placeholder scan:** Sin TBD/TODO; cada paso de código muestra el código completo.

**Type consistency:** `decide(rmse_candidate, rmse_production, margin)` usado igual en
Tasks 4 y 5; `build_comparison(...)` firma consistente; rutas de `registry` usadas
con los mismos nombres en Tasks 3, 5, 6, 7, 8. `promote(...)` con la misma firma en
Task 7 (impl y tests). ✅
