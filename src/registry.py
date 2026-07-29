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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_json(path):
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)
