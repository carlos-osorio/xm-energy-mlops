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
