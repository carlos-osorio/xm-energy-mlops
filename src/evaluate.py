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
