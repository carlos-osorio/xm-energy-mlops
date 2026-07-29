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
