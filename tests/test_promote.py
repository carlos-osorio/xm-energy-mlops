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
