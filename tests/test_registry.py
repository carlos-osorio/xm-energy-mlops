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
