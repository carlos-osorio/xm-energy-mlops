"""Verifica que params.yaml tenga la estructura que train.py espera."""
import yaml


def load():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def test_train_section_exists():
    assert "train" in load()


def test_required_keys_present():
    p = load()["train"]
    for key in [
        "seed",
        "lookback_days",
        "validation_days",
        "num_boost_round",
        "early_stopping_rounds",
        "lgbm",
    ]:
        assert key in p, f"Falta '{key}' en params.yaml"


def test_lgbm_keys_present():
    lgbm = load()["train"]["lgbm"]
    for key in ["learning_rate", "num_leaves", "feature_fraction",
                "bagging_fraction", "bagging_freq"]:
        assert key in lgbm, f"Falta '{key}' en train.lgbm"


def test_sane_values():
    p = load()["train"]
    assert p["lookback_days"] > 0
    assert p["validation_days"] > 0
    assert 0 < p["lgbm"]["learning_rate"] <= 1
