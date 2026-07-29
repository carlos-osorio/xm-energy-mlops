"""Tests de la lógica de features compartida entre train e inferencia.

El test clave es `test_train_serve_alignment`: bloquea el bug de desfase
(train/serve skew) que corregimos, para que no reaparezca.
"""
import numpy as np
import pandas as pd

from src.features import build_lag_features, feature_columns, build_inference_vector
from src.features import make_supervised, chronological_split

LOOKBACK = 7


def _sample_prices():
    # Serie fácil de rastrear: precio = 100 + índice del día
    return [100 + i for i in range(30)]


def test_lag1_is_today_and_target_is_tomorrow():
    prices = _sample_prices()
    df = pd.DataFrame({"PrecioPromedio": prices})
    feat = build_lag_features(df, LOOKBACK)

    i = 10  # una fila cualquiera con contexto suficiente
    assert feat["precio_lag_1"].iloc[i] == prices[i]        # lag_1 = HOY
    assert feat["precio_lag_2"].iloc[i] == prices[i - 1]    # lag_2 = ayer
    assert feat["precio_lag_7"].iloc[i] == prices[i - 6]    # lag_7 = hace 6 días
    assert feat["precio_target"].iloc[i] == prices[i + 1]   # target = MAÑANA


def test_no_gap_between_features_and_target():
    # El target debe ser exactamente el día siguiente a lag_1 (sin hueco).
    prices = _sample_prices()
    df = pd.DataFrame({"PrecioPromedio": prices})
    feat = build_lag_features(df, LOOKBACK)
    i = 15
    assert feat["precio_target"].iloc[i] - feat["precio_lag_1"].iloc[i] == 1


def test_train_serve_alignment():
    """El vector de inferencia debe ser idéntico a como train arma las features
    para el día más reciente. Este test detecta cualquier desfase train/serve."""
    prices = _sample_prices()
    df = pd.DataFrame({"PrecioPromedio": prices})

    # Features tal como las construye train (sin dropna para conservar la última fila)
    feat_all = build_lag_features(df, LOOKBACK)
    last_row_features = feat_all[feature_columns(LOOKBACK)].iloc[-1].to_numpy()

    # Vector tal como lo construye predict
    inference_vector = build_inference_vector(prices, LOOKBACK)[0]

    assert np.array_equal(last_row_features, inference_vector)


def test_inference_vector_order_and_length():
    prices = [10, 20, 30, 40, 50, 60, 70, 80]
    vec = build_inference_vector(prices, LOOKBACK)
    assert vec.shape == (1, LOOKBACK)
    assert vec[0][0] == 80   # lag_1 = el más reciente
    assert vec[0][-1] == 20  # lag_7 = 6 días atrás


def test_feature_columns():
    assert feature_columns(3) == ["precio_lag_1", "precio_lag_2", "precio_lag_3"]


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
