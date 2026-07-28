"""Construcción de features de lag, compartida por entrenamiento e inferencia.

Tener una única fuente para armar el vector de features evita el 'train/serve
skew': que train.py y predict.py construyan las features de forma distinta.
"""
import numpy as np
import pandas as pd


def feature_columns(lookback):
    """Nombres de las columnas de features: precio_lag_1 .. precio_lag_lookback."""
    return [f"precio_lag_{i}" for i in range(1, lookback + 1)]


def build_lag_features(df, lookback):
    """Agrega columnas de lag y la variable objetivo a un DataFrame de precios.

    Convención (debe ser idéntica en train e inferencia):
      - precio_lag_1 = precio de HOY (shift 0)
      - precio_lag_2 = ayer (shift 1)
      - ...
      - precio_lag_N = hace N-1 días (shift N-1)
      - precio_target = precio del día siguiente (shift -1)

    No hace dropna: el que llama decide (train elimina NaN; para inspeccionar la
    última fila de inferencia conviene conservarlos).
    """
    df = df.copy()
    for lag in range(1, lookback + 1):
        df[f"precio_lag_{lag}"] = df["PrecioPromedio"].shift(lag - 1)
    df["precio_target"] = df["PrecioPromedio"].shift(-1)
    return df


def build_inference_vector(prices, lookback):
    """Arma el vector [[lag_1, ..., lag_lookback]] para predecir el día siguiente.

    `prices` es una secuencia cronológica ascendente; se toman los últimos
    `lookback` valores y se invierten para que el más reciente quede en lag_1,
    igual que en build_lag_features.
    """
    last = np.asarray(prices)[-lookback:]
    return np.array([last[::-1]])
