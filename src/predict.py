import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

def predict_next_day():
    # 1. Cargar el modelo entrenado localmente en la etapa 'train'
    model_path = Path("models/lgbm_model.txt")
    print(f"🔍 Cargando modelo local desde: {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {model_path}. ¿Corrió la etapa 'train'?"
        )

    model = lgb.Booster(model_file=str(model_path))
    print("✅ Modelo cargado.")

    # 2. Cargar los últimos datos disponibles
    data_path = Path("data/raw/price_xm_daily.csv")
    df = pd.read_csv(data_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)

    # Tomar los últimos 7 precios (lookback)
    last_7_prices = df['PrecioPromedio'].tail(7).values

    # 3. Preparar el vector de features
    # El modelo espera: [precio_lag_1, precio_lag_2, ..., precio_lag_7]
    # precio_lag_1 es el precio de HOY (el más reciente), precio_lag_7 es el más antiguo.
    # last_7_prices viene en orden cronológico (ascendente); lo invertimos para que
    # el más reciente quede en lag_1, igual que en el entrenamiento.
    X_pred = np.array([last_7_prices[::-1]])

    # 4. Predecir el precio del día siguiente
    precio_manana_pred = model.predict(X_pred)[0]
    precio_hoy = last_7_prices[-1]  # El más reciente en el CSV (último cronológicamente)

    print("-" * 40)
    print(f"📊 Precio de referencia (último dato): ${precio_hoy:.2f} $/MWh")
    print(f"🎯 Precio predicho para mañana:      ${precio_manana_pred:.2f} $/MWh")
    print("-" * 40)

    # 5. Guardar la predicción en disco
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_df = pd.DataFrame({
        'fecha_generacion': [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        'fecha_objetivo': [(df['Fecha'].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
        'precio_referencia': [precio_hoy],
        'precio_predicho': [precio_manana_pred]
    })

    output_path = output_dir / "prediction.csv"
    prediction_df.to_csv(output_path, index=False)
    print(f"💾 Predicción guardada en: {output_path}")
    print("🚀 Inferencia completada.")

if __name__ == "__main__":
    predict_next_day()
