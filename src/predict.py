import pandas as pd
import lightgbm as lgb
from pathlib import Path
from src.features import build_inference_vector
from src import registry


def append_prediction(history_path, row):
    """Añade `row` (dict) a un CSV historial en `history_path`.

    Si el archivo no existe, lo crea con encabezado (orden de las keys del dict)
    y escribe la fila. Si ya existe, añade la fila sin repetir encabezado.
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row])
    write_header = not history_path.exists()
    row_df.to_csv(history_path, mode="a", index=False, header=write_header)


def predict_next_day():
    # 1. Cargar el modelo entrenado localmente en la etapa 'train'
    model_path = registry.PRODUCTION_MODEL
    print(f"🔍 Cargando modelo local desde: {model_path}...")
    if not model_path.exists():
        raise FileNotFoundError(
            f"No hay modelo de producción en {model_path}. "
            "¿Se promovió algún modelo con promote.yml?"
        )

    model = lgb.Booster(model_file=str(model_path))
    print("✅ Modelo cargado.")

    # 2. Cargar los últimos datos disponibles
    data_path = Path("data/raw/price_xm_daily.csv")
    df = pd.read_csv(data_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)

    # 3. Preparar el vector de features con la lógica compartida.
    # El lookback lo determina el propio modelo (nº de features con que se entrenó),
    # así predict nunca se desalinea aunque cambie lookback_days en params.yaml.
    lookback = model.num_feature()
    prices = df['PrecioPromedio'].values
    X_pred = build_inference_vector(prices, lookback)

    # 4. Predecir el precio del día siguiente
    precio_manana_pred = model.predict(X_pred)[0]
    precio_hoy = prices[-1]  # El más reciente en el CSV (último cronológicamente)

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

    # 6. Acumular la predicción en un historial versionado por git
    history_path = Path("data/predictions_history.csv")
    append_prediction(history_path, {
        "fecha_generacion": prediction_df["fecha_generacion"].iloc[0],
        "fecha_objetivo": prediction_df["fecha_objetivo"].iloc[0],
        "precio_referencia": prediction_df["precio_referencia"].iloc[0],
        "precio_predicho": prediction_df["precio_predicho"].iloc[0],
    })
    print(f"📈 Historial de predicciones actualizado en: {history_path}")
    print("🚀 Inferencia completada.")

if __name__ == "__main__":
    predict_next_day()
