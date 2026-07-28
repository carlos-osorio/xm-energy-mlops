import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import wandb
from src.lineage import log_lineage_to_wandb

def predict_next_day():
    # 1. Descargar el modelo de producción directamente por su alias
    print("🔍 Buscando modelo con alias 'production' en W&B...")
    api = wandb.Api()
    entity = api.default_entity
    project = "xm-energy-mlops"
    
    try:
        artifact = api.artifact(f"{entity}/{project}/lgbm-price-model:production")
        download_dir = artifact.download()
        model_path = Path(download_dir) / "lgbm_model.txt"
        print(f"✅ Modelo descargado en: {model_path}")
    except Exception as e:
        print(f"❌ Error al descargar el modelo. ¿Está marcado como 'production'? Detalle: {e}")
        return

    # 2. Cargar el modelo LightGBM
    model = lgb.Booster(model_file=str(model_path))
    
    # 3. Cargar los últimos datos disponibles
    data_path = Path("data/raw/price_xm_daily.csv")
    df = pd.read_csv(data_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    # Tomar los últimos 7 precios (lookback)
    last_7_prices = df['PrecioPromedio'].tail(7).values
    
    # 4. Preparar el vector de features
    # El modelo espera: [precio_lag_1, precio_lag_2, ..., precio_lag_7]
    # precio_lag_1 es el precio de "ayer" (el más reciente), precio_lag_7 es el más antiguo.
    # Invertimos el array para que coincida con el orden de entrenamiento.
    X_pred = np.array([last_7_prices[::-1]])
    
    # 5. Predecir el precio del día siguiente
    precio_manana_pred = model.predict(X_pred)[0]
    precio_hoy = last_7_prices[0]  # El más reciente en el CSV
    
    print("-" * 40)
    print(f"📊 Precio de referencia (último dato): ${precio_hoy:.2f} $/MWh")
    print(f"🎯 Precio predicho para mañana:      ${precio_manana_pred:.2f} $/MWh")
    print("-" * 40)
    
    # 6. Guardar la predicción en disco
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
    
    # 7. Registrar la inferencia en W&B
    run = wandb.init(
        project=project,
        name="batch-inference-price",
        config={
            "model_alias": "production",
            "artifact_version": artifact.version
        }
    )
    
    log_lineage_to_wandb(run)
    run.log({"predicted_price": precio_manana_pred})
    
    # Guardar el CSV como artefacto para trazabilidad
    pred_artifact = wandb.Artifact(
        name="daily-price-prediction",
        type="prediction",
        description="Predicción batch de precio diario generada automáticamente"
    )
    pred_artifact.add_file(str(output_path))
    run.log_artifact(pred_artifact)
    
    run.finish()
    print("🚀 Inferencia completada y registrada en W&B")

if __name__ == "__main__":
    predict_next_day()