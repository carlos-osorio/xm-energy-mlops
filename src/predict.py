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
        # Esta es la forma más robusta: pedir el artefacto por nombre y alias
        artifact = api.artifact(f"{entity}/{project}/lgbm-mvp-model:production")
        download_dir = artifact.download()
        model_path = Path(download_dir) / "lgbm_model.txt"
        print(f"✅ Modelo descargado en: {model_path}")
    except Exception as e:
        print(f"❌ Error al descargar el modelo. ¿Está marcado como 'production'? Detalle: {e}")
        return

    # 2. Cargar el modelo LightGBM
    model = lgb.Booster(model_file=str(model_path))
    
    # 3. Cargar los últimos datos disponibles
    data_path = Path("data/raw/sample_timeseries.csv")
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Tomar las últimas 7 temperaturas (lookback)
    last_7_temps = df['Temp'].tail(7).values
    
    # 4. Preparar el vector de features
    # El modelo espera: [temp_lag_1, temp_lag_2, ..., temp_lag_7]
    # temp_lag_1 es la temperatura de "ayer" (la más reciente), temp_lag_7 es la más antigua.
    # Invertimos el array para que coincida con el orden de entrenamiento.
    X_pred = np.array([last_7_temps[::-1]])
    
    # 5. Predecir el delta y la temperatura final
    delta_pred = model.predict(X_pred)[0]
    temp_hoy = last_7_temps[0]  # La más reciente en el CSV
    temp_manana = temp_hoy + delta_pred
    
    print("-" * 40)
    print(f"📊 Temperatura de referencia (último dato): {temp_hoy:.2f}°C")
    print(f"📈 Delta predicho por el modelo:          {delta_pred:+.2f}°C")
    print(f"🎯 Temperatura predicha para mañana:      {temp_manana:.2f}°C")
    print("-" * 40)
    
    # 6. Guardar la predicción en disco
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prediction_df = pd.DataFrame({
        'fecha_generacion': [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        'fecha_objetivo': [(df['Date'].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")],
        'temperatura_referencia': [temp_hoy],
        'delta_predicho': [delta_pred],
        'temperatura_predicha': [temp_manana]
    })
    
    output_path = output_dir / "prediction.csv"
    prediction_df.to_csv(output_path, index=False)
    print(f"💾 Predicción guardada en: {output_path}")
    
    # 7. Registrar la inferencia en W&B
    run = wandb.init(
        project=project,
        name="batch-inference",
        config={
            "model_alias": "production",
            "artifact_version": artifact.version
        }
    )
    
    log_lineage_to_wandb(run)
    run.log({"predicted_temperature": temp_manana})
    
    # Guardar el CSV como artefacto para trazabilidad
    pred_artifact = wandb.Artifact(
        name="daily-prediction",
        type="prediction",
        description="Predicción batch generada automáticamente"
    )
    pred_artifact.add_file(str(output_path))
    run.log_artifact(pred_artifact)
    
    run.finish()
    print("🚀 Inferencia completada y registrada en W&B")

if __name__ == "__main__":
    predict_next_day()