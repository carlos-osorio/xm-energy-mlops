import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from pathlib import Path
import wandb
from src.lineage import log_lineage_to_wandb

def create_features(df, lookback=7):
    """Crea features de ventana deslizante y la variable objetivo (delta)."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Crear features de lookback (temperaturas de días anteriores)
    for i in range(1, lookback + 1):
        df[f'temp_lag_{i}'] = df['Temp'].shift(i)
    
    # Variable objetivo: delta de temperatura respecto al día anterior
    df['target'] = df['Temp'].diff()
    
    # Eliminar filas con NaN (por los lags y el diff)
    df = df.dropna().reset_index(drop=True)
    
    return df

def train_model():
    # 1. Cargar datos desde DVC
    data_path = Path("data/raw/sample_timeseries.csv")
    print(f"Cargando datos desde {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Crear features
    lookback = 7
    df_features = create_features(df, lookback=lookback)
    
    # 3. Split temporal (80% train, 20% validation)
    split_idx = int(len(df_features) * 0.8)
    train_df = df_features.iloc[:split_idx]
    val_df = df_features.iloc[split_idx:]
    
    # 4. Preparar X e y
    feature_cols = [f'temp_lag_{i}' for i in range(1, lookback + 1)]
    
    X_train = train_df[feature_cols]
    y_train = train_df['target']
    X_val = val_df[feature_cols]
    y_val = val_df['target']
    
    # 5. Entrenar LightGBM
    print("Entrenando modelo LightGBM...")
    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
    )
    
    # 6. Evaluar en validación
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"RMSE en validación: {rmse:.4f}")
    
    # 7. Registrar en W&B
    run = wandb.init(
        project="xm-energy-mlops",
        name="lgbm-mvp-training",
        config={
            "lookback": lookback,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "train_size": len(train_df),
            "val_size": len(val_df)
        }
    )
    
    # Registrar linaje (Git + DVC)
    log_lineage_to_wandb(run)
    
    # Registrar métrica
    run.log({"rmse_val": rmse})
    
    # Guardar modelo como artefacto de W&B
    model_path = Path("models/lgbm_model.txt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    
    artifact = wandb.Artifact(
        name="lgbm-mvp-model",
        type="model",
        description="LightGBM MVP para predicción de deltas de temperatura"
    )
    artifact.add_file(str(model_path))
    run.log_artifact(artifact)
    
    run.finish()
    print(f"Modelo guardado y registrado en W&B. RMSE: {rmse:.4f}")

if __name__ == "__main__":
    train_model()