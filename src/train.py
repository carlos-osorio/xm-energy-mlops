import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path
import wandb
from sklearn.metrics import mean_squared_error
from src.lineage import log_lineage_to_wandb

def train_model():
    print("🔄 Iniciando entrenamiento del modelo LightGBM...")
    
    # 1. Cargar datos reales de precios diarios de XM
    data_path = Path("data/raw/price_xm_daily.csv")
    print(f"📂 Cargando datos desde {data_path}...")
    
    df = pd.read_csv(data_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    print(f"📊 Total de registros: {len(df)}")
    
    # 2. Crear features de lag: los últimos 7 días INCLUYENDO hoy.
    #    lag_1 = precio de hoy (shift 0), lag_2 = ayer, ..., lag_7 = hace 6 días.
    #    Esto debe coincidir con cómo predict.py arma el vector en inferencia.
    for lag in range(1, 8):
        df[f'precio_lag_{lag}'] = df['PrecioPromedio'].shift(lag - 1)

    # La variable objetivo es el precio del día siguiente
    df['precio_target'] = df['PrecioPromedio'].shift(-1)

    # Eliminar filas con NaN (las primeras 6 por los lags y la última por el target)
    df = df.dropna()
    
    X = df[[f'precio_lag_{i}' for i in range(1, 8)]]
    y = df['precio_target']
    
    # 3. Dividir en entrenamiento y validación (últimos 30 días para validación)
    split_idx = len(df) - 30
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"🎯 Entrenamiento: {len(X_train)} muestras")
    print(f"🎯 Validación: {len(X_val)} muestras")
    
    # 4. Entrenar modelo LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'regression',
        'metric': 'l2',
        'verbosity': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5
    }
    
    print("🚀 Entrenando modelo...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
    )
    
    # 5. Evaluar en validación
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"✅ RMSE en validación: {rmse:.2f} $/MWh")
    
    # 6. Guardar modelo localmente
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"💾 Modelo guardado en: {model_path}")
    
    # 7. Registrar en W&B
    run = wandb.init(
        project="xm-energy-mlops",
        name="price-daily-prediction",
        config={
            "model_type": "lightgbm",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "lookback_days": 7,
            "rmse": rmse
        }
    )
    
    # Registrar linaje de Git y DVC
    log_lineage_to_wandb(run)
    
    # Registrar métricas
    run.log({
        "rmse": rmse,
        "train_samples": len(X_train),
        "val_samples": len(X_val)
    })
    
    # Registrar el modelo como artefacto
    model_artifact = wandb.Artifact(
        name="lgbm-price-model",
        type="model",
        description="Modelo LightGBM para predicción de precio diario de bolsa"
    )
    model_artifact.add_file(str(model_path))
    run.log_artifact(model_artifact)
    
    run.finish()
    print("✅ Entrenamiento completado y registrado en W&B")

if __name__ == "__main__":
    train_model()