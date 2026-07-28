import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
from pathlib import Path
import wandb
from sklearn.metrics import mean_squared_error
from src.lineage import log_lineage_to_wandb

def load_params():
    """Carga los hiperparámetros desde params.yaml (fuente única de verdad)."""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)["train"]

def train_model():
    p = load_params()
    print("🔄 Iniciando entrenamiento del modelo LightGBM...")
    
    # 1. Cargar datos reales de precios diarios de XM
    data_path = Path("data/raw/price_xm_daily.csv")
    print(f"📂 Cargando datos desde {data_path}...")
    
    df = pd.read_csv(data_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').reset_index(drop=True)
    
    print(f"📊 Total de registros: {len(df)}")
    
    # 2. Crear features de lag: los últimos `lookback_days` precios INCLUYENDO hoy.
    #    lag_1 = precio de hoy (shift 0), lag_2 = ayer, ..., lag_N = hace N-1 días.
    #    Esto debe coincidir con cómo predict.py arma el vector en inferencia.
    lookback = p["lookback_days"]
    for lag in range(1, lookback + 1):
        df[f'precio_lag_{lag}'] = df['PrecioPromedio'].shift(lag - 1)

    # La variable objetivo es el precio del día siguiente
    df['precio_target'] = df['PrecioPromedio'].shift(-1)

    # Eliminar filas con NaN (las primeras por los lags y la última por el target)
    df = df.dropna()

    X = df[[f'precio_lag_{i}' for i in range(1, lookback + 1)]]
    y = df['precio_target']

    # 3. Dividir en entrenamiento y validación (últimos N días para validación)
    split_idx = len(df) - p["validation_days"]
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
        **p['lgbm'],           # hiperparámetros de LightGBM desde params.yaml
        'seed': p['seed'],     # fija bagging_seed, feature_fraction_seed, etc.
    }

    print("🚀 Entrenando modelo...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=p['num_boost_round'],
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(p['early_stopping_rounds']), lgb.log_evaluation(10)]
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
            **p,               # todos los hiperparámetros, sin duplicar
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