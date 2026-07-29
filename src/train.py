import pandas as pd
import numpy as np
import lightgbm as lgb
import yaml
from pathlib import Path
from datetime import date
from sklearn.metrics import mean_squared_error
from src import registry
from src.lineage import get_git_commit, get_dvc_hash
from src.features import make_supervised, chronological_split

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
    
    # 2. Crear features de lag con la lógica compartida (ver src/features.py).
    #    lag_1 = hoy, lag_2 = ayer, ..., target = día siguiente.
    lookback = p["lookback_days"]
    X, y = make_supervised(df, lookback)
    X_train, X_val, y_train, y_val = chronological_split(X, y, p["validation_days"])

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
    
    # 6. Guardar el modelo candidato
    registry.CANDIDATE_MODEL.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(registry.CANDIDATE_MODEL))
    print(f"💾 Modelo candidato guardado en: {registry.CANDIDATE_MODEL}")

    # 7. Registrar métricas y trazabilidad del candidato
    metrics = {
        "rmse": rmse,
        "trained_at": date.today().isoformat(),
        "git_sha": get_git_commit(),
        "data_hash": get_dvc_hash(),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "params": p,
    }
    registry.write_json(registry.CANDIDATE_METRICS, metrics)
    print(f"📝 Métricas del candidato guardadas en: {registry.CANDIDATE_METRICS}")
    print("✅ Entrenamiento completado.")

if __name__ == "__main__":
    train_model()