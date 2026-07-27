import os
import yaml
from pathlib import Path

def get_git_commit():
    """Extrae el hash del commit actual de Git desde la variable de entorno de GitHub Actions."""
    # En GitHub Actions, esta variable existe automáticamente
    # En local, la simularemos con otra variable de entorno
    return os.environ.get('GITHUB_SHA', 'local-dev-commit')

def get_dvc_hash():
    """Extrae el hash de DVC desde la variable de entorno inyectada por GitHub Actions."""
    # En GitHub Actions, inyectaremos este hash con yq
    # En local, lo leeremos directamente del archivo dvc.lock
    dvc_hash = os.environ.get('DVC_HASH')
    
    if dvc_hash:
        return dvc_hash
    
    # Fallback para desarrollo local: leer dvc.lock directamente
    dvc_lock_path = Path('dvc.lock')
    if dvc_lock_path.exists():
        with open(dvc_lock_path, 'r') as f:
            dvc_lock = yaml.safe_load(f)
        # Navegar a la estructura del stage de datos
        # Asumimos que el stage se llama 'prepare_data'
        if 'stages' in dvc_lock:
            for stage_name, stage_data in dvc_lock['stages'].items():
                if 'outs' in stage_data and len(stage_data['outs']) > 0:
                    return stage_data['outs'][0].get('md5', 'unknown')
    
    return 'unknown'

def log_lineage_to_wandb(run):
    """Registra los hashes de Git y DVC en el run de W&B."""
    git_commit = get_git_commit()
    dvc_hash = get_dvc_hash()
    
    # Registrar como configuración inmutable del run
    run.config.update({
        'git_commit': git_commit,
        'dvc_data_hash': dvc_hash
    })
    
    print(f"Linaje registrado - Git: {git_commit}, DVC: {dvc_hash}")