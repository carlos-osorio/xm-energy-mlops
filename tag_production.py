import wandb

api = wandb.Api()

entity = api.default_entity
project = "xm-energy-mlops"
target_run_name = "lgbm-mvp-training"

print(f"🔍 Buscando en: {entity}/{project}")

try:
    runs = api.runs(f"{entity}/{project}")
    
    for run in runs:
        if run.name == target_run_name or target_run_name in run.name:
            print(f"✅ Run encontrado: '{run.name}' (ID: {run.id})")
            
            # Buscar artefactos que este run haya guardado (logged)
            artifacts = run.logged_artifacts()
            found_artifact = False
            
            for artifact in artifacts:
                print(f"  -> Artefacto encontrado: {artifact.name} (tipo: {artifact.type})")
                
                if "lgbm-mvp-model" in artifact.name:
                    if "production" not in artifact.aliases:
                        artifact.aliases.append("production")
                        artifact.save()
                        print("🚀 ¡Modelo marcado exitosamente como 'production'!")
                    else:
                        print("ℹ️ El modelo ya tenía el alias 'production'.")
                    found_artifact = True
                    break  # Salimos del loop de artefactos
            
            if not found_artifact:
                print("⚠️ No se encontró ningún artefacto con 'lgbm-mvp-model' en este run.")
            
            break  # Salimos del loop de runs una vez encontramos el correcto

except Exception as e:
    print(f"❌ Error: {e}")