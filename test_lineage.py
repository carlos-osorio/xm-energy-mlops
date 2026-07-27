import wandb
from src.lineage import log_lineage_to_wandb

# Inicializar un run de prueba en W&B
run = wandb.init(
    project="xm-energy-mlops",
    name="test-lineage-run",
    tags=["test"]
)

# Llamar a la función de linaje
log_lineage_to_wandb(run)

# Finalizar el run
run.finish()

print("Prueba de linaje completada. Revisa tu dashboard de W&B.")