import requests
import pandas as pd
from pathlib import Path

def ingest_data():
    # 1. Crear la carpeta de datos crudos si no existe
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. URL de un dataset público estable (series de tiempo) para probar la tubería
    # Más adelante la reemplazaremos por la API o CSV real de XM
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    
    # 3. Descargar, procesar y guardar
    print(f"Descargando datos desde la URL...")
    df = pd.read_csv(url)
    
    # Guardamos el archivo con un nombre descriptivo
    output_path = raw_dir / "sample_timeseries.csv"
    df.to_csv(output_path, index=False)
    print(f"Datos guardados exitosamente en: {output_path}")

if __name__ == "__main__":
    ingest_data()