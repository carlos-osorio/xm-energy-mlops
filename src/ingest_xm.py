import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from pydataxm.pydatasimem import ReadSIMEM

def download_xm_data():
    print(" Descargando datos de Precio de Bolsa desde SIMEM...")
    
    try:
        # Dataset ID oficial para Precio de Bolsa Nacional
        dataset_id = 'EC6945'
        
        # Fechas: últimos 2 años
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=2*365)).strftime("%Y-%m-%d")
        
        print(f"📅 Solicitando datos desde {start_date} hasta {end_date}...")
        
        # Descargar datos horarios
        simem = ReadSIMEM(dataset_id=dataset_id, start_date=start_date, end_date=end_date)
        df_hourly = simem.main()
        
        if df_hourly is not None and not df_hourly.empty:
            # Filtrar Precio de Bolsa Nacional
            df_hourly = df_hourly[df_hourly['CodigoVariable'] == 'PB_Nal'].copy()
            
            # Renombrar columnas
            df_hourly = df_hourly.rename(columns={'FechaHora': 'FechaHora', 'Valor': 'Precio'})
            
            # Asegurar tipos
            df_hourly['FechaHora'] = pd.to_datetime(df_hourly['FechaHora'])
            df_hourly['Precio'] = pd.to_numeric(df_hourly['Precio'], errors='coerce')
            df_hourly = df_hourly.dropna(subset=['Precio'])
            
            # ============================================
            # AGREGACIÓN A NIVEL DIARIO
            # ============================================
            # Extraer solo la fecha (sin hora)
            df_hourly['Fecha'] = df_hourly['FechaHora'].dt.date
            
            # Calcular el precio promedio diario
            df_daily = df_hourly.groupby('Fecha').agg({
                'Precio': 'mean'  # Promedio diario
            }).reset_index()
            
            # Renombrar para claridad
            df_daily = df_daily.rename(columns={'Precio': 'PrecioPromedio'})
            
            # También podrías calcular otras métricas si las necesitas:
            # - Precio máximo del día: df_hourly.groupby('Fecha')['Precio'].max()
            # - Precio mínimo del día: df_hourly.groupby('Fecha')['Precio'].min()
            # - Precio pico (hora 18-22): filtrar por hora y promediar
            
            # Ordenar y limpiar
            df_daily = df_daily.sort_values('Fecha').reset_index(drop=True)
            
            # Guardar datos diarios
            output_dir = Path("data/raw")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "price_xm_daily.csv"
            
            df_daily.to_csv(output_path, index=False)
            print(f"✅ Datos diarios descargados y agregados: {len(df_daily)} registros")
            print(f"💾 Guardado en: {output_path}")
            print(f" Rango de fechas: {df_daily['Fecha'].min()} a {df_daily['Fecha'].max()}")
            print(f"📈 Precio promedio general: ${df_daily['PrecioPromedio'].mean():.2f} $/MWh")
            
            # Opcional: También guardar los datos horarios si los quieres tener
            hourly_path = output_dir / "price_xm_hourly.csv"
            df_hourly.to_csv(hourly_path, index=False)
            print(f" Datos horarios también guardados en: {hourly_path}")
            
            return df_daily
        else:
            print("⚠️ No se encontraron datos en la respuesta de SIMEM")
            return None
            
    except Exception as e:
        print(f" Error al descargar datos de XM: {e}")
        return None

if __name__ == "__main__":
    download_xm_data()