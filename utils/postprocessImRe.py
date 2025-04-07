import numpy as np
import pandas as pd


def compute_and_export_fourier(receiver_csv_path, ricker_csv_path, freq, 
                               receiver_output_csv='LayerSim/receiver_fourier.csv', 
                               ricker_output_csv='LayerSim/ricker_fourier.csv'):
    """
    Lee la información de dos CSV: uno con las señales de 100 geófonos y otro con la señal
    de una ricker wavelet, calcula la magnitud de la transformada de Fourier a una frecuencia dada y 
    exporta los resultados a dos archivos CSV.
    
    Parámetros:
      receiver_csv_path: str
          Ruta del CSV con las señales de los receptores. Se asume que la primera columna es tiempo 
          y las columnas 2 a 101 contienen las señales de los geófonos.
      ricker_csv_path: str
          Ruta del CSV con la señal de la ricker wavelet. Se asume que la primera columna es tiempo y 
          la segunda la señal.
      freq: float
          Frecuencia a la que se desea calcular la transformada de Fourier.
      receiver_output_csv: str, opcional
          Nombre del CSV de salida para los valores calculados en los receptores (por defecto 'receiver_fourier.csv').
      ricker_output_csv: str, opcional
          Nombre del CSV de salida para el valor calculado en la ricker wavelet (por defecto 'ricker_fourier.csv').
    
    Retorna:
      df_receiver_ft: DataFrame con las magnitudes de la transformada para cada geófono.
      df_ricker_ft: DataFrame con la magnitud de la transformada para la ricker wavelet.
    """
    
    # 1) Leer el CSV de señales de los receptores
    receiver_df = pd.read_csv(receiver_csv_path)
    # Se asume que la primera columna es tiempo y las siguientes las señales
    tiempo = receiver_df.iloc[:, 0].values
    
    # Calcular la transformada de Fourier para cada geófono (columnas 2 a 101)
    receiver_fourier_magnitude = []
    reciever_fourier_real = []
    reciever_fourier_im = []
    for col in receiver_df.columns[1:]:
        señal = receiver_df[col].values
        # Cálculo de la FT en la frecuencia deseada: ∫ s(t)*exp(-i 2π f t) dt
        ft_val = np.trapz(señal * np.exp(-1j * 2 * np.pi * freq * tiempo), tiempo)
        # Se guarda la magnitud
        receiver_fourier_magnitude.append(np.abs(ft_val))
        reciever_fourier_real.append(ft_val.real)
        reciever_fourier_im.append(ft_val.imag)
    
    # Crear un DataFrame con los resultados para los receptores
    df_receiver_ft = pd.DataFrame({
        'Geofono': np.arange(1, len(receiver_fourier_magnitude) + 1),
        'Magnitud_Fourier': receiver_fourier_magnitude,
        'Real Fourier': reciever_fourier_real,
        'Imag Fourier': reciever_fourier_im
    })
    
    # 2) Leer el CSV de la ricker wavelet
    ricker_df = pd.read_csv(ricker_csv_path)
    tiempo_ricker = ricker_df.iloc[:, 0].values
    señal_ricker = ricker_df.iloc[:, 1].values
    
    # Calcular la transformada de Fourier para la ricker wavelet
    ricker_ft_val = np.trapz(señal_ricker * np.exp(-1j * 2 * np.pi * freq * tiempo_ricker), tiempo_ricker)
    ricker_magnitude = np.abs(ricker_ft_val)
    ricker_real = ricker_ft_val.real
    ricker_complex = ricker_ft_val.imag
    
    # Crear un DataFrame para el resultado de la ricker wavelet
    df_ricker_ft = pd.DataFrame({
        'Magnitud_Fourier': [ricker_magnitude],
        'Real Fourier': [ricker_real],
        'Img Fourier': [ricker_complex]
    })
    
    # 3) Exportar los resultados a CSV
    df_receiver_ft.to_csv(receiver_output_csv, index=False)
    df_ricker_ft.to_csv(ricker_output_csv, index=False)
    
    return df_receiver_ft, df_ricker_ft

# Ejemplo de uso:
if __name__ == "__main__":
    # Definir la frecuencia deseada, por ejemplo 5 Hz
    frecuencia = 0.02
    # Llamar a la función con las rutas de tus archivos CSV
    receiver_ft, ricker_ft = compute_and_export_fourier(receiver_csv_path="LayerSim/geofonos_layer.csv",ricker_csv_path="LayerSim/ricker_layers.csv", freq=frecuencia)
    print("Transformada de Fourier calculada y exportada a CSV.")
