import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def normalizar_clientes():
    print("Iniciando normalización de vista_clientes_kmeans.csv para K-Medias...")
    file_path = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/vista_clientes_kmeans.csv'
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error cargando el archivo: {e}")
        return

    # 1. Separar variables categóricas y numéricas
    categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numericas = df.select_dtypes(include=['number']).columns.tolist()

    print(f"Variables Categóricas detectadas ({len(categoricas)}): {categoricas}")
    print(f"Variables Numéricas detectadas ({len(numericas)}): {numericas}")

    # 2. Eliminar variables categóricas
    # Para K-Medias clásico, las distancias euclidianas pierden sentido lógico en variables 
    # categóricas codificadas. Nos quedamos solo con las cuantitativas continuas.
    df_numerico = df[numericas].copy()
    print(f"\nVariables categóricas eliminadas. Nos quedamos con {df_numerico.shape[1]} columnas puramente cuantitativas.")

    # 3. Normalización con StandardScaler
    # K-Medias asume distribuciones normales y esferas geométricas.
    scaler = StandardScaler()
    
    columnas_finales = df_numerico.columns
    df_escalado = pd.DataFrame(scaler.fit_transform(df_numerico), columns=columnas_finales)

    # 4. Exportar
    out_dir = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables'
    out_path = os.path.join(out_dir, 'vista_clientes_kmeans_normalizada.csv')
    
    df_escalado.to_csv(out_path, index=False)
    
    print(f"\n-> Dataset normalizado guardado exitosamente en: {out_path}")
    print("¡El dataset está 100% puro, escalado y listo para instanciar KMeans!")

if __name__ == '__main__':
    normalizar_clientes()
