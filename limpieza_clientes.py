import pandas as pd
import os

def limpiar_dataset_clientes():
    print("Iniciando limpieza del dataset Clientes...")
    file_path = '/home/vitto/Documents/GitHub/ciencia-de-datos/Clientes.xlsx'
    
    # 1. Cargar el Dataset
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error cargando el archivo Excel: {e}")
        return

    # Validar que el directorio de salida exista
    out_dir = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables'
    os.makedirs(out_dir, exist_ok=True)

    # 2. Detección de Whales (Ballenas) - Percentil 95
    # Calculamos el límite (P95) para las variables con cola derecha crítica
    p95_gasto = df['gasto_acumulado'].quantile(0.95)
    p95_ingreso = df['ingreso_mensual'].quantile(0.95)
    p95_millas = df['cantidad_millas'].quantile(0.95)

    print(f"Límites P95 detectados:")
    print(f"- Gasto Acumulado: {p95_gasto:.2f}")
    print(f"- Ingreso Mensual: {p95_ingreso:.2f}")
    print(f"- Cantidad Millas: {p95_millas:.2f}")

    # Condición: si supera el P95 en CUALQUIERA de las 3, es ballena.
    condicion_ballena = (
        (df['gasto_acumulado'] >= p95_gasto) |
        (df['ingreso_mensual'] >= p95_ingreso) |
        (df['cantidad_millas'] >= p95_millas)
    )

    df_premium = df[condicion_ballena].copy()
    df_normal = df[~condicion_ballena].copy()

    print(f"\nClientes totales originales: {len(df)}")
    print(f"Clientes Premium (Whales) aislados: {len(df_premium)}")
    print(f"Clientes Normales (K-Medias): {len(df_normal)}")

    # Guardar Premium
    premium_path = os.path.join(out_dir, 'clientes_premium.csv')
    df_premium.to_csv(premium_path, index=False)
    print(f"-> Guardado clientes_premium.csv exitosamente en {premium_path}")

    # 3. Limpieza del Dataset General
    # Eliminar Multicolinealidad y el ID
    columnas_a_eliminar = ['gasto_acumulado_extra', 'id_cliente']
    
    # Dropear asegurando que existen
    columnas_existentes = [col for col in columnas_a_eliminar if col in df_normal.columns]
    df_normal_limpio = df_normal.drop(columns=columnas_existentes)
    
    print(f"\nColumnas eliminadas por limpieza en vista K-Medias: {columnas_existentes}")

    # 4. Exportación Final
    vista_path = os.path.join(out_dir, 'vista_clientes_kmeans.csv')
    df_normal_limpio.to_csv(vista_path, index=False)
    print(f"-> Guardado vista_clientes_kmeans.csv exitosamente en {vista_path}")

    print("\nProceso de limpieza completado de forma impecable.")

if __name__ == '__main__':
    limpiar_dataset_clientes()
