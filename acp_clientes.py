import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
print("Cargando dataset...")
df = pd.read_excel('Clientes.xlsx')

# Limpiar columnas innecesarias y dejar solo las clave
columnas_relevantes = ['ingreso_mensual', 'cant_vuelos', 'gasto_acumulado']
df_numeric = df[columnas_relevantes].copy()

print(f"Columnas utilizadas para el ACP: {df_numeric.columns.tolist()}")

# Manejar valores nulos si los hay (imputar con la media)
df_numeric = df_numeric.fillna(df_numeric.mean())

# Estandarizar los datos
print("Estandarizando los datos...")
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Realizar el ACP (PCA) - Buscamos los 2 componentes principales
print("Ejecutando ACP (2 componentes)...")
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Crear un DataFrame con los resultados
n_components = pca_result.shape[1]
pca_columns = [f'PC{i+1}' for i in range(n_components)]
df_pca = pd.DataFrame(data=pca_result, columns=pca_columns)

# Si queremos mantener el ID para cruzarlo después, lo agregamos (si existe)
if 'id_cliente' in df.columns:
    df_pca.insert(0, 'id_cliente', df['id_cliente'])

# Guardar los resultados
output_file = 'clientes_pca_resultados.csv'
df_pca.to_csv(output_file, index=False)
print(f"Resultados guardados exitosamente en: {output_file}")

# Mostrar la varianza explicada por cada componente principal
explained_variance = pca.explained_variance_ratio_
for i, var in enumerate(explained_variance):
    print(f"Varianza explicada por PC{i+1}: {var:.4f}")
