import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Configurar estilo
sns.set_theme(style="whitegrid")

print("Cargando resultados del ACP...")
df_pca = pd.read_csv('clientes_pca_resultados.csv')

# Extraer solo las variables de componentes principales para el clustering
X = df_pca[['PC1', 'PC2']]

print("Ejecutando K-Means con k=2...")
# K-Means con k=2
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df_pca['cluster'] = kmeans.fit_predict(X)

# Asignar nombres más amigables a los clusters (dependiendo de sus centroides)
# Vamos a ver cuál cluster tiene el centroide con mayor PC1 (mayor actividad)
centroids = kmeans.cluster_centers_
if centroids[0][0] > centroids[1][0]:
    cluster_nombres = {0: 'VIP / Alta Actividad', 1: 'Estándar'}
else:
    cluster_nombres = {0: 'Estándar', 1: 'VIP / Alta Actividad'}

df_pca['perfil_cluster'] = df_pca['cluster'].map(cluster_nombres)

print("Generando gráfico de los clusters...")
plt.figure(figsize=(10, 8))

# Limites para el zoom (excluyendo el 0.1% de outliers de cada lado)
x_min, x_max = np.percentile(df_pca['PC1'], [0.1, 99.9])
y_min, y_max = np.percentile(df_pca['PC2'], [0.1, 99.9])

# Graficar los puntos
sns.scatterplot(
    data=df_pca, 
    x='PC1', 
    y='PC2', 
    hue='perfil_cluster', 
    alpha=0.5, 
    s=15,
    palette=['#e74c3c', '#3498db'] # Rojo y Azul
)

# Graficar los centroides
plt.scatter(
    centroids[:, 0], 
    centroids[:, 1], 
    c='black', 
    s=200, 
    marker='X', 
    label='Centroides'
)

plt.xlabel('PC1 (Nivel de Actividad)')
plt.ylabel('PC2 (Ingreso vs Consumo)')
plt.title('Clusters de Clientes (K-Means, k=2)')
plt.xlim(x_min - 0.5, x_max + 0.5)
plt.ylim(y_min - 0.5, y_max + 0.5)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
plt.close()

# Cruzar con Clientes.xlsx
print("Cruzando clusters con datos originales...")
df_original = pd.read_excel('Clientes.xlsx')
df_final = pd.merge(df_original, df_pca[['id_cliente', 'cluster', 'perfil_cluster']], on='id_cliente', how='left')

output_file = 'Clientes_Clustered.xlsx'
df_final.to_excel(output_file, index=False)
print(f"Datos guardados exitosamente en: {output_file}")

# Resumen de los clusters
print("\nResumen de Clusters:")
print(df_final.groupby('perfil_cluster')[['ingreso_mensual', 'cant_vuelos', 'gasto_acumulado']].mean().round(2))
print("\nCantidad de clientes por cluster:")
print(df_final['perfil_cluster'].value_counts())
