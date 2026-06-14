import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np
import os

def aplicar_kmeans():
    print("Iniciando clustering K-Medias (K=2)...")
    
    file_path = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/vista_clientes_kmeans_normalizada.csv'
    df = pd.read_csv(file_path)
    
    # 1. Instanciar y Entrenar K-Means (K=2 como dictaminó el negocio)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(df)
    
    df_con_clusters = df.copy()
    df_con_clusters['Cluster'] = cluster_labels
    
    # 2. Evaluación del Modelo (Coeficiente de Silhouette)
    # Usamos una muestra de 5000 para evitar que la matriz de distancias O(N^2) tarde una eternidad.
    sil_score = silhouette_score(df, cluster_labels, sample_size=5000, random_state=42)
    print(f"Coeficiente de Silhouette (estimado 5k): {sil_score:.4f}")
    
    # Directorio de salida para los gráficos de evaluación (artifact dir)
    out_dir = '/home/vitto/.gemini/antigravity-ide/brain/f193a960-98e2-496f-85ba-16a7821e5e77/'
    
    # 3. Gráficos de Evaluación por Variable (Boxplots)
    # Ya que descartamos PCA, analizamos cada dimensión por separado cruzándola contra el cluster.
    columnas_cuantitativas = df.columns
    
    # Creamos un gráfico múltiple para ver cómo se divide cada grupo en cada variable original
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(columnas_cuantitativas):
        sns.boxplot(x='Cluster', y=col, data=df_con_clusters, ax=axes[i], palette='Set2')
        axes[i].set_title(f'Distribución de {col}')
        axes[i].set_ylabel('Valor (Z-Score)')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kmeans_boxplots_variables.png'))
    plt.close()

    # 4. Guardar resultados
    # Exportamos el dataset con la etiqueta del cluster para la fase productiva.
    resultados_path = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/clientes_con_clusters.csv'
    df_con_clusters.to_csv(resultados_path, index=False)
    
    print(f"Dataset con clusters asignados guardado en {resultados_path}")
    print("Boxplots de evaluación generados exitosamente.")

if __name__ == '__main__':
    aplicar_kmeans()
