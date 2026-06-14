import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def experimento_kmeans():
    out_dir = '/home/vitto/.gemini/antigravity-ide/brain/f193a960-98e2-496f-85ba-16a7821e5e77/'
    
    # 1. Cargar datos
    # Datos normalizados (para entrenar matemáticamente K-Medias y ACP sin sesgo de escala)
    df_norm = pd.read_csv('/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/vista_clientes_kmeans_normalizada.csv')
    
    # Datos crudos (para poder ver los valores reales como edad o $ en los boxplots)
    df_crudo = pd.read_csv('/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/vista_clientes_kmeans.csv')
    numericas = ['edad', 'cant_vuelos', 'gasto_acumulado', 'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio']
    df_crudo = df_crudo[numericas].copy()

    # 2. Entrenar K-Medias con K=3 (Experimento)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_norm)
    
    df_crudo['Cluster'] = clusters
    df_norm['Cluster'] = clusters
    
    # 3. Análisis de Componentes Principales (ACP)
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(df_norm.drop(columns=['Cluster']))
    
    # Imprimir los pesos para que el LLM pueda explicar qué variables dominan cada componente
    pesos = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=numericas)
    print("---PESOS ACP---")
    print(pesos.to_json(orient='index'))
    print("---END PESOS ACP---")

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=componentes[:, 0], y=componentes[:, 1], hue=clusters, palette='Set1', s=50, alpha=0.6)
    plt.title(f'K-Medias (K=3) en 2D (ACP)\nVarianza Total Explicada: {sum(pca.explained_variance_ratio_):.2%}')
    plt.xlabel(f'Componente 1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'Componente 2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend(title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(out_dir, 'kmeans_k3_pca.png'))
    plt.close()

    # 4. Boxplots de Datos No Normalizados
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(numericas):
        # Dibujamos las cajas reales
        sns.boxplot(x='Cluster', y=col, data=df_crudo, ax=axes[i], palette='Set2')
        
        # Marcamos la media global del dataset como línea punteada roja de referencia
        media_global = df_crudo[col].mean()
        axes[i].axhline(media_global, color='red', linestyle='--', linewidth=2, label='Media Global')
        
        axes[i].set_title(f'Distribución Real de {col}')
        axes[i].set_ylabel(f'Valor Absoluto')
        if i == 0:
            axes[i].legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kmeans_k3_boxplots_reales.png'))
    plt.close()

    # Imprimir medias por cluster para armar la tabla
    medias_cluster = df_crudo.groupby('Cluster').mean()
    print("---MEDIAS POR CLUSTER---")
    print(medias_cluster.to_json(orient='index'))
    print("---END MEDIAS POR CLUSTER---")

if __name__ == '__main__':
    experimento_kmeans()
