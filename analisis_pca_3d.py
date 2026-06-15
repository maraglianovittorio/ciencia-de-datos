import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

def analisis_pca_3d():
    print("Iniciando análisis PCA 3D...")
    
    # 1. Cargar datos
    # vista_clientes_kmeans.csv y vista_clientes_kmeans_normalizada.csv ya NO incluyen los clientes premium.
    # Los clientes premium (ballenas > P95) fueron apartados en clientes_premium.csv
    
    df_norm = pd.read_csv('vistas_minables/vista_clientes_kmeans_normalizada.csv')
    df_crudo = pd.read_csv('vistas_minables/vista_clientes_kmeans.csv')
    
    numericas = ['edad', 'cant_vuelos', 'gasto_acumulado', 'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio']
    categoricas = ['sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra']
    # Mantenemos las columnas relevantes (tanto numéricas como categóricas)
    df_crudo = df_crudo[numericas + categoricas].copy()

    # Entrenar K-Medias (K=2 según el modelo final reportado en TP_Parte2_Clustering.md)
    k = 2
    print(f"Entrenando K-Medias con K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_norm)
    
    df_crudo['Cluster'] = clusters
    df_norm['Cluster'] = clusters
    
    # 2. Análisis de Componentes Principales (ACP) con 3 dimensiones
    print("Aplicando PCA con 3 componentes...")
    pca = PCA(n_components=3)
    componentes = pca.fit_transform(df_norm.drop(columns=['Cluster']))
    
    df_norm['PC1'] = componentes[:, 0]
    df_norm['PC2'] = componentes[:, 1]
    df_norm['PC3'] = componentes[:, 2]
    
    # Imprimir los pesos para cada componente
    pesos = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2', 'PC3'], index=numericas)
    print("--- PESOS PCA (3 Componentes) ---")
    print(pesos.round(4))
    print(f"Varianza Explicada Total (3 Componentes): {sum(pca.explained_variance_ratio_):.2%}")
    print("---------------------------------")
    
    # 3. Gráfico 3D Interactivo con Plotly
    print("Generando gráfico 3D interactivo...")
    out_dir = 'TPI/graficos/kmeans'
    os.makedirs(out_dir, exist_ok=True)
    
    # Creamos un dataframe temporal para plotly
    df_plotly = df_norm.copy()
    df_plotly['Cluster'] = df_plotly['Cluster'].astype(str) # Plotly maneja mejor colores discretos si es string
    
    import plotly.express as px
    fig_3d = px.scatter_3d(df_plotly, x='PC1', y='PC2', z='PC3',
                           color='Cluster', opacity=0.7,
                           title=f'K-Medias (K={k}) en 3D (ACP)<br>Varianza Total Explicada: {sum(pca.explained_variance_ratio_):.2%}',
                           color_discrete_sequence=px.colors.qualitative.Set1)
    
    fig_3d.update_traces(marker=dict(size=4))
    fig_3d.write_html(os.path.join(out_dir, 'kmeans_pca_3d_interactivo.html'))
    
    # Gráfico estático con Matplotlib (opcional, como backup)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_norm['PC1'], df_norm['PC2'], df_norm['PC3'], 
                         c=df_norm['Cluster'], cmap='Set1', s=20, alpha=0.6)
    
    ax.set_title(f'K-Medias (K={k}) en 3D (ACP)\nVarianza Total Explicada: {sum(pca.explained_variance_ratio_):.2%}')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'kmeans_pca_3d_estatico.png'))
    plt.close()
    
    # 4. Boxplots Globales (Sin dividir por cluster y sin clientes premium)
    print("Generando boxplots individuales globales (excluyendo clientes premium)...")
    
    out_dir_boxplots = 'TPI/graficos/kmeans/boxplots_individuales'
    os.makedirs(out_dir_boxplots, exist_ok=True)
    
    for col in numericas:
        plt.figure(figsize=(6, 8))
        # Boxplot global de la variable, sin separar por cluster
        sns.boxplot(y=df_crudo[col], color='skyblue')
        
        media_global = df_crudo[col].mean()
        plt.axhline(media_global, color='red', linestyle='--', linewidth=2, label='Media Global')
        
        plt.title(f'Distribución Global de {col}\n(Sin VIPs)')
        plt.ylabel(f'Valor Absoluto')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_boxplots, f'boxplot_global_{col}.png'))
        plt.close()
        
    # 5. Gráficas Comparativas de Distribución (Densidad/KDE) por Cluster
    print("Generando gráficas comparativas por cluster...")
    out_dir_comparativas = 'TPI/graficos/kmeans/comparativas_distribucion'
    os.makedirs(out_dir_comparativas, exist_ok=True)
    
    for col in numericas:
        plt.figure(figsize=(10, 6))
        # KDE plot para ver cómo se distribuye cada cluster suavemente
        sns.kdeplot(data=df_crudo, x=col, hue='Cluster', fill=True, common_norm=False, palette='Set1', alpha=0.5)
        
        plt.title(f'Comparativa de Distribución por Cluster: {col}')
        plt.xlabel(f'Valor de {col}')
        plt.ylabel('Densidad')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_comparativas, f'distribucion_cluster_{col}.png'))
        plt.close()
        
    # 6. Gráficas Comparativas para Variables Cualitativas
    print("Generando gráficas comparativas para variables cualitativas...")
    out_dir_cualitativas = 'TPI/graficos/kmeans/comparativas_cualitativas'
    os.makedirs(out_dir_cualitativas, exist_ok=True)
    
    for col in categoricas:
        plt.figure(figsize=(12, 8))
        # Utilizamos countplot horizontal y ordenamos por las categorías más frecuentes
        sns.countplot(data=df_crudo, y=col, hue='Cluster', palette='Set1', 
                      order=df_crudo[col].value_counts().index)
        
        plt.title(f'Comparativa de Cluster por {col}')
        plt.xlabel('Cantidad de Clientes')
        plt.ylabel(col)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_cualitativas, f'comparativa_cualitativa_{col}.png'))
        plt.close()

    print(f"Gráficos guardados exitosamente en {out_dir}, {out_dir_boxplots}, {out_dir_comparativas} y {out_dir_cualitativas}")

if __name__ == '__main__':
    analisis_pca_3d()
