import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

def generar_entregables():
    print("Iniciando generación de entregables para el TP...")
    out_dir = '/home/vitto/Documents/GitHub/ciencia-de-datos/graficos'
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Cargar datasets
    # Original para obtener los IDs
    df_original = pd.read_excel('/home/vitto/Documents/GitHub/ciencia-de-datos/Clientes.xlsx')
    
    # Replicar filtro P95 para mantener el orden exacto
    p95_gasto = df_original['gasto_acumulado'].quantile(0.95)
    p95_ingreso = df_original['ingreso_mensual'].quantile(0.95)
    p95_millas = df_original['cantidad_millas'].quantile(0.95)

    condicion_ballena = (
        (df_original['gasto_acumulado'] >= p95_gasto) |
        (df_original['ingreso_mensual'] >= p95_ingreso) |
        (df_original['cantidad_millas'] >= p95_millas)
    )

    df_premium_recreado = df_original[condicion_ballena].copy()
    df_normal_recreado = df_original[~condicion_ballena].copy()

    # Cargar etiquetas K-Means (K=2) desde el CSV de resultados normalizados
    df_clusters = pd.read_csv('/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/clientes_con_clusters.csv')
    df_normal_recreado['Cluster_KMeans'] = df_clusters['Cluster'].values

    # 2. Generar CSV Final Unificado (Juntando VIP con Cluster 0)
    # Por instrucción de negocio, los VIP pasan a ser Cluster 0 (Alto Valor)
    df_premium_recreado['Cluster_KMeans'] = 0

    df_final = pd.concat([df_normal_recreado, df_premium_recreado]).sort_index()
    
    csv_out_path = '/home/vitto/Documents/GitHub/ciencia-de-datos/vistas_minables/clientes_etiquetados_final.csv'
    df_final[['id_cliente', 'Cluster_KMeans']].to_csv(csv_out_path, index=False)
    print(f"-> CSV Final guardado con 0 y 1 en: {csv_out_path}")

    # 3. Generar Gráfico PCA Unificado
    # Entrenar Scaler y PCA SOLO con clientes normales (para mantener la consistencia del modelo original)
    numericas = ['edad', 'cant_vuelos', 'gasto_acumulado', 'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio']
    
    scaler = StandardScaler()
    scaler.fit(df_normal_recreado[numericas])
    
    pca = PCA(n_components=2)
    # Transformamos normales
    norm_scaled = scaler.transform(df_normal_recreado[numericas])
    componentes_normales = pca.fit_transform(norm_scaled)
    
    # Transformamos VIPs proyectándolos sobre el mismo espacio de los normales
    vip_scaled = scaler.transform(df_premium_recreado[numericas])
    componentes_vips = pca.transform(vip_scaled)

    # Preparar el scatterplot
    plt.figure(figsize=(12, 8))
    
    # Cluster 1 (Estándar)
    mask_c1 = df_normal_recreado['Cluster_KMeans'] == 1
    plt.scatter(componentes_normales[mask_c1, 0], componentes_normales[mask_c1, 1], 
                alpha=0.3, label='Cluster 1 (Estándar)', color='dodgerblue', s=20)
                
    # Cluster 0 (Alto Valor - KMedias)
    mask_c0 = df_normal_recreado['Cluster_KMeans'] == 0
    plt.scatter(componentes_normales[mask_c0, 0], componentes_normales[mask_c0, 1], 
                alpha=0.5, label='Cluster 0 (K-Medias Alto Valor)', color='crimson', s=20)
                
    # VIPs (Se plotean con un marcador distinto para que se note en el gráfico, aunque sean del grupo 0)
    plt.scatter(componentes_vips[:, 0], componentes_vips[:, 1], 
                alpha=0.7, label='Cluster 0 (Segmento VIP / Ballenas)', color='gold', marker='*', s=80, edgecolor='black')

    plt.title('Proyección PCA de Clusters (K-Medias + VIP Segment)')
    plt.xlabel(f'Componente Principal 1 (Volumen) - {pca.explained_variance_ratio_[0]:.2%} varianza')
    plt.ylabel(f'Componente Principal 2 (Demografía) - {pca.explained_variance_ratio_[1]:.2%} varianza')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plot_out = os.path.join(out_dir, 'kmeans_pca_scatter.png')
    plt.savefig(plot_out)
    plt.close()
    
    print(f"-> Scatterplot unificado guardado en: {plot_out}")
    print("Entregables técnicos finalizados con éxito.")

if __name__ == '__main__':
    generar_entregables()
