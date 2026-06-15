"""
================================================================================
Clustering Jerarquico (2 Clusters)
================================================================================
- Clustering: sobre vista_clientes_kmeans_normalizada (datos ya estandarizados)
- CSV + graficos originales: desde vista_clientes_kmeans.csv (sin normalizar,
  mismo orden que la vista normalizada, alineado por posicion)
================================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import clustering_utils as cu


OUTPUT_DIR = "TPI/graficos/jerarquico"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITMO_NOMBRE = "agglomerative_ward"
N_CLUSTERS = 2
SAMPLE_SILHOUETTE = 5000

RUTA_VISTA_NORM = "vistas_minables/vista_clientes_kmeans_normalizada.csv"
RUTA_VISTA_ORIG = "vistas_minables/vista_clientes_kmeans.csv"

NUMERIC_COLS = [
    'edad', 'cant_vuelos', 'gasto_acumulado',
    'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio'
]


def cargar_vista(pb):
    pb.desc = "Cargando vista normalizada"
    df = pd.read_csv(RUTA_VISTA_NORM)
    print(f"      Vista cargada: {df.shape[0]} filas x {df.shape[1]} columnas")
    pb.update(100)
    return df


def clusterizar_vista(df, pb):
    pb.desc = "Clusterizando vista"
    x = df.values
    modelo = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='ward')
    cluster_labels = modelo.fit_predict(x)
    pb.update(50)

    n = len(df)
    if n > SAMPLE_SILHOUETTE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SAMPLE_SILHOUETTE, replace=False)
        sil = silhouette_score(x[idx], cluster_labels[idx])
        sil_sample = SAMPLE_SILHOUETTE
    else:
        sil = silhouette_score(x, cluster_labels)
        sil_sample = None
    pb.update(50)

    print("\n      Clustering completado (2 clusters).")
    cu.imprimir_tamanios_clusters(pd.DataFrame({'cluster': cluster_labels}))
    cu.imprimir_silhouette(sil, ALGORITMO_NOMBRE, sample_size=sil_sample)

    return cluster_labels, sil, sil_sample


def obtener_originales_de_vista(cluster_labels):
    """
    Lee la vista SIN normalizar (mismo orden que la normalizada).
    Los labels quedan alineados por posicion, sin necesidad de join por id.
    """
    print("\n[2/4] Leyendo vista sin normalizar para graficos...")

    df_original = pd.read_csv(RUTA_VISTA_ORIG)

    print(f"      Clientes en la vista: {len(df_original)} | clusters asignados: {len(cluster_labels)}")
    assert len(df_original) == len(cluster_labels), (
        f"Desajuste: {len(df_original)} filas vs {len(cluster_labels)} labels."
    )

    ids = df_original['id_cliente'].values if 'id_cliente' in df_original.columns else np.arange(len(df_original))
    df_num = df_original[NUMERIC_COLS].copy()

    for c in NUMERIC_COLS:
        if df_num[c].isnull().any():
            df_num[c] = df_num[c].fillna(df_num[c].median())

    scaler = StandardScaler()
    scaler.fit(df_num)

    return ids, df_num, cluster_labels, scaler


def exportar_csv(ids, cluster_labels, pb):
    pb.desc = "Exportando CSV"
    out = pd.DataFrame({'id_cliente': ids, 'cluster': cluster_labels})
    ruta = "clientes_clusters_jerarquico.csv"
    out.to_csv(ruta, index=False)
    pb.update(100)
    print(f"      [OK] {ruta} ({len(out)} filas)")


def graficar_dendrograma(scaler, df_original, pb):
    pb.desc = "Dendrograma"
    x_scaled = scaler.transform(df_original)
    n = len(df_original)

    SAMPLE_DENDRO = min(10000, n)
    if n > SAMPLE_DENDRO:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SAMPLE_DENDRO, replace=False)
        x_dendro = x_scaled[idx]
    else:
        x_dendro = x_scaled

    z = linkage(x_dendro, method='ward')
    pb.update(50)

    # Cada hoja representa al menos ~500 observaciones
    p_hojas = max(2, SAMPLE_DENDRO // 500)

    fig, ax = plt.subplots(figsize=(20, 9))
    dendrogram(
        z,
        truncate_mode='lastp',
        p=p_hojas,
        leaf_rotation=90.,
        leaf_font_size=10.,
        show_contracted=True,
        show_leaf_counts=True,
        ax=ax
    )
    pb.update(30)

    corte_y = z[-(N_CLUSTERS - 1), 2]
    ax.axhline(y=corte_y, color='red', linestyle='--', linewidth=1.8,
               label=f'Corte k={N_CLUSTERS} (d={corte_y:.2f})')

    ax.set_title(
        f'Dendrograma - Ward  |  muestra={SAMPLE_DENDRO:,} obs  |  '
        f'cada hoja ≥ ~{SAMPLE_DENDRO // p_hojas:,} obs',
        fontsize=14, weight='bold', pad=15
    )
    ax.set_xlabel('Hojas (cada etiqueta muestra cantidad de obs agrupadas)', fontsize=11)
    ax.set_ylabel('Distancia Ward', fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()

    ruta = f"{OUTPUT_DIR}/dendrograma.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    pb.update(20)
    print(f"      [OK] Dendrograma guardado en: {ruta}  (muestra={SAMPLE_DENDRO:,}, hojas={p_hojas})")


def graficar_boxplots(df_original, cluster_labels, pb):
    pb.desc = "Boxplots originales"
    df = df_original.copy()
    df['cluster'] = cluster_labels
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(x='cluster', y=col, data=df, ax=axes[i],
                    palette=['#4A90E2', '#FF6B6B'])
        axes[i].set_title(col)
        axes[i].set_xlabel('Cluster')
        pb.update(100 // n)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Variables numericas (valores originales) por Cluster',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    ruta = f"{OUTPUT_DIR}/boxplots_variables_originales.png"
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Boxplots guardados en: {ruta}")


def graficar_dashboard(df_original, cluster_labels, pb):
    pb.desc = "Dashboard KDE"
    df = df_original.copy()
    df['cluster'] = cluster_labels
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        c0 = df[df['cluster'] == 0]
        c1 = df[df['cluster'] == 1]
        sns.kdeplot(data=c0, x=col, fill=True, alpha=0.2,
                    color='#4A90E2', label='Cluster 0', linewidth=2, ax=axes[i])
        sns.kdeplot(data=c1, x=col, fill=True, alpha=0.2,
                    color='#FF6B6B', label='Cluster 1', linewidth=2, ax=axes[i])
        sns.kdeplot(data=df, x=col, fill=False, color='#95A5A6',
                    label='Total (vista)', linewidth=2.5, linestyle='--', ax=axes[i])
        m0 = c0[col].mean() if len(c0) else 0
        m1 = c1[col].mean() if len(c1) else 0
        mt = df[col].mean()
        axes[i].axvline(m0, color='#4A90E2', linestyle=':', linewidth=2, label=f'Media C0: {m0:.1f}')
        axes[i].axvline(m1, color='#FF6B6B', linestyle=':', linewidth=2, label=f'Media C1: {m1:.1f}')
        axes[i].axvline(mt, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Media Global: {mt:.1f}')
        axes[i].set_title(col, fontsize=13, weight='bold')
        axes[i].set_ylabel('Densidad')
        axes[i].legend(fontsize=8)
        pb.update(100 // n)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Distribuciones por Cluster vs Poblacion de la Vista (valores originales)',
                 fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    ruta = f"{OUTPUT_DIR}/dashboard_distribuciones_originales.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard guardado en: {ruta}")


def main():
    print("=" * 80)
    print(" CLUSTERING JERARQUICO - CLIENTES (2 CLUSTERS)")
    print("=" * 80)

    cu.configurar_estilo_graficos()

    try:
        # 1. Clustering sobre la vista normalizada
        print("[1/4] Clustering sobre vista normalizada...")
        with cu.ProgressBar(total=100, desc="Cargando vista") as pb:
            df_vista = cargar_vista(pb)
        with cu.ProgressBar(total=100, desc="Clusterizando") as pb:
            cluster_labels, sil, sil_sample = clusterizar_vista(df_vista, pb)

        # 2. Valores originales desde la vista sin normalizar (mismo orden)
        ids, df_original, cluster_labels, scaler = obtener_originales_de_vista(cluster_labels)

        # 3. CSV
        with cu.ProgressBar(total=100, desc="Exportando CSV") as pb:
            exportar_csv(ids, cluster_labels, pb)

        # 4. Graficos
        with cu.ProgressBar(total=100, desc="Dendrograma") as pb:
            graficar_dendrograma(scaler, df_original, pb)
        with cu.ProgressBar(total=100, desc="Boxplots") as pb:
            graficar_boxplots(df_original, cluster_labels, pb)
        with cu.ProgressBar(total=100, desc="Dashboard") as pb:
            graficar_dashboard(df_original, cluster_labels, pb)

        cu.exportar_silhouette(sil, ALGORITMO_NOMBRE, N_CLUSTERS,
                               f"{OUTPUT_DIR}/silhouette.json", sample_size=sil_sample)

        print("\n" + "=" * 80)
        print(" PROCESAMIENTO FINALIZADO")
        print("=" * 80)
        print(f"   - clientes_clusters_jerarquico.csv  ({len(ids)} clientes de la vista)")
        print(f"   - {OUTPUT_DIR}/dendrograma.png")
        print(f"   - {OUTPUT_DIR}/boxplots_variables_originales.png")
        print(f"   - {OUTPUT_DIR}/dashboard_distribuciones_originales.png")
        print(f"   - {OUTPUT_DIR}/silhouette.json")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()