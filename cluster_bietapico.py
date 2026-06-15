"""
================================================================================
Clustering Bietapico (Birch) sobre Clientes (2 Clusters)
================================================================================
- Replica el pipeline de limpieza de limpiar_dataset_clientes() en memoria:
    * Elimina whales (P95 en gasto_acumulado, ingreso_mensual o cantidad_millas)
    * Excluye gasto_acumulado_extra (multicolinealidad)
- Estandariza numericas, one-hot encodea categoricas (drop_first=True)
- Aplica Birch (k=2) y genera graficos en rangos originales + Silhouette
================================================================================
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import clustering_utils as cu


OUTPUT_DIR = "TPI/graficos/bietapico"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALGORITMO_NOMBRE = "birch_two_step"
N_CLUSTERS = 2
THRESHOLD = 0.5
BRANCHING_FACTOR = 50
SAMPLE_SILHOUETTE = 5000

NUMERIC_COLS = [
    'edad', 'cant_vuelos', 'gasto_acumulado',
    'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio'
]
CATEGORICAL_COLS = [
    'sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra'
]


def cargar_y_preparar(pb):
    """
    Replica el pipeline de limpiar_dataset_clientes():
      1. Carga Clientes.xlsx
      2. Detecta y elimina whales (P95 en gasto, ingreso o millas)
      3. Descarta gasto_acumulado_extra (multicolinealidad)
      4. Imputa nulos
      5. Estandariza numericas + OHE categoricas
    """
    pb.desc = "Cargando Clientes.xlsx"
    df = pd.read_excel('Clientes.xlsx').dropna(subset=['id_cliente'])
    pb.update(10)

    # --- Filtro whales (mismo criterio que limpiar_dataset_clientes) ---
    p95_gasto   = df['gasto_acumulado'].quantile(0.95)
    p95_ingreso = df['ingreso_mensual'].quantile(0.95)
    p95_millas  = df['cantidad_millas'].quantile(0.95)

    condicion_ballena = (
        (df['gasto_acumulado'] >= p95_gasto) |
        (df['ingreso_mensual'] >= p95_ingreso) |
        (df['cantidad_millas'] >= p95_millas)
    )
    df = df[~condicion_ballena].copy()

    print(f"      P95 gasto={p95_gasto:.2f} | ingreso={p95_ingreso:.2f} | millas={p95_millas:.2f}")
    print(f"      Clientes tras eliminar whales: {len(df)}")
    pb.update(10)

    # --- Guardar ids antes de cualquier transformacion ---
    ids = df['id_cliente'].values

    # --- Imputacion ---
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str).str.strip()
    pb.update(10)

    # --- Guardar originales (sin gasto_acumulado_extra, igual que la vista) ---
    df_original = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()

    # --- Estandarizacion + OHE ---
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUMERIC_COLS])
    df_num = pd.DataFrame(X_num, columns=NUMERIC_COLS, index=df.index)

    df_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True, dtype=int)

    df_vista = pd.concat([df_num, df_cat], axis=1).reset_index(drop=True)
    df_original = df_original.reset_index(drop=True)
    pb.update(70)

    print(f"      Vista lista: {df_vista.shape[0]} filas x {df_vista.shape[1]} columnas")

    return df_vista, df_original, ids


def ejecutar_bietapico(df_vista, df_original, pb):
    pb.desc = "Ejecutando Birch"
    x = df_vista.values
    modelo = Birch(n_clusters=N_CLUSTERS, threshold=THRESHOLD, branching_factor=BRANCHING_FACTOR)
    cluster_labels = modelo.fit_predict(x)
    pb.update(30)

    df_vista = df_vista.copy()
    df_original = df_original.copy()
    df_vista['cluster'] = cluster_labels
    df_original['cluster'] = cluster_labels

    n = len(df_original)
    if n > SAMPLE_SILHOUETTE:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SAMPLE_SILHOUETTE, replace=False)
        sil = silhouette_score(x[idx], cluster_labels[idx])
        sil_sample = SAMPLE_SILHOUETTE
    else:
        sil = silhouette_score(x, cluster_labels)
        sil_sample = None
    pb.update(70)

    print(f"\n      Micro-clusters Birch: {modelo.subcluster_centers_.shape[0]}")
    cu.imprimir_tamanios_clusters(df_original)
    cu.imprimir_silhouette(sil, ALGORITMO_NOMBRE, sample_size=sil_sample)

    return df_vista, df_original, modelo, sil, sil_sample


def exportar_csv(ids, cluster_labels, pb):
    pb.desc = "Exportando CSV"
    out = pd.DataFrame({'id_cliente': ids, 'cluster': cluster_labels})
    ruta = "clientes_clusters_bietapico.csv"
    out.to_csv(ruta, index=False)
    pb.update(100)
    print(f"      [OK] {ruta} ({len(out)} filas)")


def graficar_pca_birch(df_vista, modelo, pb):
    pb.desc = "PCA 2D"
    cols_pca = [c for c in df_vista.columns if c != 'cluster']
    x = df_vista[cols_pca].values
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x)
    sub_2d = pca.transform(modelo.subcluster_centers_)

    fig, ax = plt.subplots(figsize=(14, 9))
    for cid in sorted(np.unique(df_vista['cluster'])):
        mask = df_vista['cluster'].values == cid
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1],
                   c=cu.PALETA_CLUSTERS[cid], label=f'Cluster {cid}',
                   alpha=0.30, s=15, edgecolors='none')
    ax.scatter(sub_2d[:, 0], sub_2d[:, 1], c='black', marker='X', s=140,
               zorder=5, edgecolors='white', linewidths=1.2,
               label=f'Micro-clusters ({len(sub_2d)})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
    ax.set_title(f'Birch (Numericas std + OHE) - k={N_CLUSTERS}', fontsize=15, weight='bold')
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    ruta = f"{OUTPUT_DIR}/birch_pca_subclusters.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    pb.update(100)
    print(f"      [OK] PCA guardado en: {ruta}")


def graficar_boxplots(df_original, pb):
    pb.desc = "Boxplots originales"
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(x='cluster', y=col, data=df_original, ax=axes[i],
                    palette=['#4A90E2', '#FF6B6B'])
        axes[i].set_title(col)
        axes[i].set_xlabel('Cluster')
        pb.update(100 // n)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Variables numericas (valores originales) por Cluster',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    ruta = f"{OUTPUT_DIR}/boxplots_numericas.png"
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Boxplots guardados en: {ruta}")


def graficar_radar(df_vista, pb):
    pb.desc = "Radar z-score"
    medias = df_vista.groupby('cluster')[NUMERIC_COLS].mean()
    n_var = len(NUMERIC_COLS)
    angulos = np.linspace(0, 2 * np.pi, n_var, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for cid in sorted(medias.index):
        valores = medias.loc[cid].tolist() + [medias.loc[cid].iloc[0]]
        ax.plot(angulos, valores, 'o-', linewidth=2,
                color=cu.PALETA_CLUSTERS[cid], label=f'Cluster {cid}')
        ax.fill(angulos, valores, alpha=0.1, color=cu.PALETA_CLUSTERS[cid])
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(NUMERIC_COLS, fontsize=10)
    ax.set_title('Perfil z-score promedio por Cluster', fontsize=15, weight='bold', pad=25)
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    ruta = f"{OUTPUT_DIR}/radar_perfil_numerico.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    pb.update(100)
    print(f"      [OK] Radar guardado en: {ruta}")


def main():
    print("=" * 80)
    print(" CLUSTERING BIETAPICO (BIRCH) - CLIENTES (NUM + OHE CAT)")
    print("=" * 80)

    cu.configurar_estilo_graficos()

    try:
        with cu.ProgressBar(total=100, desc="Preparando datos") as pb:
            df_vista, df_original, ids = cargar_y_preparar(pb)
        with cu.ProgressBar(total=100, desc="Ejecutando Birch") as pb:
            df_vista, df_original, modelo, sil, sil_sample = ejecutar_bietapico(df_vista, df_original, pb)

        cluster_labels = df_original['cluster'].values

        with cu.ProgressBar(total=100, desc="Exportando CSV") as pb:
            exportar_csv(ids, cluster_labels, pb)
        with cu.ProgressBar(total=100, desc="PCA 2D") as pb:
            graficar_pca_birch(df_vista, modelo, pb)
        with cu.ProgressBar(total=100, desc="Boxplots") as pb:
            graficar_boxplots(df_original, pb)
        with cu.ProgressBar(total=100, desc="Radar") as pb:
            graficar_radar(df_vista, pb)

        cu.exportar_silhouette(sil, ALGORITMO_NOMBRE, N_CLUSTERS,
                               f"{OUTPUT_DIR}/silhouette.json", sample_size=sil_sample)

        print("\n" + "=" * 80)
        print(" PROCESAMIENTO FINALIZADO")
        print("=" * 80)
        print(f"   - clientes_clusters_bietapico.csv  ({len(ids)} clientes, whales excluidos)")
        print(f"   - {OUTPUT_DIR}/birch_pca_subclusters.png")
        print(f"   - {OUTPUT_DIR}/boxplots_numericas.png")
        print(f"   - {OUTPUT_DIR}/radar_perfil_numerico.png")
        print(f"   - {OUTPUT_DIR}/silhouette.json")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()