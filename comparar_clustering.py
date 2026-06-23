"""
================================================================================
Orquestador de comparacion de clustering: K-Medias, Jerarquico y Bietapico
================================================================================
Corre 3 algoritmos x k={2, 3} = 6 corridas y genera, por cada una:
  - tabla_promedios_vs_poblacion.csv (6 numericas + delta vs media global)
  - dashboard_numericas.png (KDE por cluster + KDE poblacional + medias)
  - boxplots_numericas.png (con linea de media global)
  - silhouette.json (global + por cluster)
  - silhouette_plot.png (clasico plot de silhouettes por cluster)
  - metricas.json (silhouette, separacion de medias, tamanios)
  - [solo jerarquico y bietapico] dashboard_categoricas.png + comparativas
  - especificos: pca_2d (kmeans) / dendrograma (jerarquico) /
                 pca_subclusters + radar (bietapico)

Al final produce TPI/graficos/comparacion_clustering.md con tabla
algoritmo x k y veredicto (mayor silhouette + mayor separacion de medias)
para elegir la corrida ganadora de la fase de evaluacion.

Entradas:
  - K-Medias:      vista_clientes_kmeans_normalizada.csv  (6 numericas Z-score)
  - Jerarquico:    idem para clustering; reporta categoricas cruzando por
                   posicion con vista_clientes_kmeans.csv
  - Bietapico:     Clientes.xlsx in-memory (6 num std + OHE 6 cat, whales P95)
================================================================================
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, Birch, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import clustering_utils as cu


# --------------------------------------------------------------------------- #
# Configuracion general
# --------------------------------------------------------------------------- #
NUMERIC_COLS = [
    'edad', 'cant_vuelos', 'gasto_acumulado',
    'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio'
]
CATEGORICAL_COLS = [
    'sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra'
]

RUTA_VISTA_NORM = "vistas_minables/vista_clientes_kmeans_normalizada.csv"
RUTA_VISTA_ORIG = "vistas_minables/vista_clientes_kmeans.csv"
RUTA_CLIENTES_XLSX = "Clientes.xlsx"

SAMPLE_SIL = 5000
SAMPLE_DENDRO = 10000
KS = [2, 3]
ALGORITMOS = ["kmeans", "jerarquico", "bietapico"]

BASE_OUT = "TPI/graficos"


# --------------------------------------------------------------------------- #
# Carga de datos por algoritmo
# --------------------------------------------------------------------------- #
def cargar_vista_kmeans():
    """Carga la vista normalizada (6 numericas Z-score) y la vista cruda
    (mismo orden, con numericas + categoricas). Alineadas por posicion."""
    df_norm = pd.read_csv(RUTA_VISTA_NORM)
    df_orig = pd.read_csv(RUTA_VISTA_ORIG)
    assert len(df_norm) == len(df_orig), \
        f"Desajuste de filas entre vista normalizada y cruda: {len(df_norm)} vs {len(df_orig)}"
    return df_norm, df_orig


def cargar_bietapico():
    """Replica el pipeline de cluster_bietapico.cargar_y_preparar:
    limpieza whales P95, imputacion, estandarizacion numerica + OHE categorica.
    Devuelve (df_vista_mixta, df_originales, ids)."""
    df = pd.read_excel(RUTA_CLIENTES_XLSX).dropna(subset=['id_cliente'])

    p95_gasto = df['gasto_acumulado'].quantile(0.95)
    p95_ingreso = df['ingreso_mensual'].quantile(0.95)
    p95_millas = df['cantidad_millas'].quantile(0.95)
    cond_ballena = (
        (df['gasto_acumulado'] >= p95_gasto) |
        (df['ingreso_mensual'] >= p95_ingreso) |
        (df['cantidad_millas'] >= p95_millas)
    )
    df = df[~cond_ballena].copy()
    print(f"      P95 gasto={p95_gasto:.2f} | ingreso={p95_ingreso:.2f} | millas={p95_millas:.2f}")
    print(f"      Clientes tras eliminar whales: {len(df)}")

    ids = df['id_cliente'].values

    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    for col in CATEGORICAL_COLS:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str).str.strip()

    df_original = df[NUMERIC_COLS + CATEGORICAL_COLS].copy().reset_index(drop=True)

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUMERIC_COLS])
    df_num = pd.DataFrame(X_num, columns=NUMERIC_COLS, index=df.index)
    df_cat = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True, dtype=int)
    df_vista = pd.concat([df_num, df_cat], axis=1).reset_index(drop=True)

    print(f"      Vista mixta lista: {df_vista.shape[0]} filas x {df_vista.shape[1]} columnas")
    return df_vista, df_original, ids


# --------------------------------------------------------------------------- #
# Ejecucion de cada algoritmo
# --------------------------------------------------------------------------- #
def correr_kmeans(df_norm, k):
    """Fitea K-Medias sobre la vista normalizada. Devuelve labels y x."""
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(df_norm.values)
    return labels, df_norm.values, modelo


def correr_jerarquico(df_norm, k):
    """Fitea AgglomerativeClustering (Ward) sobre la vista normalizada."""
    modelo = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = modelo.fit_predict(df_norm.values)
    return labels, df_norm.values, modelo


def correr_bietapico(df_vista, k):
    """Fitea Birch sobre la vista mixta (num std + OHE cat)."""
    modelo = Birch(n_clusters=k, threshold=0.5, branching_factor=50)
    labels = modelo.fit_predict(df_vista.values)
    return labels, df_vista.values, modelo


# --------------------------------------------------------------------------- #
# Salidas comunes a las 6 corridas
# --------------------------------------------------------------------------- #
def salidas_comunes(df_reporte, labels, x, out_dir, k, algoritmo_nombre):
    """Genera tabla, dashboard numerico, boxplots, silhouette y metricas.
    `df_reporte` debe tener las 6 numericas (+ categ si aplica) SIN cluster;
    se le agrega la columna cluster aqui. `x` es la matriz usada para clustering
    (sin la columna cluster)."""
    os.makedirs(out_dir, exist_ok=True)
    df_reporte = df_reporte.copy()
    df_reporte['cluster'] = labels

    print("\n      Tamanios de clusters:")
    cu.imprimir_tamanios_clusters(df_reporte)

    # 1. Tabla promedios vs poblacion
    cu.tabla_promedios_vs_poblacion(df_reporte, NUMERIC_COLS, f"{out_dir}/tabla_promedios_vs_poblacion.csv")

    # 2. Dashboard + comparativas individuales numericas
    cu.generar_dashboard_consolidado_numericas(df_reporte, NUMERIC_COLS, out_dir)
    cu.graficar_comparativa_numericas(df_reporte, NUMERIC_COLS, out_dir)

    # 3. Boxplots con media global
    cu.graficar_boxplots_numericas(df_reporte, NUMERIC_COLS, out_dir)

    # 4. Silhouette global + por cluster + plot
    sil_global, sil_por_cluster, _, sil_sample = cu.calcular_silhouette_por_cluster(
        x, labels, sample_size=SAMPLE_SIL
    )
    cu.imprimir_silhouette(sil_global, algoritmo_nombre, sample_size=sil_sample)
    cu.exportar_silhouette(
        sil_global, algoritmo_nombre, k, f"{out_dir}/silhouette.json",
        sample_size=sil_sample, por_cluster=sil_por_cluster
    )
    cu.graficar_silhouette(
        x, labels, f"{out_dir}/silhouette_plot.png",
        sample_size=SAMPLE_SIL, algoritmo_nombre=algoritmo_nombre
    )

    # 5. Separacion de medias (F between/within sobre las 6 numericas)
    separacion = cu.calcular_separacion_medias(df_reporte, NUMERIC_COLS, k)

    # 6. Metricas consolidadas
    tamanios = {int(c): int((labels == c).sum()) for c in sorted(np.unique(labels))}
    metricas = {
        "algoritmo": algoritmo_nombre,
        "k": int(k),
        "silhouette_global": round(float(sil_global), 4),
        "silhouette_por_cluster": {int(c): round(float(v), 4) for c, v in sil_por_cluster.items()},
        "silhouette_sample_size": int(sil_sample) if sil_sample else None,
        "separacion_medias_F": round(float(separacion), 4),
        "tamanios": tamanios,
        "n_total": int(len(labels)),
    }
    with open(f"{out_dir}/metricas.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print(f"      [OK] Metricas guardadas en: {out_dir}/metricas.json")

    return metricas


def salidas_categoricas(df_reporte, labels, out_dir):
    """Dashboard + comparativas individuales de variables categoricas."""
    df = df_reporte.copy()
    df['cluster'] = labels
    cu.graficar_comparativa_categoricas(df, CATEGORICAL_COLS, out_dir)
    cu.generar_dashboard_consolidado(df, CATEGORICAL_COLS, out_dir)


# --------------------------------------------------------------------------- #
# Salidas especificas por algoritmo
# --------------------------------------------------------------------------- #
def graficar_pca_2d(x, labels, out_dir, algoritmo_nombre, k):
    """PCA 2D sobre la matriz de clustering, scatter coloreado por cluster."""
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(12, 8))
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        color = cu.PALETA_CLUSTERS.get(int(cid), cu.COLORES_LISTA[int(cid) % len(cu.COLORES_LISTA)])
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], c=color,
                   label=f'Cluster {int(cid)}', alpha=0.35, s=15, edgecolors='none')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'{algoritmo_nombre} (k={k}) - PCA 2D', fontsize=14, weight='bold')
    ax.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    ruta = f"{out_dir}/pca_2d.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] PCA 2D guardado en: {ruta}")


def graficar_dendrograma(x, labels, out_dir, k):
    """Dendrograma Ward con linea de corte al k indicado (sobre muestra)."""
    n = len(x)
    if n > SAMPLE_DENDRO:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=SAMPLE_DENDRO, replace=False)
        x_dendro = x[idx]
    else:
        x_dendro = x

    z = linkage(x_dendro, method='ward')
    p_hojas = max(2, len(x_dendro) // 500)

    fig, ax = plt.subplots(figsize=(20, 9))
    dendrogram(
        z, truncate_mode='lastp', p=p_hojas,
        leaf_rotation=90., leaf_font_size=10.,
        show_contracted=True, show_leaf_counts=True, ax=ax
    )
    corte_y = z[-(k - 1), 2]
    ax.axhline(y=corte_y, color='red', linestyle='--', linewidth=1.8,
               label=f'Corte k={k} (d={corte_y:.2f})')
    ax.set_title(
        f'Dendrograma - Ward  |  muestra={len(x_dendro):,} obs  |  '
        f'cada hoja >= ~{len(x_dendro) // p_hojas:,} obs',
        fontsize=14, weight='bold', pad=15
    )
    ax.set_xlabel('Hojas (cada etiqueta muestra cantidad de obs agrupadas)', fontsize=11)
    ax.set_ylabel('Distancia Ward', fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()
    ruta = f"{out_dir}/dendrograma.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dendrograma guardado en: {ruta}")


def graficar_pca_subclusters_birch(df_vista, modelo, out_dir, k):
    """PCA 2D de la vista mixta + micro-clusters Birch proyectados."""
    cols = [c for c in df_vista.columns if c != 'cluster']
    x = df_vista[cols].values
    pca = PCA(n_components=2, random_state=42)
    x_2d = pca.fit_transform(x)
    sub_2d = pca.transform(modelo.subcluster_centers_)
    labels = df_vista['cluster'].values

    fig, ax = plt.subplots(figsize=(14, 9))
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        color = cu.PALETA_CLUSTERS.get(int(cid), cu.COLORES_LISTA[int(cid) % len(cu.COLORES_LISTA)])
        ax.scatter(x_2d[mask, 0], x_2d[mask, 1], c=color,
                   label=f'Cluster {int(cid)}', alpha=0.30, s=15, edgecolors='none')
    ax.scatter(sub_2d[:, 0], sub_2d[:, 1], c='black', marker='X', s=140,
               zorder=5, edgecolors='white', linewidths=1.2,
               label=f'Micro-clusters ({len(sub_2d)})')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=13)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=13)
    ax.set_title(f'Birch (Num std + OHE) - k={k}', fontsize=15, weight='bold')
    ax.legend(fontsize=12, frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    ruta = f"{out_dir}/pca_subclusters.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] PCA subclusters guardado en: {ruta}")


def graficar_radar_birch(df_vista, out_dir, k):
    """Radar de perfiles z-score (numericas) por cluster."""
    medias = df_vista.groupby('cluster')[NUMERIC_COLS].mean()
    n_var = len(NUMERIC_COLS)
    angulos = np.linspace(0, 2 * np.pi, n_var, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for cid in sorted(medias.index):
        color = cu.PALETA_CLUSTERS.get(int(cid), cu.COLORES_LISTA[int(cid) % len(cu.COLORES_LISTA)])
        valores = medias.loc[cid].tolist() + [medias.loc[cid].iloc[0]]
        ax.plot(angulos, valores, 'o-', linewidth=2, color=color, label=f'Cluster {int(cid)}')
        ax.fill(angulos, valores, alpha=0.1, color=color)
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(NUMERIC_COLS, fontsize=10)
    ax.set_title(f'Perfil z-score promedio por Cluster (k={k})', fontsize=15, weight='bold', pad=25)
    ax.legend(loc='upper right', fontsize=12, bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    ruta = f"{out_dir}/radar_perfil_numerico.png"
    plt.savefig(ruta, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Radar guardado en: {ruta}")


# --------------------------------------------------------------------------- #
# Orquestacion por algoritmo
# --------------------------------------------------------------------------- #
def ejecutar_kmeans(k):
    nombre = "kmeans"
    print(f"\n   >> K-Medias | k={k}")
    out_dir = f"{BASE_OUT}/{nombre}/k{k}"
    df_norm, df_orig = cargar_vista_kmeans()
    labels, x, modelo = correr_kmeans(df_norm, k)

    # Reporte SOLO numericas (segun decision del usuario)
    df_reporte = df_orig[NUMERIC_COLS].copy().reset_index(drop=True)
    assert len(df_reporte) == len(labels)

    metricas = salidas_comunes(df_reporte, labels, x, out_dir, k, nombre)
    graficar_pca_2d(x, labels, out_dir, "K-Medias", k)

    # Exportar CSV de asignaciones
    out_csv = pd.DataFrame({
        'id_cliente': df_orig['id_cliente'].values if 'id_cliente' in df_orig.columns else np.arange(len(labels)),
        'cluster': labels
    })
    out_csv.to_csv(f"clientes_clusters_kmeans.csv", index=False)
    print(f"      [OK] clientes_clusters_kmeans.csv ({len(out_csv)} filas)")
    return metricas


def ejecutar_jerarquico(k):
    nombre = "jerarquico"
    print(f"\n   >> Jerarquico (Ward) | k={k}")
    out_dir = f"{BASE_OUT}/{nombre}/k{k}"
    df_norm, df_orig = cargar_vista_kmeans()
    labels, x, modelo = correr_jerarquico(df_norm, k)

    # Reporte numericas + categoricas (cruce por posicion con la vista cruda)
    df_reporte = df_orig[NUMERIC_COLS + CATEGORICAL_COLS].copy().reset_index(drop=True)
    assert len(df_reporte) == len(labels)

    metricas = salidas_comunes(df_reporte, labels, x, out_dir, k, nombre)
    salidas_categoricas(df_reporte, labels, out_dir)
    graficar_dendrograma(x, labels, out_dir, k)

    # Exportar CSV de asignaciones
    out_csv = pd.DataFrame({
        'id_cliente': df_orig['id_cliente'].values if 'id_cliente' in df_orig.columns else np.arange(len(labels)),
        'cluster': labels
    })
    out_csv.to_csv("clientes_clusters_jerarquico.csv", index=False)
    print(f"      [OK] clientes_clusters_jerarquico.csv ({len(out_csv)} filas)")
    return metricas


def ejecutar_bietapico(k):
    nombre = "bietapico"
    print(f"\n   >> Bietapico (Birch) | k={k}")
    out_dir = f"{BASE_OUT}/{nombre}/k{k}"
    df_vista, df_original, ids = cargar_bietapico()
    labels, x, modelo = correr_bietapico(df_vista, k)

    # Reporte numericas + categoricas (desde los originales pre-OHE)
    df_reporte = df_original.copy()
    assert len(df_reporte) == len(labels)

    metricas = salidas_comunes(df_reporte, labels, x, out_dir, k, nombre)
    salidas_categoricas(df_reporte, labels, out_dir)

    # Especificos: PCA subclusters + radar (necesitan cluster en df_vista)
    df_vista_con_cluster = df_vista.copy()
    df_vista_con_cluster['cluster'] = labels
    graficar_pca_subclusters_birch(df_vista_con_cluster, modelo, out_dir, k)
    graficar_radar_birch(df_vista_con_cluster, out_dir, k)

    # Exportar CSV de asignaciones
    out_csv = pd.DataFrame({'id_cliente': ids, 'cluster': labels})
    out_csv.to_csv("clientes_clusters_bietapico.csv", index=False)
    print(f"      [OK] clientes_clusters_bietapico.csv ({len(out_csv)} filas)")
    return metricas


# --------------------------------------------------------------------------- #
# Reporte comparativo final
# --------------------------------------------------------------------------- #
def generar_reporte_comparativo(metricas_por_corrida):
    """Genera TPI/graficos/comparacion_clustering.md con tabla algoritmo x k
    y veredicto (mayor silhouette + mayor separacion de medias)."""
    os.makedirs(BASE_OUT, exist_ok=True)
    ruta_md = f"{BASE_OUT}/comparacion_clustering.md"

    # --- Normalizacion min-max de silhouette y separacion para score combinado ---
    sils = [m["silhouette_global"] for m in metricas_por_corrida]
    seps = [m["separacion_medias_F"] for m in metricas_por_corrida]
    sil_min, sil_max = min(sils), max(sils)
    sep_min, sep_max = min(seps), max(seps)

    def norm(v, vmin, vmax):
        if vmax == vmin:
            return 0.5
        return (v - vmin) / (vmax - vmin)

    filas = []
    for m in metricas_por_corrida:
        sil_n = norm(m["silhouette_global"], sil_min, sil_max)
        sep_n = norm(m["separacion_medias_F"], sep_min, sep_max)
        score = 0.5 * sil_n + 0.5 * sep_n
        filas.append({**m, "score_combinado": score})

    # Veredicto: mayor score combinado
    ganador = max(filas, key=lambda r: r["score_combinado"])
    # Desempate secundario: mayor silhouette global
    mejor_sil = max(filas, key=lambda r: r["silhouette_global"])
    mejor_sep = max(filas, key=lambda r: r["separacion_medias_F"])

    with open(ruta_md, "w", encoding="utf-8") as f:
        f.write("# Comparacion de Algoritmos de Clustering\n\n")
        f.write("Se ejecutaron 3 algoritmos (K-Medias, Jerarquico-Ward, Bietapico-Birch) "
                "con k=2 y k=3 sobre la base de clientes (whales P95 excluidos).\n\n")
        f.write("> **Entradas:** K-Medias y Jerarquico clusterizan sobre "
                "`vista_clientes_kmeans_normalizada.csv` (6 numericas Z-score). "
                "Bietapico clusteriza sobre una vista mixta (6 numericas std + OHE de 6 "
                "categoricas) construida in-memory desde `Clientes.xlsx`.\n")
        f.write("> **Advertencia:** los Silhouette del Bietapico no son estrictamente "
                "comparables con los de K-Medias/Jerarquico por calcularse en espacios de "
                "dimension distinta. La metrica de **separacion de medias (F)** si es "
                "comparable (se calcula sobre las mismas 6 numericas en los 3 casos).\n\n")

        # --- Tabla principal ---
        f.write("## Tabla comparativa\n\n")
        f.write("| Algoritmo | k | Silhouette global | Silhouette por cluster | "
                "Separacion medias (F) | Score combinado | Tamanios |\n")
        f.write("|-----------|---|-------------------|------------------------|"
                "------------------------|-----------------|----------|\n")
        for r in filas:
            sil_pc = ", ".join(f"C{c}={v}" for c, v in
                               sorted(r["silhouette_por_cluster"].items()))
            tam = ", ".join(f"C{c}={n}" for c, n in sorted(r["tamanios"].items()))
            f.write(
                f"| {r['algoritmo']} | {r['k']} | "
                f"**{r['silhouette_global']:.4f}** | {sil_pc} | "
                f"**{r['separacion_medias_F']:.4f}** | "
                f"{r['score_combinado']:.4f} | {tam} |\n"
            )

        f.write("\n## Veredicto\n\n")
        f.write(f"- **Mejor Silhouette global:** `{mejor_sil['algoritmo']}` con k="
                f"{mejor_sil['k']} -> {mejor_sil['silhouette_global']:.4f}\n")
        f.write(f"- **Mejor separacion de medias (F):** `{mejor_sep['algoritmo']}` con k="
                f"{mejor_sep['k']} -> {mejor_sep['separacion_medias_F']:.4f}\n")
        f.write(f"- **Score combinado (50% Silhouette + 50% Separacion, normalizados):** "
                f"ganador `{ganador['algoritmo']}` con k={ganador['k']} "
                f"-> score={ganador['score_combinado']:.4f}\n\n")

        f.write("### Recomendacion para la fase de evaluacion\n\n")
        f.write(f"Se recomienda continuar la fase de evaluacion con **{ganador['algoritmo']} "
                f"(k={ganador['k']})**, por presentar el mejor balance entre coherencia "
                f"interna (Silhouette) y separacion entre grupos (F between/within).\n\n")
        f.write(f"Archivo de asignaciones correspondiente: `clientes_clusters_{ganador['algoritmo']}.csv`.\n")

    print(f"\n[OK] Reporte comparativo generado en: {ruta_md}")
    return ruta_md, ganador


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("=" * 80)
    print(" COMPARADOR DE CLUSTERING  (K-Medias | Jerarquico | Bietapico)  k=2,3")
    print("=" * 80)

    cu.configurar_estilo_graficos()

    metricas_por_corrida = []
    try:
        # Bietapico: la carga in-memory es la mas pesada, se hace dentro de su funcion.
        # K-Medias y Jerarquico comparten la misma vista normalizada.
        for k in KS:
            m = ejecutar_kmeans(k)
            metricas_por_corrida.append(m)
        for k in KS:
            m = ejecutar_jerarquico(k)
            metricas_por_corrida.append(m)
        for k in KS:
            m = ejecutar_bietapico(k)
            metricas_por_corrida.append(m)

        ruta_md, ganador = generar_reporte_comparativo(metricas_por_corrida)

        print("\n" + "=" * 80)
        print(" PROCESAMIENTO FINALIZADO")
        print("=" * 80)
        print(f"   6 corridas generadas en {BASE_OUT}/{{kmeans,jerarquico,bietapico}}/k{{2,3}}/")
        print(f"   Reporte comparativo: {ruta_md}")
        print(f"   Ganador: {ganador['algoritmo']} (k={ganador['k']})  "
              f"| Silhouette={ganador['silhouette_global']:.4f}  "
              f"| F={ganador['separacion_medias_F']:.4f}")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
