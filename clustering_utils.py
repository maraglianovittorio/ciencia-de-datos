"""
================================================================================
Utilidades compartidas para clustering de Clientes
================================================================================
Funciones reutilizables por cluster_jerarquico.py, cluster_bietapico.py,
cluster_kmeans.py y comparar_clustering.py.

Generalizado para soportar k variable (2, 3 o mas clusters):
  - Paleta dinamica (PALETA_CLUSTERS / COLORES_LISTA con 3 colores base).
  - Funciones de perfilado y graficos iteran sobre los clusters presentes.
  - Silhouette descompuesto por cluster via silhouette_samples.
================================================================================
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import silhouette_score, silhouette_samples


PALETA_CLUSTERS = {0: '#4A90E2', 1: '#FF6B6B', 2: '#2ECC71'}
COLORES_LISTA = ['#4A90E2', '#FF6B6B', '#2ECC71', '#9B59B6', '#F39C12', '#1ABC9C']
COLOR_POBLACIONAL = '#95A5A6'


def configurar_estilo_graficos():
    """Aplica el mismo estilo premium que kmeans.py."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'figure.figsize': (11, 6),
        'axes.edgecolor': '#cccccc',
        'grid.color': '#f0f0f0',
        'axes.titlepad': 15,
        'axes.labelpad': 10,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10
    })


def cargar_y_limpiar_clientes():
    """
    Carga Clientes.xlsx y aplica limpieza inicial de filas vacías.
    Mismo criterio que kmeans.py.
    """
    print("[1/5] Cargando base de datos Clientes.xlsx...")
    if not os.path.exists('Clientes.xlsx'):
        raise FileNotFoundError("No se encontró el archivo Clientes.xlsx en el directorio actual.")

    df = pd.read_excel('Clientes.xlsx')
    n_inicial = len(df)

    df = df.dropna(subset=['id_cliente'])
    n_final = len(df)
    print(f"      Dataset cargado correctamente. Filas leidas: {n_inicial} -> Filas validas: {n_final}")

    numeric_cols = [
        'edad', 'cant_vuelos', 'gasto_acumulado', 'gasto_acumulado_extra',
        'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio'
    ]
    categorical_cols = [
        'sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra'
    ]

    for col in numeric_cols:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    for col in categorical_cols:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)

    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()

    return df, numeric_cols, categorical_cols


def imprimir_tamanios_clusters(df):
    """Imprime el tamaño (cantidad y %) de cada cluster."""
    tamanios = df['cluster'].value_counts().sort_index()
    for cluster_id, count in tamanios.items():
        pct = (count / len(df)) * 100
        print(f"      - Cluster {cluster_id}: {count} clientes ({pct:.2f}%)")


def imprimir_silhouette(score, algoritmo, sample_size=None):
    """Imprime en consola el coeficiente de Silhouette global."""
    extra = f" (muestra={sample_size})" if sample_size else ""
    print(f"      Coeficiente de Silhouette [{algoritmo}]{extra}: {score:.4f}")


def exportar_silhouette(score, algoritmo, n_clusters, ruta_json, sample_size=None,
                        por_cluster=None):
    """
    Persiste el coeficiente de Silhouette a un archivo JSON.
    Retrocompatible: si se omite `por_cluster`, solo guarda el global.
    """
    payload = {
        "algoritmo": algoritmo,
        "n_clusters": int(n_clusters),
        "silhouette": round(float(score), 4),
    }
    if sample_size is not None:
        payload["sample_size"] = int(sample_size)
        payload["muestreado"] = True
    else:
        payload["muestreado"] = False

    if por_cluster is not None:
        payload["silhouette_por_cluster"] = {
            int(c): round(float(v), 4) for c, v in por_cluster.items()
        }

    os.makedirs(os.path.dirname(ruta_json), exist_ok=True)
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"      [OK] Silhouette exportado a: {ruta_json}")


def calcular_silhouette_por_cluster(x, labels, sample_size=5000, random_state=42):
    """
    Calcula el coeficiente de Silhouette global + el promedio por cluster
    usando silhouette_samples sobre una muestra (para evitar O(N^2) en memoria).

    Devuelve:
        sil_global: float
        sil_por_cluster: dict {cluster_id: float}
        labels_muestra: np.ndarray (labels alineados a la muestra usada)
        sample_size_usado: int o None
    """
    x = np.asarray(x)
    labels = np.asarray(labels)
    n = len(labels)

    if sample_size is not None and n > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample_size, replace=False)
        xs, ls = x[idx], labels[idx]
        usado = int(sample_size)
    else:
        xs, ls = x, labels
        usado = None

    sil_global = float(silhouette_score(xs, ls))
    sil_samples = silhouette_samples(xs, ls)

    por_cluster = {}
    for c in sorted(np.unique(ls)):
        por_cluster[int(c)] = float(sil_samples[ls == c].mean())

    return sil_global, por_cluster, ls, usado


def graficar_silhouette(x, labels, ruta_png, sample_size=5000, random_state=42,
                        algoritmo_nombre=""):
    """
    Genera el clasico grafico de silhouettes por cluster (barras horizontales
    ordenadas, con linea vertical del promedio global).
    """
    x = np.asarray(x)
    labels = np.asarray(labels)
    n_total = len(labels)
    clusters = sorted(np.unique(labels))
    k = len(clusters)

    if sample_size is not None and n_total > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_total, size=sample_size, replace=False)
        xs, ls = x[idx], labels[idx]
    else:
        xs, ls = x, labels

    sil_samples = silhouette_samples(xs, ls)
    sil_global = float(silhouette_score(xs, ls))

    fig, ax = plt.subplots(figsize=(10, 7))
    y_lower = 10
    for c in clusters:
        vals = np.sort(sil_samples[ls == c])
        n_c = len(vals)
        y_upper = y_lower + n_c
        color = PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         alpha=0.75, color=color)
        # n real del cluster (en todo el dataset, no en la muestra)
        n_real = int((labels == c).sum())
        ax.text(-0.02, y_lower + n_c / 2.0, f'Cluster {c} (n={n_real})',
                fontsize=11, va='center')
        y_lower = y_upper + 10

    ax.axvline(sil_global, color='red', linestyle='--', linewidth=2,
               label=f'Silhouette global = {sil_global:.4f}')
    ax.set_xlabel('Coeficiente de Silhouette')
    ax.set_ylabel('Cluster (observaciones ordenadas)')
    titulo = f'Silhouette por Cluster  |  k={k}'
    if algoritmo_nombre:
        titulo = f'{algoritmo_nombre}  |  {titulo}'
    ax.set_title(titulo, fontsize=14, weight='bold', pad=15)
    ax.set_yticks([])
    ax.set_xlim(-0.1, 1.0)
    ax.legend(loc='upper right', fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(ruta_png), exist_ok=True)
    plt.savefig(ruta_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Plot de Silhouette guardado en: {ruta_png}")


def tabla_promedios_vs_poblacion(df, numeric_cols, ruta_csv):
    """
    Genera un CSV con filas=variables numericas y columnas:
        media_c0, media_c1, ..., media_poblacional, delta_c0, delta_c1, ...
    donde delta_ci = media_ci - media_poblacional.
    Requiere columna 'cluster' en df.
    """
    clusters = sorted(df['cluster'].unique())
    medias = df.groupby('cluster')[numeric_cols].mean()
    media_pob = df[numeric_cols].mean()

    tabla = pd.DataFrame(index=numeric_cols)
    for c in clusters:
        tabla[f'media_c{int(c)}'] = medias.loc[c]
        tabla[f'delta_c{int(c)}'] = medias.loc[c] - media_pob
    tabla['media_poblacional'] = media_pob

    tabla = tabla.round(4)
    os.makedirs(os.path.dirname(ruta_csv), exist_ok=True)
    tabla.to_csv(ruta_csv, encoding='utf-8')
    print(f"      [OK] Tabla promedios vs poblacion guardada en: {ruta_csv}")
    return tabla


def calcular_separacion_medias(df, numeric_cols, k):
    """
    Ratio between-cluster / within-cluster (estadistico F del ANOVA one-way)
    promediado sobre las variables numericas. Mayor valor => mayor separacion
    entre clusters respecto de la variabilidad interna.

    Requiere columna 'cluster' en df.
    """
    N = len(df)
    if N - k <= 0:
        return 0.0
    global_mean = df[numeric_cols].mean()
    ratios = []
    for col in numeric_cols:
        ss_between = 0.0
        ss_within = 0.0
        for c in range(k):
            sub = df[df['cluster'] == c]
            nc = len(sub)
            if nc == 0:
                continue
            ss_between += nc * (sub[col].mean() - global_mean[col]) ** 2
            ss_within += ((sub[col] - sub[col].mean()) ** 2).sum()
        denom_within = ss_within / (N - k)
        denom_between = ss_between / (k - 1) if k > 1 else ss_between
        if denom_within > 0:
            ratios.append(denom_between / denom_within)
    return float(np.mean(ratios)) if ratios else 0.0


def generar_resumen_perfil(df, numeric_cols):
    """
    Imprime en consola el perfil promedio numerico de cada cluster + la media
    poblacional y el delta. Generalizado para k variable.
    """
    print("\n" + "=" * 80)
    print(" PERFILES PROMEDIO DE LOS CLUSTERS (VARIABLES NUMERICAS)")
    print("=" * 80)

    clusters = sorted(df['cluster'].unique())
    medias = df.groupby('cluster')[numeric_cols].mean()
    media_pob = df[numeric_cols].mean()

    resumen = medias.T
    resumen.columns = [f'Cluster {c} (Promedio)' for c in resumen.columns]
    resumen['Media Poblacional'] = media_pob
    for c in clusters:
        col = f'Cluster {c} (Promedio)'
        resumen[f'Delta C{c}'] = resumen[col] - resumen['Media Poblacional']

    print(resumen.round(2).to_string())
    print("=" * 80 + "\n")

    # Interpretacion: cluster de mayor vs menor ingreso
    ingresos = medias['ingreso_mensual'] if 'ingreso_mensual' in numeric_cols else None
    if ingresos is not None and len(ingresos) >= 2:
        c_max = ingresos.idxmax()
        c_min = ingresos.idxmin()
        print("Interpretacion de perfiles:")
        print(f"  - Cluster {c_max}: MAYOR nivel adquisitivo "
              f"(Ingreso: ${ingresos[c_max]:.2f}, "
              f"Gasto: ${medias.loc[c_max, 'gasto_acumulado']:.2f})")
        print(f"  - Cluster {c_min}: MENOR nivel adquisitivo "
              f"(Ingreso: ${ingresos[c_min]:.2f}, "
              f"Gasto: ${medias.loc[c_min, 'gasto_acumulado']:.2f})")
        print("\n" + "-" * 80)


def graficar_comparativa_categoricas(df, categorical_cols, output_dir):
    """Genera graficos de barras comparativos para cada variable categorica,
    incluyendo el promedio poblacional. Generalizado para k variable."""
    print("[3/5] Generando graficos de comparacion categorica con promedio poblacional...")

    clusters = sorted(df['cluster'].unique())
    paleta_grupos = {f'Cluster {int(c)}': PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
                     for c in clusters}
    paleta_grupos['Promedio Poblacional'] = COLOR_POBLACIONAL

    for col in categorical_cols:
        print(f"      Graficando variable: {col}...")

        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map(lambda c: f'Cluster {int(c)}')

        df_pop = df[col].value_counts(normalize=True).rename('porcentaje').reset_index()
        df_pop['porcentaje'] *= 100
        df_pop['grupo'] = 'Promedio Poblacional'

        df_plot = pd.concat([
            df_grouped[[col, 'porcentaje', 'grupo']],
            df_pop[[col, 'porcentaje', 'grupo']]
        ], ignore_index=True)

        unique_vals = df[col].nunique()

        plt.figure(figsize=(12, 6))

        if unique_vals > 8:
            top_categories = df[col].value_counts().index[:12]
            df_filtered = df_plot[df_plot[col].isin(top_categories)].copy()
            order = top_categories.tolist()

            sns.barplot(
                data=df_filtered,
                y=col,
                x='porcentaje',
                hue='grupo',
                palette=paleta_grupos,
                order=order
            )
            plt.xlabel('Porcentaje (%)')
            plt.ylabel(col)
            plt.title(f'Comparacion de {col.upper()} (Top 12 categorias vs Promedio Poblacional)')
        else:
            order = df[col].value_counts().index.tolist()
            sns.barplot(
                data=df_plot,
                x=col,
                y='porcentaje',
                hue='grupo',
                palette=paleta_grupos,
                order=order
            )
            plt.ylabel('Porcentaje (%)')
            plt.xlabel(col)
            plt.title(f'Comparacion de {col.upper()} por Cluster vs Promedio Poblacional')
            plt.xticks(rotation=15 if unique_vals > 3 else 0)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/comparativa_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")


def generar_dashboard_consolidado(df, categorical_cols, output_dir):
    """Genera un unico grafico (dashboard) con subplots comparando clusters y
    el promedio poblacional. Generalizado para k variable."""
    print("[4/5] Generando dashboard consolidado con promedio poblacional...")

    clusters = sorted(df['cluster'].unique())
    paleta_grupos = {f'Cluster {int(c)}': PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
                     for c in clusters}
    paleta_grupos['Promedio Poblacional'] = COLOR_POBLACIONAL

    cols_dashboard = ['sexo', 'clase_preferida', 'programaMillas', 'canal_compra']
    cols_dashboard = [c for c in cols_dashboard if c in categorical_cols]
    if len(cols_dashboard) < 4:
        cols_dashboard = categorical_cols[:4]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()

    for idx, col in enumerate(cols_dashboard[:4]):
        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map(lambda c: f'Cluster {int(c)}')

        df_pop = df[col].value_counts(normalize=True).rename('porcentaje').reset_index()
        df_pop['porcentaje'] *= 100
        df_pop['grupo'] = 'Promedio Poblacional'

        df_plot = pd.concat([
            df_grouped[[col, 'porcentaje', 'grupo']],
            df_pop[[col, 'porcentaje', 'grupo']]
        ], ignore_index=True)

        order = df[col].value_counts().index.tolist()

        sns.barplot(
            data=df_plot,
            x=col,
            y='porcentaje',
            hue='grupo',
            palette=paleta_grupos,
            order=order,
            ax=axes[idx]
        )

        axes[idx].set_title(f'Distribucion de {col.upper()}', fontsize=13, weight='bold', pad=10)
        axes[idx].set_ylabel('Porcentaje (%)' if idx % 2 == 0 else '')
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', labelrotation=10 if df[col].nunique() > 3 else 0)
        axes[idx].get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=len(paleta_grupos), fontsize=12, frameon=True)

    plt.suptitle('COMPARACION DE CLASES CATEGORICAS ENTRE CLUSTERS VS PROMEDIO POBLACIONAL',
                 fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/dashboard_comparativo_categoricas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard guardado en: {filename}")


def graficar_comparativa_numericas(df, numeric_cols, output_dir):
    """Genera graficos de distribucion (KDE) comparativos para cada variable
    numerica, incluyendo la media y el KDE poblacional. Generalizado para k."""
    print("[3.5/5] Generando graficos de comparacion numerica (distribuciones)...")

    clusters = sorted(df['cluster'].unique())
    for col in numeric_cols:
        print(f"      Graficando variable numerica: {col}...")

        plt.figure(figsize=(12, 6))

        for c in clusters:
            color = PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
            sub = df[df['cluster'] == c]
            sns.kdeplot(data=sub, x=col, fill=True, alpha=0.15,
                        color=color, label=f'Cluster {int(c)}', linewidth=2)
            mean_c = sub[col].mean()
            plt.axvline(mean_c, color=color, linestyle=':', linewidth=2,
                        label=f'Media C{int(c)}: {mean_c:.2f}')

        sns.kdeplot(data=df, x=col, fill=False, color=COLOR_POBLACIONAL,
                    label='Poblacion Total', linewidth=2.5, linestyle='--')
        mean_tot = df[col].mean()
        plt.axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2,
                    label=f'Media Global: {mean_tot:.2f}')

        plt.title(f'Distribucion de {col.upper()} por Cluster vs Poblacion Total')
        plt.xlabel(col)
        plt.ylabel('Densidad')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/comparativa_num_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")


def generar_dashboard_consolidado_numericas(df, numeric_cols, output_dir):
    """Genera un unico grafico (dashboard) con subplots de las variables
    numericas. Generalizado para k variable."""
    print("[4.5/5] Generando dashboard consolidado de variables numericas...")

    n_vars = len(numeric_cols)
    clusters = sorted(df['cluster'].unique())
    # Layout adaptativo: 3 cols x ceil(n/3) filas
    ncols = 3
    nrows = int(np.ceil(n_vars / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 5))
    axes = np.array(axes).ravel()

    for idx, col in enumerate(numeric_cols):
        for c in clusters:
            color = PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
            sub = df[df['cluster'] == c]
            sns.kdeplot(data=sub, x=col, fill=True, alpha=0.15,
                        color=color, label=f'Cluster {int(c)}', linewidth=2, ax=axes[idx])
            mean_c = sub[col].mean()
            axes[idx].axvline(mean_c, color=color, linestyle=':', linewidth=2,
                              label=f'Media C{int(c)}: {mean_c:.1f}')

        sns.kdeplot(data=df, x=col, fill=False, color=COLOR_POBLACIONAL,
                    label='Poblacion Total', linewidth=2.5, linestyle='--', ax=axes[idx])
        mean_tot = df[col].mean()
        axes[idx].axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2,
                          label=f'Media Global: {mean_tot:.1f}')

        axes[idx].set_title(f'Distribucion de {col.upper()}', fontsize=13, weight='bold', pad=10)
        axes[idx].set_ylabel('Densidad')
        axes[idx].set_xlabel('')
        axes[idx].legend(frameon=True, fontsize=8)

    if len(axes) > n_vars:
        for empty_idx in range(n_vars, len(axes)):
            axes[empty_idx].axis('off')

    plt.suptitle('DIFERENCIAS EN DISTRIBUCIONES NUMERICAS ENTRE CLUSTERS VS POBLACION TOTAL',
                 fontsize=16, weight='bold', y=1.01)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/dashboard_comparativo_numericas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard numerico guardado en: {filename}")


def graficar_boxplots_numericas(df, numeric_cols, output_dir, nombre_archivo="boxplots_numericas.png"):
    """Genera boxplots de las variables numericas por cluster con linea de
    media global. Generalizado para k variable."""
    print("[box] Generando boxplots de variables numericas...")

    clusters = sorted(df['cluster'].unique())
    palette = [PALETA_CLUSTERS.get(int(c), COLORES_LISTA[int(c) % len(COLORES_LISTA)])
               for c in clusters]

    n = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5 + 1, nrows * 4))
    axes = np.array(axes).ravel()

    for i, col in enumerate(numeric_cols):
        sns.boxplot(x='cluster', y=col, data=df, ax=axes[i], palette=palette)
        media_global = df[col].mean()
        axes[i].axhline(media_global, color='red', linestyle='--', linewidth=1.5,
                        label='Media Global')
        axes[i].set_title(col, weight='bold')
        axes[i].set_xlabel('Cluster')
        if i == 0:
            axes[i].legend(fontsize=8)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis('off')

    fig.suptitle('Variables numericas por Cluster vs Media Global',
                 fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    ruta = f"{output_dir}/{nombre_archivo}"
    plt.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Boxplots guardados en: {ruta}")


class ProgressBar:
    """
    Barra de progreso simple sin dependencias externas.
    Uso:
        pb = ProgressBar(total=100, desc="Procesando")
        for i in range(100):
            pb.update(1)

    O como context manager con iterable:
        for item in pb(iterable):
            ...
    """

    def __init__(self, total, desc="Progreso", bar_len=30):
        self.total = total
        self.desc = desc
        self.bar_len = bar_len
        self.n = 0
        self.start = time.time()
        self.last_update = 0
        self._draw(0)

    def _draw(self, n):
        elapsed = time.time() - self.start
        pct = n / self.total if self.total else 0
        filled = int(self.bar_len * pct)
        bar = "█" * filled + "░" * (self.bar_len - filled)
        if n > 0 and elapsed > 0:
            rate = n / elapsed
            eta = (self.total - n) / rate if rate > 0 else 0
            eta_str = f" ETA: {eta:.0f}s"
        else:
            eta_str = " ETA: --"
        sys.stdout.write(f"\r  {self.desc}: |{bar}| {n}/{self.total}  [{elapsed:.0f}s{eta_str}]")
        sys.stdout.flush()

    def update(self, n=1):
        self.n += n
        self._draw(self.n)
        if self.n >= self.total:
            print()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._draw(self.total)
        print()

    def __call__(self, iterable):
        self.total = len(iterable)
        self._draw(0)
        for i, item in enumerate(iterable, 1):
            yield item
            self._draw(i)
        print()
