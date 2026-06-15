"""
================================================================================
Utilidades compartidas para clustering de Clientes
================================================================================
Funciones reutilizables por cluster_jerarquico.py, cluster_bietapico.py
y eventualmente otros algoritmos que se apliquen sobre Clientes.xlsx.
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


PALETA_CLUSTERS = {0: '#4A90E2', 1: '#FF6B6B'}
COLORES_LISTA = ['#4A90E2', '#FF6B6B']
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


def generar_resumen_perfil(df, numeric_cols):
    """Imprime en consola el perfil promedio numérico de cada cluster."""
    print("\n" + "=" * 80)
    print(" PERFILES PROMEDIO DE LOS CLUSTERS (VARIABLES NUMERICAS)")
    print("=" * 80)

    resumen = df.groupby('cluster')[numeric_cols].mean().T
    resumen.columns = [f'Cluster {c} (Promedio)' for c in resumen.columns]
    col_c0 = resumen.columns[0]
    col_c1 = resumen.columns[1]
    resumen['Diferencia Absoluta'] = (resumen[col_c0] - resumen[col_c1]).abs()

    print(resumen.round(2).to_string())
    print("=" * 80 + "\n")

    c0_ingreso = resumen.loc['ingreso_mensual', col_c0]
    c1_ingreso = resumen.loc['ingreso_mensual', col_c1]
    c0_gasto = resumen.loc['gasto_acumulado', col_c0]
    c1_gasto = resumen.loc['gasto_acumulado', col_c1]

    print("Interpretacion de perfiles:")
    if c0_ingreso > c1_ingreso:
        print(f"  - Cluster 0: Clientes de MAYOR nivel adquisitivo (Ingreso mensual promedio: ${c0_ingreso:.2f}, Gasto acumulado: ${c0_gasto:.2f})")
        print(f"  - Cluster 1: Clientes de MENOR nivel adquisitivo (Ingreso mensual promedio: ${c1_ingreso:.2f}, Gasto acumulado: ${c1_gasto:.2f})")
    else:
        print(f"  - Cluster 0: Clientes de MENOR nivel adquisitivo (Ingreso mensual promedio: ${c0_ingreso:.2f}, Gasto acumulado: ${c0_gasto:.2f})")
        print(f"  - Cluster 1: Clientes de MAYOR nivel adquisitivo (Ingreso mensual promedio: ${c1_ingreso:.2f}, Gasto acumulado: ${c1_gasto:.2f})")
    print("\n" + "-" * 80)


def graficar_comparativa_categoricas(df, categorical_cols, output_dir):
    """Genera graficos de barras comparativos para cada variable categorica, incluyendo el promedio poblacional."""
    print("[3/5] Generando graficos de comparacion categorica con promedio poblacional...")

    paleta_grupos = {
        'Cluster 0': '#4A90E2',
        'Cluster 1': '#FF6B6B',
        'Promedio Poblacional': '#95A5A6'
    }

    for col in categorical_cols:
        print(f"      Graficando variable: {col}...")

        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})

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
        filename = f"{output_dir}/comparativa_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")


def generar_dashboard_consolidado(df, categorical_cols, output_dir):
    """Genera un unico grafico (dashboard) con subplots comparando clusters y el promedio poblacional."""
    print("[4/5] Generando dashboard consolidado con promedio poblacional...")

    cols_dashboard = ['sexo', 'clase_preferida', 'programaMillas', 'canal_compra']
    cols_dashboard = [c for c in cols_dashboard if c in categorical_cols]
    if len(cols_dashboard) < 4:
        cols_dashboard = categorical_cols[:4]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.ravel()

    paleta_grupos = {
        'Cluster 0': '#4A90E2',
        'Cluster 1': '#FF6B6B',
        'Promedio Poblacional': '#95A5A6'
    }

    for idx, col in enumerate(cols_dashboard[:4]):
        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})

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
               bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12, frameon=True)

    plt.suptitle('COMPARACION DE CLASES CATEGORICAS ENTRE CLUSTERS VS PROMEDIO POBLACIONAL', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()

    filename = f"{output_dir}/dashboard_comparativo_categoricas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard guardado en: {filename}")


def graficar_comparativa_numericas(df, numeric_cols, output_dir):
    """Genera graficos de distribucion (KDE) comparativos para cada variable numerica, incluyendo la media."""
    print("[3.5/5] Generando graficos de comparacion numerica (distribuciones)...")

    for col in numeric_cols:
        print(f"      Graficando variable numerica: {col}...")

        plt.figure(figsize=(12, 6))

        sns.kdeplot(data=df[df['cluster'] == 0], x=col, fill=True, alpha=0.15, color='#4A90E2', label='Cluster 0', linewidth=2)
        sns.kdeplot(data=df[df['cluster'] == 1], x=col, fill=True, alpha=0.15, color='#FF6B6B', label='Cluster 1', linewidth=2)
        sns.kdeplot(data=df, x=col, fill=False, color='#95A5A6', label='Poblacion Total', linewidth=2.5, linestyle='--')

        mean_c0 = df[df['cluster'] == 0][col].mean()
        mean_c1 = df[df['cluster'] == 1][col].mean()
        mean_tot = df[col].mean()

        plt.axvline(mean_c0, color='#4A90E2', linestyle=':', linewidth=2, label=f'Media Cluster 0: {mean_c0:.2f}')
        plt.axvline(mean_c1, color='#FF6B6B', linestyle=':', linewidth=2, label=f'Media Cluster 1: {mean_c1:.2f}')
        plt.axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Media Global: {mean_tot:.2f}')

        plt.title(f'Distribucion de {col.upper()} por Cluster vs Poblacion Total')
        plt.xlabel(col)
        plt.ylabel('Densidad')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()

        filename = f"{output_dir}/comparativa_num_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")


def generar_dashboard_consolidado_numericas(df, numeric_cols, output_dir):
    """Genera un unico grafico (dashboard) con subplots de las variables numericas."""
    print("[4.5/5] Generando dashboard consolidado de variables numericas...")

    n_vars = len(numeric_cols)
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.ravel()

    for idx, col in enumerate(numeric_cols):
        sns.kdeplot(data=df[df['cluster'] == 0], x=col, fill=True, alpha=0.15, color='#4A90E2', label='Cluster 0', linewidth=2, ax=axes[idx])
        sns.kdeplot(data=df[df['cluster'] == 1], x=col, fill=True, alpha=0.15, color='#FF6B6B', label='Cluster 1', linewidth=2, ax=axes[idx])
        sns.kdeplot(data=df, x=col, fill=False, color='#95A5A6', label='Poblacion Total', linewidth=2.5, linestyle='--', ax=axes[idx])

        mean_c0 = df[df['cluster'] == 0][col].mean()
        mean_c1 = df[df['cluster'] == 1][col].mean()
        mean_tot = df[col].mean()

        axes[idx].axvline(mean_c0, color='#4A90E2', linestyle=':', linewidth=2, label=f'Media C0: {mean_c0:.1f}')
        axes[idx].axvline(mean_c1, color='#FF6B6B', linestyle=':', linewidth=2, label=f'Media C1: {mean_c1:.1f}')
        axes[idx].axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Media Global: {mean_tot:.1f}')

        axes[idx].set_title(f'Distribucion de {col.upper()}', fontsize=13, weight='bold', pad=10)
        axes[idx].set_ylabel('Densidad')
        axes[idx].set_xlabel('')
        axes[idx].legend(frameon=True, fontsize=10)

    if len(axes) > n_vars:
        for empty_idx in range(n_vars, len(axes)):
            axes[empty_idx].axis('off')

    plt.suptitle('DIFERENCIAS EN DISTRIBUCIONES NUMERICAS ENTRE CLUSTERS VS POBLACION TOTAL', fontsize=18, weight='bold', y=1.01)
    plt.tight_layout()

    filename = f"{output_dir}/dashboard_comparativo_numericas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard numerico guardado en: {filename}")


def exportar_silhouette(score, algoritmo, n_clusters, ruta_json, sample_size=None):
    """Persiste el coeficiente de Silhouette a un archivo JSON."""
    payload = {
        "algoritmo": algoritmo,
        "n_clusters": n_clusters,
        "silhouette": round(float(score), 4),
    }
    if sample_size is not None:
        payload["sample_size"] = int(sample_size)
        payload["muestreado"] = True
    else:
        payload["muestreado"] = False

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"      [OK] Silhouette exportado a: {ruta_json}")


def imprimir_silhouette(score, algoritmo, sample_size=None):
    """Imprime en consola el coeficiente de Silhouette."""
    extra = f" (muestra={sample_size})" if sample_size else ""
    print(f"      Coeficiente de Silhouette [{algoritmo}]{extra}: {score:.4f}")


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
