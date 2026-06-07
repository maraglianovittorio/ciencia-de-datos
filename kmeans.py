"""
================================================================================
Análisis de Clientes con K-Means (2 Clusters)
================================================================================
Este script realiza un agrupamiento K-Means sobre la base de datos `Clientes.xlsx`.
Estandariza las variables numéricas, entrena un modelo con 2 clusters y genera
gráficos comparativos de las variables categóricas para analizar el perfil de
cada grupo de clientes.
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuración de estilo premium para gráficos
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

# Paleta de colores para los dos clusters (Azul elegante y Coral suave)
PALETA_CLUSTERS = {0: '#4A90E2', 1: '#FF6B6B'}
COLORES_LISTA = ['#4A90E2', '#FF6B6B']

OUTPUT_DIR = "TPI/graficos/kmeans"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def cargar_y_limpiar_datos():
    """Carga Clientes.xlsx y aplica limpieza inicial de filas vacías."""
    print("[1/5] Cargando base de datos Clientes.xlsx...")
    if not os.path.exists('Clientes.xlsx'):
        raise FileNotFoundError("No se encontró el archivo Clientes.xlsx en el directorio actual.")
    
    df = pd.read_excel('Clientes.xlsx')
    n_inicial = len(df)
    
    # Eliminar filas completamente vacías o donde id_cliente sea nulo (suele ser una fila en blanco al final)
    df = df.dropna(subset=['id_cliente'])
    n_final = len(df)
    
    print(f"      Dataset cargado correctamente. Filas leidas: {n_inicial} -> Filas validas: {n_final}")
    
    # Definir variables
    numeric_cols = [
        'edad', 'cant_vuelos', 'gasto_acumulado', 'gasto_acumulado_extra', 
        'cantidad_millas', 'ingreso_mensual', 'anticipacion_compra_promedio'
    ]
    
    categorical_cols = [
        'sexo', 'provincia', 'ocupacion', 'clase_preferida', 'programaMillas', 'canal_compra'
    ]
    
    # Tratar nulos remanentes si existieran
    for col in numeric_cols:
        if df[col].isnull().any():
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)
            
    for col in categorical_cols:
        if df[col].isnull().any():
            moda = df[col].mode()[0]
            df[col] = df[col].fillna(moda)
            
    # Corregir strings con caracteres extraños de codificación comunes si es necesario
    # Por ejemplo, para evitar problemas al graficar o imprimir
    for col in categorical_cols:
        df[col] = df[col].astype(str).str.strip()
        
    return df, numeric_cols, categorical_cols

def ejecutar_kmeans(df, numeric_cols):
    """Estandariza los datos y ejecuta el agrupamiento K-Means con 2 clusters."""
    print("[2/5] Estandarizando variables numericas y ejecutando K-Means...")
    
    # Estandarizar
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[numeric_cols])
    
    # K-Means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(x_scaled)
    
    print("      Algoritmo K-Means completado con exito (2 clusters creados).")
    
    # Calcular tamaños de los clusters
    tamaños = df['cluster'].value_counts()
    for cluster_id, count in tamaños.items():
        pct = (count / len(df)) * 100
        print(f"      - Cluster {cluster_id}: {count} clientes ({pct:.2f}%)")
        
    return df, kmeans, scaler

def generar_resumen_perfil(df, numeric_cols):
    """Calcula y muestra en consola el perfil promedio numérico de cada cluster."""
    print("\n" + "="*80)
    print(" PERFILES PROMEDIO DE LOS CLUSTERS (VARIABLES NUMERICAS)")
    print("="*80)
    
    resumen = df.groupby('cluster')[numeric_cols].mean().T
    resumen.columns = [f'Cluster {c} (Promedio)' for c in resumen.columns]
    
    # Agregar diferencia absoluta y relativa para entender qué los separa
    col_c0 = resumen.columns[0]
    col_c1 = resumen.columns[1]
    resumen['Diferencia Absoluta'] = (resumen[col_c0] - resumen[col_c1]).abs()
    
    print(resumen.round(2).to_string())
    print("="*80 + "\n")
    
    # Interpretación automática preliminar
    c0_ingreso = resumen.loc['ingreso_mensual', col_c0]
    c1_ingreso = resumen.loc['ingreso_mensual', col_c1]
    c0_gasto = resumen.loc['gasto_acumulado', col_c0]
    c1_gasto = resumen.loc['gasto_acumulado', col_c1]
    
    print("Interpretación de perfiles:")
    if c0_ingreso > c1_ingreso:
        print(f"  - Cluster 0: Clientes de MAYOR nivel adquisitivo (Ingreso mensual promedio: ${c0_ingreso:.2f}, Gasto acumulado: ${c0_gasto:.2f})")
        print(f"  - Cluster 1: Clientes de MENOR nivel adquisitivo (Ingreso mensual promedio: ${c1_ingreso:.2f}, Gasto acumulado: ${c1_gasto:.2f})")
    else:
        print(f"  - Cluster 0: Clientes de MENOR nivel adquisitivo (Ingreso mensual promedio: ${c0_ingreso:.2f}, Gasto acumulado: ${c0_gasto:.2f})")
        print(f"  - Cluster 1: Clientes de MAYOR nivel adquisitivo (Ingreso mensual promedio: ${c1_ingreso:.2f}, Gasto acumulado: ${c1_gasto:.2f})")
    print("\n" + "-"*80)

def graficar_comparativa_categoricas(df, categorical_cols):
    """Genera gráficos de barras comparativos para cada variable categórica, incluyendo el promedio poblacional."""
    print("[3/5] Generando graficos de comparacion categorica con promedio poblacional...")
    
    paleta_grupos = {
        'Cluster 0': '#4A90E2', 
        'Cluster 1': '#FF6B6B', 
        'Promedio Poblacional': '#95A5A6'
    }
    
    for col in categorical_cols:
        print(f"      Graficando variable: {col}...")
        
        # 1. Porcentaje dentro de cada cluster
        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})
        
        # 2. Porcentaje poblacional total
        df_pop = df[col].value_counts(normalize=True).rename('porcentaje').reset_index()
        df_pop['porcentaje'] *= 100
        df_pop['grupo'] = 'Promedio Poblacional'
        
        # Combinar ambos dataframes
        df_plot = pd.concat([
            df_grouped[[col, 'porcentaje', 'grupo']],
            df_pop[[col, 'porcentaje', 'grupo']]
        ], ignore_index=True)
        
        unique_vals = df[col].nunique()
        
        plt.figure(figsize=(12, 6))
        
        if unique_vals > 8:
            # Para columnas con muchas categorías (ej. provincia, ocupacion), usamos gráfico de barras horizontal y tomamos las top 12 más comunes
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
            # Gráfico de barras vertical estándar
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
        filename = f"{OUTPUT_DIR}/comparativa_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")

def generar_dashboard_consolidado(df, categorical_cols):
    """Genera un único gráfico (dashboard) con subplots comparando clusters y el promedio poblacional."""
    print("[4/5] Generando dashboard consolidado con promedio poblacional...")
    
    # Seleccionamos las 4 más relevantes para meterlas en una cuadrícula de 2x2
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
        # 1. Porcentaje dentro de cada cluster
        df_counts = df.groupby(['cluster', col]).size().rename('cantidad').reset_index()
        df_total_cluster = df.groupby('cluster').size().rename('total_cluster').reset_index()
        df_grouped = pd.merge(df_counts, df_total_cluster, on='cluster')
        df_grouped['porcentaje'] = (df_grouped['cantidad'] / df_grouped['total_cluster']) * 100
        df_grouped['grupo'] = df_grouped['cluster'].map({0: 'Cluster 0', 1: 'Cluster 1'})
        
        # 2. Porcentaje poblacional total
        df_pop = df[col].value_counts(normalize=True).rename('porcentaje').reset_index()
        df_pop['porcentaje'] *= 100
        df_pop['grupo'] = 'Promedio Poblacional'
        
        # Combinar
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
        
        # Eliminar leyenda de cada subplot para poner una sola global
        axes[idx].get_legend().remove()
        
    # Leyenda global
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.98), ncol=3, fontsize=12, frameon=True)
    
    plt.suptitle('COMPARACION DE CLASES CATEGORICAS ENTRE CLUSTERS VS PROMEDIO POBLACIONAL', fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/dashboard_comparativo_categoricas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard guardado en: {filename}")

def graficar_comparativa_numericas(df, numeric_cols):
    """Genera gráficos de distribución (KDE) comparativos para cada variable numérica, incluyendo la media."""
    print("[3.5/5] Generando graficos de comparacion numerica (distribuciones)...")
    
    for col in numeric_cols:
        print(f"      Graficando variable numerica: {col}...")
        
        plt.figure(figsize=(12, 6))
        
        # Graficar KDE de cada cluster y poblacion
        sns.kdeplot(data=df[df['cluster'] == 0], x=col, fill=True, alpha=0.15, color='#4A90E2', label='Cluster 0', linewidth=2)
        sns.kdeplot(data=df[df['cluster'] == 1], x=col, fill=True, alpha=0.15, color='#FF6B6B', label='Cluster 1', linewidth=2)
        sns.kdeplot(data=df, x=col, fill=False, color='#95A5A6', label='Poblacion Total', linewidth=2.5, linestyle='--')
        
        # Calcular medias
        mean_c0 = df[df['cluster'] == 0][col].mean()
        mean_c1 = df[df['cluster'] == 1][col].mean()
        mean_tot = df[col].mean()
        
        # Dibujar lineas verticales de medias
        plt.axvline(mean_c0, color='#4A90E2', linestyle=':', linewidth=2, label=f'Media Cluster 0: {mean_c0:.2f}')
        plt.axvline(mean_c1, color='#FF6B6B', linestyle=':', linewidth=2, label=f'Media Cluster 1: {mean_c1:.2f}')
        plt.axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Media Global: {mean_tot:.2f}')
        
        plt.title(f'Distribucion de {col.upper()} por Cluster vs Poblacion Total')
        plt.xlabel(col)
        plt.ylabel('Densidad')
        plt.legend(frameon=True, facecolor='white', framealpha=0.9)
        plt.tight_layout()
        
        filename = f"{OUTPUT_DIR}/comparativa_num_{col}.png"
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"      [OK] Guardado en: {filename}")

def generar_dashboard_consolidado_numericas(df, numeric_cols):
    """Genera un único gráfico (dashboard) con subplots de las variables numéricas."""
    print("[4.5/5] Generando dashboard consolidado de variables numericas...")
    
    n_vars = len(numeric_cols)
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Graficar KDE
        sns.kdeplot(data=df[df['cluster'] == 0], x=col, fill=True, alpha=0.15, color='#4A90E2', label='Cluster 0', linewidth=2, ax=axes[idx])
        sns.kdeplot(data=df[df['cluster'] == 1], x=col, fill=True, alpha=0.15, color='#FF6B6B', label='Cluster 1', linewidth=2, ax=axes[idx])
        sns.kdeplot(data=df, x=col, fill=False, color='#95A5A6', label='Poblacion Total', linewidth=2, linestyle='--', ax=axes[idx])
        
        # Calcular medias
        mean_c0 = df[df['cluster'] == 0][col].mean()
        mean_c1 = df[df['cluster'] == 1][col].mean()
        mean_tot = df[col].mean()
        
        # Dibujar medias
        axes[idx].axvline(mean_c0, color='#4A90E2', linestyle=':', linewidth=2, label=f'Media C0: {mean_c0:.1f}')
        axes[idx].axvline(mean_c1, color='#FF6B6B', linestyle=':', linewidth=2, label=f'Media C1: {mean_c1:.1f}')
        axes[idx].axvline(mean_tot, color='#7F8C8D', linestyle='--', linewidth=2, label=f'Media Global: {mean_tot:.1f}')
        
        axes[idx].set_title(f'Distribucion de {col.upper()}', fontsize=13, weight='bold', pad=10)
        axes[idx].set_ylabel('Densidad')
        axes[idx].set_xlabel('')
        axes[idx].legend(frameon=True, fontsize=10)
        
    # El subplot 8 (índice 7) está vacío, lo desactivamos
    if len(axes) > n_vars:
        for empty_idx in range(n_vars, len(axes)):
            axes[empty_idx].axis('off')
            
    plt.suptitle('DIFERENCIAS EN DISTRIBUCIONES NUMERICAS ENTRE CLUSTERS VS POBLACION TOTAL', fontsize=18, weight='bold', y=1.01)
    plt.tight_layout()
    
    filename = f"{OUTPUT_DIR}/dashboard_comparativo_numericas.png"
    plt.savefig(filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"      [OK] Dashboard numerico guardado en: {filename}")

def guardar_resultados_csv(df):
    """Guarda el dataset con la asignación de clusters a un nuevo archivo CSV."""
    print("[5/5] Guardando resultados en clientes_con_clusters.csv...")
    output_path = 'clientes_con_clusters.csv'
    df.to_csv(output_path, index=False)
    print(f"      [OK] Resultados guardados en: {output_path}")

def main():
    print("="*80)
    print(" PROCESAMIENTO K-MEANS - SEGMENTACION DE CLIENTES (2 CLUSTERS)")
    print("="*80)
    
    try:
        # 1. Cargar y Limpiar
        df, numeric_cols, categorical_cols = cargar_y_limpiar_datos()
        
        # 2. Ejecutar K-Means
        df, model, scaler = ejecutar_kmeans(df, numeric_cols)
        
        # 3. Mostrar Resumen de Perfiles
        generar_resumen_perfil(df, numeric_cols)
        
        # 4. Generar Gráficos Individuales de Categorías
        graficar_comparativa_categoricas(df, categorical_cols)
        
        # 4b. Generar Gráficos Individuales de Numéricas
        graficar_comparativa_numericas(df, numeric_cols)
        
        # 5. Generar Dashboard Consolidado Categorias
        generar_dashboard_consolidado(df, categorical_cols)
        
        # 5b. Generar Dashboard Consolidado Numéricas
        generar_dashboard_consolidado_numericas(df, numeric_cols)
        
        # 6. Exportar Resultados
        guardar_resultados_csv(df)
        
        print("\n" + "="*80)
        print(" PROCESAMIENTO FINALIZADO EXITOSAMENTE")
        print("="*80)
        print(f" Los graficos comparativos se han guardado en: {OUTPUT_DIR}/")
        print(" Archivos generados:")
        print(f"   - {OUTPUT_DIR}/dashboard_comparativo_categoricas.png  (Dashboard general categorico)")
        print(f"   - {OUTPUT_DIR}/dashboard_comparativo_numericas.png  (Dashboard general numerico)")
        for col in categorical_cols:
            print(f"   - {OUTPUT_DIR}/comparativa_{col}.png")
        for col in numeric_cols:
            print(f"   - {OUTPUT_DIR}/comparativa_num_{col}.png")
        print("   - clientes_con_clusters.csv  (Base de datos con columna 'cluster' asignada)")
        print("="*80)
        
    except Exception as e:
        print(f"\n[ERROR] Ocurrio un problema durante la ejecucion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
