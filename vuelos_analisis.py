"""
=============================================================================
TP INTEGRADOR 2026 - DS Airlines
Gestión Proactiva de Demoras Aéreas
=============================================================================
Análisis Exploratorio, Limpieza y Transformación de Datos.

Metodología: CRISP-DM
  - Fase 2: Entendimiento de los datos
  - Fase 3: Preparación de los datos (limpieza + transformación)

Dataset: Vuelos.xlsx (15.000 registros, 15 variables)
Variable objetivo: demora (0 = No demorado, 1 = Demorado)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo para evitar errores de display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from itertools import combinations
import os

# Configuración de gráficos
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
sns.set_theme(style="whitegrid", palette="muted")

# Directorio de salida para gráficos
OUTPUT_DIR = "TPI/graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# UTILIDADES DE CLASIFICACIÓN DE VARIABLES
# =============================================================================

def clasificar_variables(df, excluir=None):
    """
    Clasifica las columnas del DataFrame en numéricas y categóricas.
    Detecta automáticamente el tipo sin hardcodear nombres.

    Parámetros:
        df: DataFrame
        excluir: lista de columnas a excluir (ej: id, target)

    Retorna:
        tuple: (lista_numericas, lista_categoricas)
    """
    if excluir is None:
        excluir = []

    columnas = [c for c in df.columns if c not in excluir]

    numericas = df[columnas].select_dtypes(include=[np.number]).columns.tolist()
    categoricas = df[columnas].select_dtypes(
        include=['object', 'bool', 'string', 'category']
    ).columns.tolist()

    return numericas, categoricas


# =============================================================================
# FASE 2: ENTENDIMIENTO DE LOS DATOS
# =============================================================================

def cargar_datos(ruta='TPI/Vuelos.xlsx'):
    """Carga el dataset de vuelos desde el archivo Excel."""
    # Soporta ejecución desde raíz del repo o desde la carpeta TPI.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rutas_candidatas = [
        ruta,
        os.path.join(script_dir, 'Vuelos.xlsx'),
        os.path.join(script_dir, '..', ruta),
    ]

    ruta_excel = next((r for r in rutas_candidatas if os.path.exists(r)), None)
    if ruta_excel is None:
        raise FileNotFoundError(
            "No se encontró el archivo Excel. Rutas intentadas: "
            + ", ".join(rutas_candidatas)
        )

    df = pd.read_excel(ruta_excel)
    print("=" * 70)
    print("FASE 2: ENTENDIMIENTO DE LOS DATOS")
    print("=" * 70)
    print(f"\n[INFO] Dataset cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    return df


def descripcion_general(df):
    """Muestra la descripción general de cada variable."""
    print("\n" + "-" * 70)
    print("2.1 DESCRIPCIÓN GENERAL DEL DATASET")
    print("-" * 70)
    print(f"\n{'Variable':<30} {'Tipo':<15} {'No Nulos':<12} {'Nulos':<8} {'Únicos':<8}")
    print("-" * 73)
    for col in df.columns:
        nulos = df[col].isnull().sum()
        print(
            f"{col:<30} {str(df[col].dtype):<15} "
            f"{df[col].notna().sum():<12} {nulos:<8} {df[col].nunique():<8}"
        )

    total_nulos = df.isnull().sum().sum()
    total_celdas = df.shape[0] * df.shape[1]
    print(f"\n[INFO] Valores nulos totales: {total_nulos} / {total_celdas} "
          f"({total_nulos / total_celdas * 100:.2f}%)")


# =============================================================================
# LIMPIEZA DE DATOS
# =============================================================================

def detectar_nulos(df):
    """Detecta y reporta valores ausentes (NAs) por columna."""
    print("\n" + "-" * 70)
    print("3.1 DETECCIÓN DE VALORES AUSENTES (NAs)")
    print("-" * 70)

    nulos = df.isnull().sum()
    nulos_pct = (nulos / len(df) * 100).round(2)
    resumen = pd.DataFrame({
        'Nulos': nulos,
        '% del Total': nulos_pct
    }).sort_values('Nulos', ascending=False)

    if nulos.sum() == 0:
        print("\n  ✅ No se encontraron valores ausentes en ninguna columna.")
    else:
        print(f"\n{'Columna':<30} {'Nulos':<10} {'%':<8}")
        print("-" * 48)
        for col, row in resumen.iterrows():
            if row['Nulos'] > 0:
                print(f"  {col:<28} {int(row['Nulos']):<10} {row['% del Total']:.2f}%")
        print(f"\n  [!] Total de celdas con NA: {nulos.sum()}")

    return resumen


def detectar_duplicados(df):
    """Detecta filas duplicadas en el dataset."""
    print("\n" + "-" * 70)
    print("3.2 DETECCIÓN DE DUPLICADOS")
    print("-" * 70)

    duplicados = df.duplicated().sum()
    if duplicados == 0:
        print(f"\n  ✅ No se encontraron filas duplicadas (de {len(df)} registros).")
    else:
        print(f"\n  ⚠️  Se encontraron {duplicados} filas duplicadas ({duplicados/len(df)*100:.2f}%).")
        print("  Primeros ejemplos:")
        print(df[df.duplicated(keep=False)].head(6).to_string())

    return duplicados


def detectar_outliers(df, excluir=None):
    """
    Detecta outliers usando IQR y Z-Score para todas las variables numéricas.

    Parámetros:
        df: DataFrame
        excluir: columnas a excluir del análisis (ej: ['id_vuelo', 'demora'])

    Retorna:
        dict con información de outliers por columna
    """
    print("\n" + "-" * 70)
    print("3.3 DETECCIÓN DE OUTLIERS")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)

    # --- Método IQR ---
    print("\n--- Método IQR (Rango Intercuartílico) ---")
    print(
        f"  {'Variable':<25} {'Q1':>8} {'Q3':>8} {'IQR':>8} "
        f"{'Lím Inf':>10} {'Lím Sup':>10} {'Outliers':>10} {'%':>6}"
    )
    print("  " + "-" * 87)

    outlier_info = {}
    for col in numericas:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        outliers = ((df[col] < lim_inf) | (df[col] > lim_sup)).sum()
        pct = outliers / len(df) * 100
        outlier_info[col] = {
            'count': outliers,
            'pct': pct,
            'lim_inf': lim_inf,
            'lim_sup': lim_sup,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        }
        print(
            f"  {col:<25} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} "
            f"{lim_inf:>10.2f} {lim_sup:>10.2f} {outliers:>10} {pct:>5.1f}%"
        )

    # --- Método Z-Score ---
    print("\n--- Método Z-Score (|z| > 3) ---")
    print(f"  {'Variable':<25} {'Outliers Z-Score':>18} {'%':>6}")
    print("  " + "-" * 51)
    for col in numericas:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers_z = (z_scores > 3).sum()
        pct_z = outliers_z / len(df) * 100
        print(f"  {col:<25} {outliers_z:>18} {pct_z:>5.1f}%")

    # Resumen
    total_con_outliers = sum(1 for v in outlier_info.values() if v['count'] > 0)
    print(f"\n  [INFO] {total_con_outliers} de {len(numericas)} variables numéricas "
          f"presentan outliers (método IQR).")

    return outlier_info


# =============================================================================
# ANÁLISIS DE PROPORCIONES (VARIABLES CATEGÓRICAS)
# =============================================================================

def analisis_proporciones(df, excluir=None):
    """
    Para cada variable categórica, calcula la proporción de cada clase
    en porcentaje, ordenado de mayor a menor.

    Parámetros:
        df: DataFrame
        excluir: columnas a excluir
    """
    print("\n" + "-" * 70)
    print("3.4 ANÁLISIS DE PROPORCIONES (Variables Categóricas)")
    print("-" * 70)

    _, categoricas = clasificar_variables(df, excluir=excluir)

    if not categoricas:
        print("\n  No se encontraron variables categóricas.")
        return

    for col in categoricas:
        print(f"\n  >>> {col} ({df[col].nunique()} categorías)")
        print("  " + "-" * 40)

        conteo = df[col].value_counts()
        proporciones = (conteo / len(df) * 100).round(2)

        for categoria, pct in proporciones.items():
            print(f"    {str(categoria):<30} {pct:>6.2f}%")


def analisis_demora_por_aeropuerto(df, target='demora', output_dir=OUTPUT_DIR):
    """
    Calcula el porcentaje de vuelos demorados por aeropuerto.

    El análisis se adapta a dos esquemas comunes del dataset:
      1) Una sola columna de aeropuerto.
      2) Columnas separadas de origen/destino (se unifican en un mismo ranking).
    """
    print("\n" + "-" * 70)
    print("3.5 DEMORAS POR AEROPUERTO")
    print("-" * 70)

    if target not in df.columns:
        print(f"\n  [!] No se encontró la variable objetivo '{target}'.")
        return None

    def _normalizar_columna(nombre):
        return nombre.lower().strip()

    def _es_demorado(serie):
        if pd.api.types.is_numeric_dtype(serie):
            return serie.eq(1)

        texto = serie.astype(str).str.lower().str.strip()
        valores_true = {'1', 'si', 'sí', 'true', 'demorado', 'delay', 'delayed'}
        return texto.isin(valores_true)

    # Detecta posibles columnas de aeropuerto evitando hardcodear nombres exactos.
    candidatas = []
    nombres_directos = {
        'aeropuerto', 'origen', 'destino',
        'origen_iata', 'destino_iata',
        'airport_origin', 'airport_destination'
    }
    for col in df.columns:
        nombre = _normalizar_columna(col)
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            continue
        if 'aeropuerto' in nombre or nombre in nombres_directos:
            candidatas.append(col)

    if not candidatas:
        print("\n  [!] No se detectaron columnas de aeropuerto para calcular el análisis.")
        return None

    # Si hay varias columnas de aeropuerto (p.ej. origen/destino),
    # unificamos para obtener un porcentaje por aeropuerto global.
    if len(candidatas) == 1:
        largo = df[[candidatas[0], target]].rename(columns={candidatas[0]: 'aeropuerto'})
    else:
        largo = df[candidatas + [target]].melt(
            id_vars=target,
            value_vars=candidatas,
            value_name='aeropuerto'
        )[[target, 'aeropuerto']]

    largo = largo.dropna(subset=['aeropuerto'])

    resumen = (
        largo.assign(demorado=_es_demorado(largo[target]).astype(int))
        .groupby('aeropuerto', as_index=False)
        .agg(
            total_vuelos=('demorado', 'count'),
            vuelos_demorados=('demorado', 'sum')
        )
    )

    resumen['pct_demorados'] = (resumen['vuelos_demorados'] / resumen['total_vuelos'] * 100).round(2)
    resumen = resumen.sort_values('pct_demorados', ascending=False)

    if resumen.empty:
        print("\n  [!] No hay datos válidos para calcular demoras por aeropuerto.")
        return None

    print(f"\n{'Aeropuerto':<35} {'Demorados':>10} {'Total':>10} {'% Demora':>10}")
    print("-" * 70)
    for _, row in resumen.iterrows():
        print(
            f"{str(row['aeropuerto']):<35} "
            f"{int(row['vuelos_demorados']):>10} "
            f"{int(row['total_vuelos']):>10} "
            f"{row['pct_demorados']:>9.2f}%"
        )

    csv_path = f'{output_dir}/demora_por_aeropuerto.csv'
    resumen.to_csv(csv_path, index=False)
    print(f"\n   [+] demora_por_aeropuerto.csv")

    fig, ax = plt.subplots(figsize=(11, max(5, len(resumen) * 0.4)))
    sns.barplot(data=resumen, y='aeropuerto', x='pct_demorados', color='tomato', ax=ax)
    ax.set_title('Porcentaje de vuelos demorados por aeropuerto', fontweight='bold')
    ax.set_xlabel('% de vuelos demorados')
    ax.set_ylabel('Aeropuerto')

    for i, v in enumerate(resumen['pct_demorados']):
        ax.text(v + 0.2, i, f'{v:.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/demora_por_aeropuerto.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [+] demora_por_aeropuerto.png")

    return resumen


def _detectar_columnas_aeropuerto(df):
    """Detecta columnas que representan aeropuerto (origen/destino o única)."""
    candidatas = []
    nombres_directos = {
        'aeropuerto', 'origen', 'destino',
        'origen_iata', 'destino_iata',
        'airport_origin', 'airport_destination'
    }

    for col in df.columns:
        nombre = col.lower().strip()
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            continue
        if 'aeropuerto' in nombre or nombre in nombres_directos:
            candidatas.append(col)

    return candidatas


def _serie_demorada(serie):
    """Convierte una columna de demora a indicador binario (1=demorado)."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.eq(1).astype(int)

    texto = serie.astype(str).str.lower().str.strip()
    valores_true = {'1', 'si', 'sí', 'true', 'demorado', 'delay', 'delayed'}
    return texto.isin(valores_true).astype(int)


def _expandir_aeropuertos(df, columnas_aeropuerto, columnas_extra=None):
    """Pasa de formato ancho a largo para consolidar métricas por aeropuerto."""
    if columnas_extra is None:
        columnas_extra = []

    if len(columnas_aeropuerto) == 1:
        largo = df[[columnas_aeropuerto[0]] + columnas_extra].rename(
            columns={columnas_aeropuerto[0]: 'aeropuerto'}
        )
    else:
        largo = df[columnas_aeropuerto + columnas_extra].melt(
            id_vars=columnas_extra,
            value_vars=columnas_aeropuerto,
            value_name='aeropuerto'
        )[columnas_extra + ['aeropuerto']]

    return largo.dropna(subset=['aeropuerto'])


def grafico_vuelos_y_demora_por_aeropuerto(df, target='demora', top_n=20, output_dir=OUTPUT_DIR):
    """
    Compara cantidad de vuelos y demora por aeropuerto en un mismo gráfico.
    """
    print("\n" + "-" * 70)
    print("3.6 VUELOS Y DEMORA POR AEROPUERTO")
    print("-" * 70)

    if target not in df.columns:
        print(f"\n  [!] No se encontró la variable objetivo '{target}'.")
        return None

    columnas_aeropuerto = _detectar_columnas_aeropuerto(df)
    if not columnas_aeropuerto:
        print("\n  [!] No se detectaron columnas de aeropuerto para este análisis.")
        return None

    largo = _expandir_aeropuertos(df, columnas_aeropuerto, columnas_extra=[target])
    largo['demorado'] = _serie_demorada(largo[target])

    resumen = (
        largo.groupby('aeropuerto', as_index=False)
        .agg(
            total_vuelos=('aeropuerto', 'count'),
            pct_demorados=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('total_vuelos', ascending=False)
    )
    resumen['pct_demorados'] = resumen['pct_demorados'].round(2)

    if resumen.empty:
        print("\n  [!] No hay datos válidos para generar el gráfico comparativo.")
        return None

    top = resumen.head(top_n).sort_values('total_vuelos', ascending=True)

    print(f"\nTop {min(top_n, len(resumen))} aeropuertos por cantidad de vuelos:")
    print(f"{'Aeropuerto':<15} {'Vuelos':>10} {'% Demora':>12}")
    print("-" * 40)
    for _, row in top.sort_values('total_vuelos', ascending=False).iterrows():
        print(f"{row['aeropuerto']:<15} {int(row['total_vuelos']):>10} {row['pct_demorados']:>11.2f}%")

    top.to_csv(f'{output_dir}/vuelos_y_demora_por_aeropuerto.csv', index=False)
    print("\n   [+] vuelos_y_demora_por_aeropuerto.csv")

    fig, ax1 = plt.subplots(figsize=(12, max(6, len(top) * 0.45)))
    ax2 = ax1.twiny()

    ax1.barh(top['aeropuerto'], top['total_vuelos'], color='#4C78A8', alpha=0.85)
    ax1.set_xlabel('Cantidad de vuelos')
    ax1.set_ylabel('Aeropuerto')

    ax2.plot(top['pct_demorados'], top['aeropuerto'], color='#E45756', marker='o', linewidth=2)
    ax2.set_xlabel('% de vuelos demorados')

    ax1.set_title('Aeropuertos: volumen de vuelos vs nivel de demora', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vuelos_y_demora_por_aeropuerto.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [+] vuelos_y_demora_por_aeropuerto.png")

    return resumen


def grafico_ocupacion_por_aeropuerto(df, col_ocupacion='ocupacion_vuelo', top_n=20, output_dir=OUTPUT_DIR):
    """
    Compara la ocupación promedio de vuelos por aeropuerto.
    """
    print("\n" + "-" * 70)
    print("3.7 OCUPACIÓN PROMEDIO POR AEROPUERTO")
    print("-" * 70)

    if col_ocupacion not in df.columns:
        print(f"\n  [!] No se encontró la columna de ocupación '{col_ocupacion}'.")
        return None

    columnas_aeropuerto = _detectar_columnas_aeropuerto(df)
    if not columnas_aeropuerto:
        print("\n  [!] No se detectaron columnas de aeropuerto para este análisis.")
        return None

    largo = _expandir_aeropuertos(df, columnas_aeropuerto, columnas_extra=[col_ocupacion])
    largo = largo.dropna(subset=[col_ocupacion])

    resumen = (
        largo.groupby('aeropuerto', as_index=False)
        .agg(
            total_vuelos=('aeropuerto', 'count'),
            ocupacion_promedio=(col_ocupacion, 'mean')
        )
        .sort_values('ocupacion_promedio', ascending=False)
    )

    if resumen.empty:
        print("\n  [!] No hay datos válidos para ocupación por aeropuerto.")
        return None

    top = resumen.head(top_n).sort_values('ocupacion_promedio', ascending=True)

    print(f"\nTop {min(top_n, len(resumen))} aeropuertos por ocupación promedio:")
    print(f"{'Aeropuerto':<15} {'Ocupación':>12} {'Vuelos':>10}")
    print("-" * 40)
    for _, row in top.sort_values('ocupacion_promedio', ascending=False).iterrows():
        print(f"{row['aeropuerto']:<15} {row['ocupacion_promedio']:>11.2f} {int(row['total_vuelos']):>10}")

    top.to_csv(f'{output_dir}/ocupacion_por_aeropuerto.csv', index=False)
    print("\n   [+] ocupacion_por_aeropuerto.csv")

    fig, ax = plt.subplots(figsize=(11, max(6, len(top) * 0.45)))
    bars = ax.barh(top['aeropuerto'], top['ocupacion_promedio'], color='#59A14F', alpha=0.9)
    ax.set_title('Ocupación promedio por aeropuerto', fontweight='bold')
    ax.set_xlabel('Ocupación promedio')
    ax.set_ylabel('Aeropuerto')

    for bar in bars:
        valor = bar.get_width()
        ax.text(valor + 0.2, bar.get_y() + bar.get_height() / 2, f'{valor:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/ocupacion_por_aeropuerto.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [+] ocupacion_por_aeropuerto.png")

    return resumen


def analisis_horario_ocupacion_demora(
    df,
    col_hora='hora_salida_programada',
    col_ocupacion='ocupacion_vuelo',
    target='demora',
    output_dir=OUTPUT_DIR
):
    """
    Analiza si la hora de salida influye en la ocupación o en la demora.
    """
    print("\n" + "-" * 70)
    print("3.8 ANÁLISIS HORARIO: INFLUENCIA EN OCUPACIÓN Y DEMORA")
    print("-" * 70)

    faltantes = [c for c in [col_hora, col_ocupacion, target] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas para análisis horario: {faltantes}")
        return None

    base = df[[col_hora, col_ocupacion, target]].copy()

    # Normaliza hora desde distintos formatos (ej. '14:35', '14', datetime, etc.).
    hora_txt = base[col_hora].astype(str).str.extract(r'(\d{1,2})', expand=False)
    base['hora'] = pd.to_numeric(hora_txt, errors='coerce')
    base = base[base['hora'].between(0, 23, inclusive='both')]

    if base.empty:
        print("\n  [!] No se pudieron interpretar horas válidas (0-23).")
        return None

    base['demorado'] = _serie_demorada(base[target])

    resumen = (
        base.groupby('hora', as_index=False)
        .agg(
            vuelos=('hora', 'count'),
            ocupacion_promedio=(col_ocupacion, 'mean'),
            pct_demorados=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('hora')
    )
    resumen['pct_demorados'] = resumen['pct_demorados'].round(2)

    corr_hora_ocup = resumen['hora'].corr(resumen['ocupacion_promedio'])
    corr_hora_demora = resumen['hora'].corr(resumen['pct_demorados'])

    print(f"\nCorrelación hora vs ocupación promedio: {corr_hora_ocup:.3f}")
    print(f"Correlación hora vs % de demora:       {corr_hora_demora:.3f}")

    pico_demora = resumen.loc[resumen['pct_demorados'].idxmax()]
    pico_ocupacion = resumen.loc[resumen['ocupacion_promedio'].idxmax()]
    print(
        f"\nHora con mayor demora: {int(pico_demora['hora']):02d}:00 "
        f"({pico_demora['pct_demorados']:.2f}% de vuelos demorados)"
    )
    print(
        f"Hora con mayor ocupación: {int(pico_ocupacion['hora']):02d}:00 "
        f"(ocupación promedio {pico_ocupacion['ocupacion_promedio']:.2f})"
    )

    resumen.to_csv(f'{output_dir}/analisis_horario_ocupacion_demora.csv', index=False)
    print("\n   [+] analisis_horario_ocupacion_demora.csv")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(
        resumen['hora'],
        resumen['ocupacion_promedio'],
        color='#59A14F',
        marker='o',
        linewidth=2,
        label='Ocupación promedio'
    )
    ax2.plot(
        resumen['hora'],
        resumen['pct_demorados'],
        color='#E45756',
        marker='s',
        linewidth=2,
        label='% demorados'
    )

    ax1.set_xlabel('Hora de salida')
    ax1.set_ylabel('Ocupación promedio', color='#59A14F')
    ax2.set_ylabel('% de vuelos demorados', color='#E45756')
    ax1.set_xticks(range(0, 24, 1))
    ax1.set_title('Influencia del horario en ocupación y demora', fontweight='bold')

    lineas = ax1.get_lines() + ax2.get_lines()
    etiquetas = [l.get_label() for l in lineas]
    ax1.legend(lineas, etiquetas, loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/influencia_horaria_ocupacion_demora.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   [+] influencia_horaria_ocupacion_demora.png")

    return resumen


# =============================================================================
# ESTADÍSTICOS DESCRIPTIVOS
# =============================================================================

def estadisticos_descriptivos(df, excluir=None):
    """Muestra estadísticos descriptivos de las variables numéricas."""
    print("\n" + "-" * 70)
    print("2.2 ESTADÍSTICOS DESCRIPTIVOS (Numéricas)")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)

    if numericas:
        print(f"\n{df[numericas].describe().round(2).to_string()}")


# =============================================================================
# VISUALIZACIONES
# =============================================================================

def visualizar_variable(df, columna, output_dir=OUTPUT_DIR):
    """
    Genera la visualización adecuada para una variable:
      - Numérica: histograma + boxplot (2 subplots)
      - Categórica: gráfico de barras con frecuencia y porcentaje

    Parámetros:
        df: DataFrame
        columna: nombre de la columna
        output_dir: directorio donde guardar el gráfico
    """
    if pd.api.types.is_numeric_dtype(df[columna]):
        # --- Variable numérica: Histograma + Boxplot ---
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        sns.histplot(data=df, x=columna, ax=ax[0], color="steelblue", kde=True)
        ax[0].title.set_text(f"Histograma de {columna}")

        sns.boxplot(data=df, x=columna, ax=ax[1], color="steelblue")
        ax[1].title.set_text(f"Boxplot de {columna}")

        plt.tight_layout()
        nombre = f"dist_{columna}.png"
        plt.savefig(f'{output_dir}/{nombre}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [+] {nombre}")

    else:
        # --- Variable categórica: Barras con frecuencia y porcentaje ---
        fig, ax = plt.subplots(figsize=(10, 5))

        conteo = df[columna].value_counts()
        proporciones = (conteo / len(df) * 100).round(1)
        orden = conteo.index

        bars = sns.countplot(
            data=df, x=columna, ax=ax, order=orden,
            hue=columna, palette='Set2', legend=False
        )

        # Agregar etiqueta con frecuencia y porcentaje sobre cada barra
        for i, (cat, cnt) in enumerate(conteo.items()):
            pct = proporciones[cat]
            ax.text(i, cnt + len(df) * 0.005, f'{cnt}\n({pct}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f"Distribución de {columna}", fontweight='bold')
        ax.set_ylabel("Frecuencia")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        nombre = f"dist_{columna}.png"
        plt.savefig(f'{output_dir}/{nombre}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [+] {nombre}")


def visualizar_todas_las_variables(df, excluir=None, output_dir=OUTPUT_DIR):
    """
    Genera visualizaciones individuales para cada variable del dataset.

    Parámetros:
        df: DataFrame
        excluir: columnas a excluir
        output_dir: directorio de salida
    """
    print("\n" + "-" * 70)
    print("4.1 VISUALIZACIÓN POR VARIABLE")
    print("-" * 70)

    columnas = [c for c in df.columns if c not in (excluir or [])]

    print(f"\n  Generando gráficos individuales para {len(columnas)} variables...\n")

    for col in columnas:
        visualizar_variable(df, col, output_dir)


def scatterplots_numericas(df, excluir=None, max_graficos=10, output_dir=OUTPUT_DIR):
    """
    Genera scatterplots entre pares de variables numéricas.
    Evita combinaciones repetidas (A vs B y B vs A).
    Limita la cantidad a max_graficos.

    Parámetros:
        df: DataFrame
        excluir: columnas a excluir
        max_graficos: máximo número de scatterplots a generar
        output_dir: directorio de salida
    """
    print("\n" + "-" * 70)
    print("4.2 SCATTERPLOTS ENTRE VARIABLES NUMÉRICAS")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)

    pares = list(combinations(numericas, 2))
    total_pares = len(pares)

    if total_pares > max_graficos:
        print(f"\n  [INFO] {total_pares} combinaciones posibles → limitado a {max_graficos}")
        pares = pares[:max_graficos]
    else:
        print(f"\n  [INFO] Generando {total_pares} scatterplots...")

    print()
    for col_x, col_y in pares:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=df, x=col_x, y=col_y, color="green", alpha=0.4, ax=ax)
        ax.set_title(f"Scatterplot de {col_x} vs {col_y}", fontweight='bold')
        plt.tight_layout()
        nombre = f"scatter_{col_x}_vs_{col_y}.png"
        plt.savefig(f'{output_dir}/{nombre}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [+] {nombre}")


# =============================================================================
# ANÁLISIS EXTRA: Variable objetivo + Categóricas vs Demora
# =============================================================================

def grafico_balance_clases(df, target='demora', output_dir=OUTPUT_DIR):
    """Visualiza el balance de clases de la variable objetivo."""
    print("\n" + "-" * 70)
    print("4.3 BALANCE DE CLASES (Variable Objetivo)")
    print("-" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    conteo = df[target].value_counts()
    colors = ['#4CAF50', '#F44336']
    labels = [f'No Demorado ({conteo.index[0]})', f'Demorado ({conteo.index[1]})']

    # Gráfico de barras
    axes[0].bar(labels, conteo.values, color=colors, edgecolor='white', linewidth=2)
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)',
                     ha='center', fontweight='bold')
    axes[0].set_title('Balance de Clases', fontweight='bold')
    axes[0].set_ylabel('Cantidad')

    # Gráfico de torta
    axes[1].pie(conteo.values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Proporción de Clases', fontweight='bold')

    plt.suptitle(f'Variable Objetivo: {target}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/balance_clases.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] balance_clases.png")


def grafico_categoricas_vs_target(df, target='demora', excluir=None, output_dir=OUTPUT_DIR):
    """Analiza la relación entre variables categóricas y la variable objetivo."""
    print("\n" + "-" * 70)
    print("4.4 CATEGÓRICAS vs VARIABLE OBJETIVO")
    print("-" * 70)

    _, categoricas = clasificar_variables(df, excluir=excluir)
    # Filtrar categóricas con cantidad razonable de valores únicos
    categoricas = [c for c in categoricas if df[c].nunique() <= 15]

    if not categoricas:
        print("\n  No hay variables categóricas adecuadas para este análisis.")
        return

    fig, axes = plt.subplots(len(categoricas), 1, figsize=(14, 5 * len(categoricas)))
    if len(categoricas) == 1:
        axes = [axes]

    for i, col in enumerate(categoricas):
        ax = axes[i]
        ct = pd.crosstab(df[col], df[target], normalize='index') * 100
        ct.columns = ['No Demorado %', 'Demorado %']
        ct.sort_values('Demorado %', ascending=True).plot(
            kind='barh', stacked=True, ax=ax,
            color=['#4CAF50', '#F44336'], edgecolor='white'
        )
        ax.set_title(f'Tasa de {target} por {col}', fontweight='bold')
        ax.set_xlabel('Porcentaje')
        ax.legend(loc='lower right')

    plt.suptitle(f'Relación entre Variables Categóricas y {target}',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/categoricas_vs_{target}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] categoricas_vs_{target}.png")


def analisis_velocidad_por_avion(df, output_dir=OUTPUT_DIR):
    """
    Calcula la velocidad del vuelo y genera gráficos por tipo de avión
    para detectar valores anómalos.
    """
    print("\n" + "-" * 70)
    print("4.5 ANÁLISIS DE VELOCIDAD DE VUELO POR TIPO DE AVIÓN")
    print("-" * 70)

    # 1. Crear nueva propiedad: velocidad (km/h)
    # Suponiendo que distancia está en km y tiempo en minutos
    df['velocidad_vuelo_kmh'] = (df['distancia_vuelo'] / df['tiempo_estimado_vuelo']) * 60

    # 2. Gráfico combinado (Boxplot) para ver outliers rápidamente
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df, x='velocidad_vuelo_kmh', y='tipo_avion', ax=ax, palette='Set2')
    ax.set_title('Distribución de Velocidad de Vuelo por Tipo de Avión', fontweight='bold')
    ax.set_xlabel('Velocidad (km/h)')
    ax.set_ylabel('Tipo de Avión')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/velocidad_vs_tipo_avion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [+] velocidad_vs_tipo_avion.png")

    # 3. Gráficos individuales por cada tipo de avión
    tipos_avion = df['tipo_avion'].dropna().unique()
    for avion in tipos_avion:
        data_avion = df[df['tipo_avion'] == avion]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        
        sns.histplot(data=data_avion, x='velocidad_vuelo_kmh', ax=ax[0], color="coral", kde=True)
        ax[0].title.set_text(f"Histograma de Velocidad - {avion}")
        
        sns.boxplot(data=data_avion, x='velocidad_vuelo_kmh', ax=ax[1], color="coral")
        ax[1].title.set_text(f"Boxplot de Velocidad - {avion}")
        
        plt.tight_layout()
        # Reemplazar espacios y caracteres problemáticos
        nombre_archivo = f"velocidad_{avion.replace(' ', '_').replace('/', '_')}.png"
        plt.savefig(f'{output_dir}/{nombre_archivo}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [+] {nombre_archivo}")


# =============================================================================
# ANÁLISIS MULTIVARIANTE: Matriz de correlación
# =============================================================================

def matriz_correlacion(df, excluir=None, output_dir=OUTPUT_DIR):
    """Genera la matriz de correlación entre variables numéricas."""
    print("\n" + "-" * 70)
    print("4.6 MATRIZ DE CORRELACIÓN")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)
    # Incluir target si es numérico y no fue excluido
    if 'demora' not in (excluir or []) and 'demora' in df.columns:
        if 'demora' not in numericas:
            numericas.append('demora')

    corr = df[numericas].corr()

    # Top correlaciones con demora
    if 'demora' in corr.columns:
        print("\n  --- Correlaciones con 'demora' ---")
        corr_target = corr['demora'].drop('demora').sort_values(ascending=False)
        for var, val in corr_target.items():
            print(f"    {var:<30} {val:>8.4f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, square=True, linewidths=0.5, ax=ax,
        cbar_kws={'shrink': 0.8}
    )
    ax.set_title('Matriz de Correlación', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/matriz_correlacion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] matriz_correlacion.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # FASE 2: Entendimiento de los datos
    # =========================================================================
    df = cargar_datos()
    descripcion_general(df)

    # Columnas que no son features reales
    EXCLUIR = ['id_vuelo']

    estadisticos_descriptivos(df, excluir=EXCLUIR)

    # =========================================================================
    # FASE 3: Limpieza de datos
    # =========================================================================
    print("\n" + "=" * 70)
    print("FASE 3: LIMPIEZA Y TRANSFORMACIÓN DE DATOS")
    print("=" * 70)

    detectar_nulos(df)
    detectar_duplicados(df)
    detectar_outliers(df, excluir=EXCLUIR + ['demora'])

    # =========================================================================
    # Análisis de proporciones (variables categóricas)
    # =========================================================================
    analisis_proporciones(df, excluir=EXCLUIR)
    analisis_demora_por_aeropuerto(df, target='demora')
    grafico_vuelos_y_demora_por_aeropuerto(df, target='demora', top_n=20)
    grafico_ocupacion_por_aeropuerto(df, col_ocupacion='ocupacion_vuelo', top_n=20)
    analisis_horario_ocupacion_demora(
        df,
        col_hora='hora_salida_programada',
        col_ocupacion='ocupacion_vuelo',
        target='demora'
    )

    # =========================================================================
    # FASE 4: Visualizaciones
    # =========================================================================
    print("\n" + "=" * 70)
    print("FASE 4: VISUALIZACIONES")
    print("=" * 70)

    visualizar_todas_las_variables(df, excluir=EXCLUIR)
    scatterplots_numericas(df, excluir=EXCLUIR + ['demora'], max_graficos=10)

    # Extras
    grafico_balance_clases(df)
    grafico_categoricas_vs_target(df, target='demora', excluir=EXCLUIR + ['demora'])
    analisis_velocidad_por_avion(df)
    matriz_correlacion(df, excluir=EXCLUIR)

    # =========================================================================
    # Resumen final
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETO")
    print("=" * 70)
    print(f"\nGráficos generados en: {OUTPUT_DIR}/")
    print("Revisá la carpeta para ver todos los archivos generados.")
