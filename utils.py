"""
Utilidades compartidas: carga de datos, clasificación de variables y helpers.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuración global de gráficos ─────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
sns.set_theme(style="whitegrid", palette="muted")

OUTPUT_DIR = "TPI/graficos"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Carga de datos ───────────────────────────────────────────────────────────

def cargar_datos(ruta='Vuelos.xlsx'):
    """Carga el dataset de vuelos desde el archivo Excel."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rutas_candidatas = [
        ruta,
        os.path.join(script_dir, 'Vuelos.xlsx'),
        os.path.join(script_dir, 'TPI', 'Vuelos.xlsx'),
    ]

    ruta_excel = next((r for r in rutas_candidatas if os.path.exists(r)), None)
    if ruta_excel is None:
        raise FileNotFoundError(
            "No se encontró el archivo Excel. Rutas intentadas: "
            + ", ".join(rutas_candidatas)
        )

    df = pd.read_excel(ruta_excel)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


# ── Clasificación automática de variables ────────────────────────────────────

def clasificar_variables(df, excluir=None):
    """
    Clasifica columnas en numéricas y categóricas.

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


# ── Helpers de aeropuerto ────────────────────────────────────────────────────

def detectar_columnas_aeropuerto(df):
    """Detecta columnas que representan aeropuerto (origen/destino o única)."""
    candidatas = []
    nombres_directos = {
        'aeropuerto', 'origen', 'destino',
        'aeropuerto_origen', 'aeropuerto_destino',
        'origen_iata', 'destino_iata',
    }

    for col in df.columns:
        nombre = col.lower().strip()
        if not pd.api.types.is_object_dtype(df[col]) and not pd.api.types.is_string_dtype(df[col]):
            continue
        if 'aeropuerto' in nombre or nombre in nombres_directos:
            candidatas.append(col)

    return candidatas


def serie_demorada(serie):
    """Convierte una columna de demora a indicador binario (1=demorado)."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie.eq(1).astype(int)

    texto = serie.astype(str).str.lower().str.strip()
    return texto.isin({'1', 'si', 'sí', 'true', 'demorado', 'delay', 'delayed'}).astype(int)


def expandir_aeropuertos(df, columnas_aeropuerto, columnas_extra=None):
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
