"""
Limpieza de datos: detección de nulos, duplicados y outliers.
"""

import numpy as np
import pandas as pd
from scipy import stats

from utils import clasificar_variables


# ── Descripción general ──────────────────────────────────────────────────────

def descripcion_general(df):
    """Muestra resumen estructural del dataset: tipos, nulos, únicos."""
    print("\n" + "=" * 70)
    print("DESCRIPCIÓN GENERAL DEL DATASET")
    print("=" * 70)
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


# ── Estadísticos descriptivos ────────────────────────────────────────────────

def estadisticos_descriptivos(df, excluir=None):
    """Muestra estadísticos descriptivos de las variables numéricas."""
    print("\n" + "-" * 70)
    print("ESTADÍSTICOS DESCRIPTIVOS (Numéricas)")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)
    if numericas:
        print(f"\n{df[numericas].describe().round(2).to_string()}")


# ── Detección de nulos ───────────────────────────────────────────────────────

def detectar_nulos(df):
    """Detecta y reporta valores ausentes por columna."""
    print("\n" + "-" * 70)
    print("DETECCIÓN DE VALORES AUSENTES (NAs)")
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


# ── Detección de duplicados ──────────────────────────────────────────────────

def detectar_duplicados(df):
    """Detecta filas duplicadas en el dataset."""
    print("\n" + "-" * 70)
    print("DETECCIÓN DE DUPLICADOS")
    print("-" * 70)

    duplicados = df.duplicated().sum()
    if duplicados == 0:
        print(f"\n  ✅ No se encontraron filas duplicadas (de {len(df)} registros).")
    else:
        print(f"\n  ⚠️  Se encontraron {duplicados} filas duplicadas "
              f"({duplicados/len(df)*100:.2f}%).")
        print("  Primeros ejemplos:")
        print(df[df.duplicated(keep=False)].head(6).to_string())

    return duplicados


# ── Detección de outliers ────────────────────────────────────────────────────

def detectar_outliers(df, excluir=None):
    """
    Detecta outliers con IQR y Z-Score para todas las variables numéricas.

    Retorna:
        dict con información de outliers por columna
    """
    print("\n" + "-" * 70)
    print("DETECCIÓN DE OUTLIERS")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)

    # --- IQR ---
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
            'count': outliers, 'pct': pct,
            'lim_inf': lim_inf, 'lim_sup': lim_sup,
            'Q1': Q1, 'Q3': Q3, 'IQR': IQR
        }
        print(
            f"  {col:<25} {Q1:>8.2f} {Q3:>8.2f} {IQR:>8.2f} "
            f"{lim_inf:>10.2f} {lim_sup:>10.2f} {outliers:>10} {pct:>5.1f}%"
        )

    # --- Z-Score ---
    print("\n--- Método Z-Score (|z| > 3) ---")
    print(f"  {'Variable':<25} {'Outliers Z-Score':>18} {'%':>6}")
    print("  " + "-" * 51)
    for col in numericas:
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers_z = (z_scores > 3).sum()
        pct_z = outliers_z / len(df) * 100
        print(f"  {col:<25} {outliers_z:>18} {pct_z:>5.1f}%")

    total_con_outliers = sum(1 for v in outlier_info.values() if v['count'] > 0)
    print(f"\n  [INFO] {total_con_outliers} de {len(numericas)} variables numéricas "
          f"presentan outliers (método IQR).")

    return outlier_info
