"""
=============================================================================
Preprocesamiento para KNN — Vista Minable Final
=============================================================================
Lee Vuelos.xlsx, aplica limpieza, normalización Min-Max y One-Hot Encoding.
Exporta la vista minable lista para modelado como 'vista_minable_knn.xlsx'.

Variable objetivo: demora (0 = No demorado, 1 = Demorado)
=============================================================================
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import cargar_datos, clasificar_variables

# ── Configuración ────────────────────────────────────────────────────────────

# Columnas a excluir del modelo (identificadores, no aportan información)
EXCLUIR = ['id_vuelo']

# Variable objetivo
TARGET = 'demora'

# Archivo de salida
OUTPUT_FILE = 'vista_minable_knn.csv'


# ── Funciones ────────────────────────────────────────────────────────────────

def limpiar_datos(df):
    """
    Limpieza básica del dataset:
    - Elimina columnas excluidas (identificadores)
    - Elimina filas duplicadas
    - Imputa nulos: mediana para numéricas, moda para categóricas
    """
    print("\n" + "=" * 70)
    print("LIMPIEZA DE DATOS")
    print("=" * 70)

    # Eliminar columnas que no son features
    cols_a_eliminar = [c for c in EXCLUIR if c in df.columns]
    if cols_a_eliminar:
        df = df.drop(columns=cols_a_eliminar)
        print(f"\n  [–] Columnas eliminadas: {cols_a_eliminar}")

    # Duplicados
    duplicados = df.duplicated().sum()
    if duplicados > 0:
        df = df.drop_duplicates()
        print(f"  [–] Filas duplicadas eliminadas: {duplicados}")
    else:
        print(f"  [✓] Sin filas duplicadas")

    # Nulos
    nulos_total = df.isnull().sum().sum()
    if nulos_total > 0:
        print(f"  [!] Nulos encontrados: {nulos_total}")

        numericas, categoricas = clasificar_variables(df, excluir=[TARGET])

        # Numéricas → mediana
        for col in numericas:
            n_nulos = df[col].isnull().sum()
            if n_nulos > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                print(f"      {col}: {n_nulos} nulos → mediana ({mediana:.2f})")

        # Categóricas → moda
        for col in categoricas:
            n_nulos = df[col].isnull().sum()
            if n_nulos > 0:
                moda = df[col].mode()[0]
                df[col] = df[col].fillna(moda)
                print(f"      {col}: {n_nulos} nulos → moda ('{moda}')")
    else:
        print(f"  [✓] Sin valores nulos")

    print(f"\n  Dataset limpio: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def normalizar_numericas(df):
    """
    Aplica normalización Min-Max (0-1) a todas las variables numéricas
    EXCEPTO la variable objetivo.

    Esto es crítico para KNN porque el algoritmo se basa en distancias
    y las variables con mayor escala dominarían el cálculo.
    """
    print("\n" + "=" * 70)
    print("NORMALIZACIÓN MIN-MAX")
    print("=" * 70)

    numericas, _ = clasificar_variables(df, excluir=[TARGET])

    if not numericas:
        print("  [!] No se encontraron variables numéricas para normalizar.")
        return df

    scaler = MinMaxScaler()
    df[numericas] = scaler.fit_transform(df[numericas])

    print(f"\n  Variables normalizadas ({len(numericas)}):")
    for col in numericas:
        print(f"    • {col:30s}  min={df[col].min():.2f}  max={df[col].max():.2f}")

    return df


def aplicar_one_hot_encoding(df):
    """
    Aplica One-Hot Encoding a todas las variables categóricas.

    A diferencia de Label Encoding, OHE no impone un orden artificial
    entre categorías. Esto es fundamental para KNN, que calcula distancias:
    con Label Encoding, 'Tormenta'=4 estaría artificialmente "más lejos"
    de 'Despejado'=0 que de 'Nublado'=3.
    """
    print("\n" + "=" * 70)
    print("ONE-HOT ENCODING")
    print("=" * 70)

    _, categoricas = clasificar_variables(df, excluir=[TARGET])

    if not categoricas:
        print("  [!] No se encontraron variables categóricas.")
        return df

    print(f"\n  Variables a codificar ({len(categoricas)}):")
    for col in categoricas:
        valores = df[col].unique()
        print(f"    • {col}: {len(valores)} categorías → {valores[:6].tolist()}"
              f"{'...' if len(valores) > 6 else ''}")

    # drop_first=True para evitar multicolinealidad (dummy variable trap)
    df = pd.get_dummies(df, columns=categoricas, drop_first=True, dtype=int)

    print(f"\n  Columnas después de OHE: {df.shape[1]}")
    print(f"  Nuevas columnas dummy creadas:")
    for col in categoricas:
        dummies = [c for c in df.columns if c.startswith(f"{col}_")]
        print(f"    • {col} → {dummies[:5]}{'...' if len(dummies) > 5 else ''}")

    return df


def exportar_vista_minable(df, output_file=OUTPUT_FILE):
    """Exporta el DataFrame procesado como la vista minable final."""
    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE VISTA MINABLE")
    print("=" * 70)

    # Asegurar que la variable objetivo esté al final
    if TARGET in df.columns:
        cols = [c for c in df.columns if c != TARGET] + [TARGET]
        df = df[cols]

    df.to_csv(output_file, index=False)
    print(f"\n  [+] Vista minable exportada: {output_file}")
    print(f"      Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"      Variable objetivo: '{TARGET}' (última columna)")

    # Resumen final
    print(f"\n  Composición de la vista minable:")
    n_numericas = df.select_dtypes(include=[np.number]).shape[1]
    print(f"    • Variables numéricas (normalizadas + dummies): {n_numericas}")
    print(f"    • Variable objetivo: {TARGET} "
          f"(0={df[TARGET].eq(0).sum()}, 1={df[TARGET].eq(1).sum()})")

    return df


# ── Pipeline principal ───────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("PREPROCESAMIENTO PARA KNN")
    print("=" * 70)

    # 1. Cargar datos originales
    df = cargar_datos()

    # 2. Limpieza
    df = limpiar_datos(df)

    # 3. Normalización Min-Max de numéricas
    df = normalizar_numericas(df)

    # 4. One-Hot Encoding de categóricas
    df = aplicar_one_hot_encoding(df)

    # 5. Exportar vista minable
    df = exportar_vista_minable(df)

    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\n  Archivo listo para KNN: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
