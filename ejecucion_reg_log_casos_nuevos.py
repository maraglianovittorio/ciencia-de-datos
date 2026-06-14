"""
================================================================================
Orquestador: Regresion Logistica sobre Casos Nuevos
================================================================================
Pipeline completo:
  1) Carga los casos crudos (sin columna 'demora').
  2) Aplica el mismo preprocesamiento que la vista 011 (filtros custom,
     limpieza, normalizacion Min-Max con los rangos del training, OHE
     alineado al schema de la vista 011).
  3) Llama a regresion_logistica.py para obtener probabilidad y clase
     predicha por vuelo.
  4) Exporta un CSV con columnas: id_vuelo, demora.
================================================================================
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import preprocesamiento as pp
import regresion_logistica as rl


CASOS_NUEVOS_CSV = "casosNuevos.xlsx - VuelosNuevos.csv"
SALIDA_CSV = "predicciones_casos_nuevos.csv"
VISTA_REFERENCIA = "012"
DIR_REFERENCIA = os.path.join("models", "logistica", VISTA_REFERENCIA)

ARCHIVO_SCALER = os.path.join(DIR_REFERENCIA, "scaler_ranges.json")
ARCHIVO_COLUMNAS = os.path.join(DIR_REFERENCIA, "columnas_esperadas.json")


def cargar_scaler(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    scaler = MinMaxScaler()
    scaler.min_ = np.array(payload["min_"], dtype=float)
    scaler.scale_ = np.array(payload["scale_"], dtype=float)
    scaler.data_min_ = np.array(payload["data_min_"], dtype=float)
    scaler.data_max_ = np.array(payload["data_max_"], dtype=float)
    scaler.n_features_in_ = len(payload["min_"])
    return scaler


def cargar_columnas_esperadas(ruta_json):
    with open(ruta_json, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocesar_casos_nuevos(df_crudo, scaler, columnas_esperadas):
    """
    Replica el preprocesamiento de la vista 012 sobre los casos crudos,
    reusando el scaler y el schema de columnas del training.

    NOTA: en inferencia NO se aplica `filtrado_custom` (los registros
    atipicos tambien se predicen; el filtro solo se usa en training).
    Pero SÍ se reconstruyen `franja_horaria` y `timestamp_vuelo` a partir
    de `hora_salida_programada`, ya que son features necesarias para el modelo.
    """
    print("\n[1/4] Construyendo franja_horaria y timestamp_vuelo...")
    df = pp.crear_franja_horaria(df_crudo)
    df = pp.crear_timestamp(df)

    print("\n[2/4] Limpieza...")
    df, _ = pp.limpiar_datos(df)

    print("\n[3/4] Normalizacion Min-Max (con rangos del training)...")
    df, _ = pp.normalizar_numericas(df, scaler=scaler)

    print("\n[4/4] One-Hot Encoding (alineado a vista 012)...")
    df, _ = pp.aplicar_one_hot_encoding(df, columnas_esperadas=columnas_esperadas)

    return df


def predecir_y_exportar(df_procesado, ids_originales, ruta_salida):
    filas = df_procesado.to_dict(orient="records")
    resultados = rl.predecir_lote(filas)
    probs = [float(p) for p, _ in resultados]
    clases = [int(c) for _, c in resultados]

    out = pd.DataFrame({
        "id_vuelo": ids_originales,
        "probabilidad": probs,
        "demora": clases,
    })
    out.to_csv(ruta_salida, index=False)

    total = len(out)
    n_demorados = sum(clases)
    n_puntuales = total - n_demorados
    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    print(f"  Archivo exportado: {ruta_salida}")
    print(f"  Filas exportadas : {total}")
    print(f"  Demora = 0       : {n_puntuales} ({n_puntuales/total*100:.1f}%)")
    print(f"  Demora = 1       : {n_demorados} ({n_demorados/total*100:.1f}%)")
    print(f"  Probabilidad - min: {min(probs):.4f}  max: {max(probs):.4f}  media: {sum(probs)/total:.4f}")
    print("=" * 70)
    return out


def main():
    print("=" * 70)
    print("EJECUCION REGRESION LOGISTICA - CASOS NUEVOS")
    print("=" * 70)

    if not os.path.exists(ARCHIVO_SCALER):
        raise FileNotFoundError(
            f"No se encontro {ARCHIVO_SCALER}. "
            f"Ejecuta primero 'python preprocesamiento.py' para generar los artefactos."
        )
    if not os.path.exists(ARCHIVO_COLUMNAS):
        raise FileNotFoundError(
            f"No se encontro {ARCHIVO_COLUMNAS}. "
            f"Ejecuta primero 'python preprocesamiento.py' para generar los artefactos."
        )

    if not os.path.exists(CASOS_NUEVOS_CSV):
        raise FileNotFoundError(f"No se encontro el archivo de casos nuevos: {CASOS_NUEVOS_CSV}")

    print(f"\n  Cargando casos nuevos: {CASOS_NUEVOS_CSV}")
    df_crudo = pd.read_csv(CASOS_NUEVOS_CSV)
    print(f"  Filas crudas: {df_crudo.shape[0]}")

    ids = df_crudo["id_vuelo"].copy()

    print(f"  Cargando scaler desde: {ARCHIVO_SCALER}")
    scaler = cargar_scaler(ARCHIVO_SCALER)
    print(f"  Cargando columnas desde: {ARCHIVO_COLUMNAS}")
    columnas_esperadas = cargar_columnas_esperadas(ARCHIVO_COLUMNAS)
    print(f"  Schema esperado: {len(columnas_esperadas)} columnas")

    df_procesado = preprocesar_casos_nuevos(df_crudo, scaler, columnas_esperadas)

    print(f"\n  Verificacion de schema:")
    print(f"    Columnas en df_procesado: {df_procesado.shape[1]}")
    print(f"    Columnas esperadas      : {len(columnas_esperadas)}")
    assert list(df_procesado.columns) == columnas_esperadas, (
        "El schema no coincide con la vista 011. Revisa el one-hot encoding."
    )

    predecir_y_exportar(df_procesado, ids, SALIDA_CSV)


if __name__ == "__main__":
    main()
