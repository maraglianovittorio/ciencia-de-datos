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

from utils import cargar_datos
from limpieza import (
    descripcion_general,
    estadisticos_descriptivos,
    detectar_nulos,
    detectar_duplicados,
    detectar_outliers,
)
from analisis import (
    visibilidad_vs_demora,
    distancia_vs_tiempo,
    clima_vs_demora,
    congestion_vs_demora,
    hora_vs_demora,
    demora_por_aeropuerto,
    velocidad_por_tipo_avion,
    balance_clases,
    matriz_correlacion,
)

# Columnas que no son features reales
EXCLUIR = ['id_vuelo']


def main():
    # ── Fase 2: Entendimiento de los datos ───────────────────────────────
    print("=" * 70)
    print("FASE 2: ENTENDIMIENTO DE LOS DATOS")
    print("=" * 70)

    df = cargar_datos()
    descripcion_general(df)
    estadisticos_descriptivos(df, excluir=EXCLUIR)

    # ── Fase 3: Limpieza de datos ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FASE 3: LIMPIEZA Y PREPARACIÓN")
    print("=" * 70)

    detectar_nulos(df)
    detectar_duplicados(df)
    detectar_outliers(df, excluir=EXCLUIR + ['demora'])

    # ── Fase 4: Análisis de relaciones útiles ────────────────────────────
    print("\n" + "=" * 70)
    print("FASE 4: ANÁLISIS DE RELACIONES ENTRE VARIABLES")
    print("=" * 70)

    # Variables que impactan la demora
    visibilidad_vs_demora(df)
    clima_vs_demora(df)
    congestion_vs_demora(df)
    hora_vs_demora(df)
    demora_por_aeropuerto(df)

    # Relaciones entre features
    distancia_vs_tiempo(df)
    velocidad_por_tipo_avion(df)

    # Visión global
    balance_clases(df)
    matriz_correlacion(df, excluir=EXCLUIR)

    # ── Resumen ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETO")
    print("=" * 70)
    print("\nGráficos generados en: TPI/graficos/")


if __name__ == "__main__":
    main()
