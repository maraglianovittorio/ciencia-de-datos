"""
================================================================================
Preprocesamiento para KNN — Vista Minable
================================================================================
Lee Vuelos.xlsx, aplica limpieza, normalización Min-Max y One-Hot Encoding.
Exporta la vista minable lista para modelado en /vistas_minables.

Variable objetivo: demora (0 = No demorado, 1 = Demorado)
================================================================================
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import cargar_datos, clasificar_variables

VISTAS_MINABLES_DIR = 'vistas_minables'
os.makedirs(VISTAS_MINABLES_DIR, exist_ok=True)

def crear_franja_horaria(df, col_hora='hora_salida_programada'):
    """
    Crea variable de franja horaria basada en hora de salida:
    - Mañana: 7:00 - 11:00
    - Mediodía: 11:01 - 15:00
    - Tarde: 15:01 - 19:00
    - Noche: 19:01 - 1:00
    - Madrugada: 1:01 - 7:00
    """
    base = df[[col_hora]].copy()
    hora = (
        base[col_hora].astype(str)
        .str.extract(r'(\d{1,2})', expand=False)
        .astype(float)
    )

    def clasificar(h):
        if pd.isna(h):
            return None
        h = int(h)
        if 7 <= h <= 11:
            return 'manana'
        elif 12 <= h <= 15:
            return 'mediodia'
        elif 16 <= h <= 19:
            return 'tarde'
        elif h >= 20 or h == 0:
            return 'noche'
        elif 1 <= h <= 6:
            return 'madrugada'
        else:
            return 'noche'

    df['franja_horaria'] = hora.apply(clasificar)
    return df


def crear_timestamp(df, col_hora='hora_salida_programada'):
    """
    Convierte la hora de salida en un valor numérico continuo.
    Representa la hora del día como un número (ej. 14:30 → 14.5).
    """
    hora = (
        df[col_hora].astype(str)
        .str.extract(r'(\d{1,2})(?::(\d{2}))?', expand=False)
        .iloc[:, 0]
        .astype(float)
    )
    minutos = (
        df[col_hora].astype(str)
        .str.extract(r'\d{1,2}:(\d{2})', expand=False)
        .astype(float)
        .fillna(0) / 60
    )
    df['timestamp_vuelo'] = hora + minutos
    return df


EXCLUIR = ['id_vuelo', 'hora_salida_programada', 'dia_semana', 'puerta_embarque']
TARGET = 'demora'


def filtrado_custom(df):
    """
    Filtros personalizados aplicados antes del preprocesamiento general.
    - Elimina vuelos cuya velocidad calculada (distancia/tiempo) sea < 100 km/h.
    - Elimina vuelos entre EZE y AEP (y viceversa): no son vuelos comerciales.
    """
    print("\n" + "=" * 70)
    print("FILTRADO CUSTOM")
    print("=" * 70)

    info = {'velocidad_baja': 0, 'eze_aep': 0, 'shape_inicial': df.shape}
    mask_velocidad = (df['distancia_vuelo'] / (df['tiempo_estimado_vuelo'] / 60)) >= 100
    info['velocidad_baja'] = (~mask_velocidad).sum()

    mask_eze_aep = ~(
        ((df['aeropuerto_origen'] == 'EZE') & (df['aeropuerto_destino'] == 'AEP')) |
        ((df['aeropuerto_origen'] == 'AEP') & (df['aeropuerto_destino'] == 'EZE'))
    )
    info['eze_aep'] = (~mask_eze_aep).sum()

    df = df[mask_velocidad & mask_eze_aep].reset_index(drop=True)

    df = crear_franja_horaria(df)
    df = crear_timestamp(df)

    info['shape_final'] = df.shape
    total_eliminados = info['shape_inicial'][0] - info['shape_final'][0]
    print(f"\n  [–] Vuelos con velocidad < 100 km/h eliminados: {info['velocidad_baja']}")
    print(f"  [–] Vuelos EZE↔AEP eliminados: {info['eze_aep']}")
    print(f"  [–] Total filas eliminadas: {total_eliminados}")
    print(f"\n  Dataset filtrado: {df.shape[0]} filas × {df.shape[1]} columnas")

    return df, info


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

    info = {'eliminadas': [], 'duplicados': 0, 'nulos': {}, 'shape_inicial': df.shape}

    cols_a_eliminar = [c for c in EXCLUIR if c in df.columns]
    if cols_a_eliminar:
        df = df.drop(columns=cols_a_eliminar)
        info['eliminadas'] = cols_a_eliminar
        print(f"\n  [–] Columnas eliminadas: {cols_a_eliminar}")

    duplicados = df.duplicated().sum()
    if duplicados > 0:
        df = df.drop_duplicates()
        info['duplicados'] = duplicados
        print(f"  [–] Filas duplicadas eliminadas: {duplicados}")
    else:
        print(f"  [✓] Sin filas duplicadas")

    nulos_total = df.isnull().sum().sum()
    if nulos_total > 0:
        print(f"  [!] Nulos encontrados: {nulos_total}")

        numericas, categoricas = clasificar_variables(df, excluir=[TARGET])

        for col in numericas:
            n_nulos = df[col].isnull().sum()
            if n_nulos > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                info['nulos'][col] = {'n': n_nulos, 'valor': mediana, 'tipo': 'mediana'}
                print(f"      {col}: {n_nulos} nulos → mediana ({mediana:.2f})")

        for col in categoricas:
            n_nulos = df[col].isnull().sum()
            if n_nulos > 0:
                moda = df[col].mode()[0]
                df[col] = df[col].fillna(moda)
                info['nulos'][col] = {'n': n_nulos, 'valor': moda, 'tipo': 'moda'}
                print(f"      {col}: {n_nulos} nulos → moda ('{moda}')")
    else:
        print(f"  [✓] Sin valores nulos")

    info['shape_final'] = df.shape
    print(f"\n  Dataset limpio: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df, info


def normalizar_numericas(df):
    """
    Aplica normalización Min-Max (0-1) a todas las variables numéricas
    EXCEPTO la variable objetivo.
    """
    print("\n" + "=" * 70)
    print("NORMALIZACIÓN MIN-MAX")
    print("=" * 70)

    numericas, _ = clasificar_variables(df, excluir=[TARGET])
    info = {'variables': numericas, 'shape_antes': df.shape}

    if not numericas:
        print("  [!] No se encontraron variables numéricas para normalizar.")
        return df, info

    scaler = MinMaxScaler()
    df[numericas] = scaler.fit_transform(df[numericas])

    print(f"\n  Variables normalizadas ({len(numericas)}):")
    for col in numericas:
        print(f"    • {col:30s}  min={df[col].min():.2f}  max={df[col].max():.2f}")

    info['shape_despues'] = df.shape
    return df, info


def aplicar_one_hot_encoding(df):
    """
    Aplica One-Hot Encoding a todas las variables categóricas.
    """
    print("\n" + "=" * 70)
    print("ONE-HOT ENCODING")
    print("=" * 70)

    _, categoricas = clasificar_variables(df, excluir=[TARGET])
    info = {'variables': categoricas, 'dummies': {}}

    if not categoricas:
        print("  [!] No se encontraron variables categóricas.")
        return df, info

    print(f"\n  Variables a codificar ({len(categoricas)}):")
    for col in categoricas:
        valores = df[col].unique()
        print(f"    • {col}: {len(valores)} categorías → {valores[:6].tolist()}"
              f"{'...' if len(valores) > 6 else ''}")

    df = pd.get_dummies(df, columns=categoricas, drop_first=True, dtype=int)

    print(f"\n  Columnas después de OHE: {df.shape[1]}")
    print(f"  Nuevas columnas dummy creadas:")
    for col in categoricas:
        dummies = [c for c in df.columns if c.startswith(f"{col}_")]
        info['dummies'][col] = dummies
        print(f"    • {col} → {dummies[:5]}{'...' if len(dummies) > 5 else ''}")

    return df, info


def exportar_vista_minable(df, nro_vista, info_custom, info_limpieza, info_norm, info_ohe):
    """Exporta el DataFrame procesado como la vista minable final."""
    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE VISTA MINABLE")
    print("=" * 70)

    if TARGET in df.columns:
        cols = [c for c in df.columns if c != TARGET] + [TARGET]
        df = df[cols]

    output_file = os.path.join(VISTAS_MINABLES_DIR, f'vista_{nro_vista}.csv')
    df.to_csv(output_file, index=False)
    print(f"\n  [+] Vista minable exportada: {output_file}")
    print(f"      Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"      Variable objetivo: '{TARGET}' (última columna)")

    print(f"\n  Composición de la vista minable:")
    n_numericas = df.select_dtypes(include=[np.number]).shape[1]
    print(f"    • Variables numéricas (normalizadas + dummies): {n_numericas}")
    print(f"    • Variable objetivo: {TARGET} "
          f"(0={df[TARGET].eq(0).sum()}, 1={df[TARGET].eq(1).sum()})")

    exportar_markdown(df, nro_vista, info_custom, info_limpieza, info_norm, info_ohe)

    return df


def exportar_markdown(df, nro_vista, info_custom, info_limpieza, info_norm, info_ohe):
    """Genera el archivo .md con toda la información del preprocesamiento."""

    nros = [c for c in df.columns if c != TARGET]
    target = df[TARGET]

    lineas = [
        f"# Vista Minable — {nro_vista}",
        "",
        "## Descripción",
        "",
        "[ _Descripción general del dataset y propósito_ ]",
        "",
        "---",
        "",
        "## Propiedades (Features)",
        "",
        "### Variables Numéricas (normalizadas Min-Max)",
        "",
    ]

    if info_norm['variables']:
        lineas.append("| Propiedad | Descripción |")
        lineas.append("|-----------|-------------|")
        for col in info_norm['variables']:
            lineas.append(f"| {col} | |")
    else:
        lineas.append("_Ninguna_")

    lineas.extend([
        "",
        "### Variables Categóricas (One-Hot Encoding)",
        "",
    ])

    for col, dummies in info_ohe['dummies'].items():
        lineas.append(f"#### {col}")
        for d in dummies:
            lineas.append(f"- {d}")
        lineas.append("")

    lineas.extend([
        "### Variable Objetivo",
        "",
        f"- **{TARGET}** (0 = No demorado, 1 = Demorado)",
        "",
        "---",
        "",
        "## Transformaciones Aplicadas",
        "",
    ])

    if info_custom['velocidad_baja'] > 0 or info_custom['eze_aep'] > 0:
        lineas.append("1. **Filtrado Custom**:")
        if info_custom['velocidad_baja'] > 0:
            lineas.append(f"   - Vuelos con velocidad < 100 km/h eliminados: {info_custom['velocidad_baja']}")
        if info_custom['eze_aep'] > 0:
            lineas.append(f"   - Vuelos EZE↔AEP eliminados: {info_custom['eze_aep']}")

    limp = info_limpieza
    if limp['eliminadas']:
        lineas.append(f"2. **Limpieza**: Columnas eliminadas: `{', '.join(limp['eliminadas'])}`")
    if limp['duplicados'] > 0:
        lineas.append(f"2. **Limpieza**: {limp['duplicados']} filas duplicadas eliminadas")
    if limp['nulos']:
        for col, datos in limp['nulos'].items():
            lineas.append(f"2. **Limpieza**: `{col}`: {datos['n']} nulos → {datos['tipo']} ({datos['valor']})")
    if not limp['eliminadas'] and limp['duplicados'] == 0 and not limp['nulos']:
        lineas.append("2. **Limpieza**: Sin duplicados ni nulos")

    lineas.extend([
        "3. **Normalización Min-Max**: Escala (0-1) para todas las variables numéricas",
        "4. **One-Hot Encoding**: Conversión de variables categóricas a dummies con `drop_first=True`",
        "",
        "---",
        "",
        "## Resumen",
        "",
    ])

    lineas.extend([
        f"- Filas: {df.shape[0]}",
        f"- Columnas (features): {len(nros)}",
        f"- Variable objetivo: `{TARGET}` (0={target.eq(0).sum()}, 1={target.eq(1).sum()})",
        "",
        "---",
        "",
        "## Notas y Anotaciones",
        "",
        "[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]",
    ])

    ruta_md = os.path.join(VISTAS_MINABLES_DIR, f'vista_{nro_vista}.md')
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lineas))

    print(f"  [+] vista_{nro_vista}.md")


def main():
    print("=" * 70)
    print("PREPROCESAMIENTO PARA KNN")
    print("=" * 70)

    vistas_existentes = sorted([
        f for f in os.listdir(VISTAS_MINABLES_DIR)
        if f.startswith('vista_') and f.endswith('.csv')
    ])

    if vistas_existentes:
        ult_vista = vistas_existentes[-1]
        ultimo_nro = int(ult_vista.replace('vista_', '').replace('.csv', ''))
        nuevo_nro = f"{ultimo_nro + 1:03d}"
    else:
        nuevo_nro = '001'

    print(f"\n  Número de vista a generar: {nuevo_nro}")
    print(f"  Salida: vistas_minables/vista_{nuevo_nro}.csv")

    df = cargar_datos()
    df, info_custom = filtrado_custom(df)
    df, info_limpieza = limpiar_datos(df)
    df, info_norm = normalizar_numericas(df)
    df, info_ohe = aplicar_one_hot_encoding(df)
    df = exportar_vista_minable(df, nuevo_nro, info_custom, info_limpieza, info_norm, info_ohe)

    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO COMPLETO")
    print("=" * 70)
    print(f"\n  Archivos listos en vistas_minables/")
    print(f"    • vista_{nuevo_nro}.csv")
    print(f"    • vista_{nuevo_nro}.md")


if __name__ == "__main__":
    main()
