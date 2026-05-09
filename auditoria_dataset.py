"""
=============================================================================
Auditoría Completa del Dataset de Vuelos
=============================================================================
Barrido secuencial columna por columna. Detecta:
  - Nulos y duplicados
  - Outliers (IQR) por variable numérica
  - Distribuciones sesgadas
  - Cardinalidad y proporciones de categóricas
  - Relaciones cruzadas sospechosas
  - Anomalías de dominio (rutas imposibles, velocidades, coherencia)
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from utils import cargar_datos, clasificar_variables, OUTPUT_DIR

warnings.filterwarnings('ignore', category=FutureWarning)

AUDIT_DIR = os.path.join(OUTPUT_DIR, 'auditoria')
os.makedirs(AUDIT_DIR, exist_ok=True)

HALLAZGOS = []  # Acumula hallazgos para el reporte final


def registrar(seccion, nivel, texto):
    """Registra un hallazgo. nivel: INFO, HALLAZGO, ANOMALÍA, CRÍTICO."""
    iconos = {'INFO': 'ℹ️', 'HALLAZGO': '🔍', 'ANOMALÍA': '⚠️', 'CRÍTICO': '🚨'}
    icono = iconos.get(nivel, '•')
    HALLAZGOS.append({'seccion': seccion, 'nivel': nivel, 'texto': texto})
    print(f"  {icono} [{nivel}] {texto}")


# =============================================================================
# 1. PANORAMA GENERAL
# =============================================================================

def panorama_general(df):
    print("\n" + "=" * 70)
    print("1. PANORAMA GENERAL")
    print("=" * 70)

    print(f"\n  Filas: {df.shape[0]:,} | Columnas: {df.shape[1]}")

    numericas, categoricas = clasificar_variables(df, excluir=['id_vuelo'])
    print(f"  Numéricas: {len(numericas)} → {numericas}")
    print(f"  Categóricas: {len(categoricas)} → {categoricas}")

    # Nulos
    nulos = df.isnull().sum()
    total_nulos = nulos.sum()
    if total_nulos > 0:
        registrar('Panorama', 'ANOMALÍA', f'{total_nulos} valores nulos en total')
        for col in nulos[nulos > 0].index:
            pct = nulos[col] / len(df) * 100
            registrar('Panorama', 'INFO', f'  {col}: {nulos[col]} nulos ({pct:.1f}%)')
    else:
        registrar('Panorama', 'INFO', 'Sin valores nulos')

    # Duplicados
    dupes_total = df.duplicated().sum()
    dupes_sin_id = df.drop(columns=['id_vuelo'], errors='ignore').duplicated().sum()
    registrar('Panorama', 'INFO' if dupes_total == 0 else 'ANOMALÍA',
              f'Filas duplicadas (con ID): {dupes_total}')
    registrar('Panorama', 'INFO' if dupes_sin_id == 0 else 'ANOMALÍA',
              f'Filas duplicadas (sin ID): {dupes_sin_id}')

    # IDs únicos
    if 'id_vuelo' in df.columns:
        n_ids = df['id_vuelo'].nunique()
        registrar('Panorama', 'INFO' if n_ids == len(df) else 'ANOMALÍA',
                  f'IDs únicos: {n_ids}/{len(df)}')


# =============================================================================
# 2. VARIABLES NUMÉRICAS — una por una
# =============================================================================

def auditar_numerica(df, col, seccion_num):
    seccion = f'Numérica: {col}'
    print(f"\n  --- {col} ---")

    s = df[col].dropna()
    stats = s.describe()

    print(f"    count={int(stats['count']):,}  mean={stats['mean']:.2f}  "
          f"std={stats['std']:.2f}  min={stats['min']:.2f}  max={stats['max']:.2f}")

    # Asimetría
    skew = s.skew()
    if abs(skew) > 2:
        registrar(seccion, 'HALLAZGO', f'Asimetría alta: {skew:.2f}')
    elif abs(skew) > 1:
        registrar(seccion, 'INFO', f'Asimetría moderada: {skew:.2f}')

    # Curtosis
    kurt = s.kurtosis()
    if abs(kurt) > 7:
        registrar(seccion, 'HALLAZGO', f'Curtosis alta: {kurt:.2f} (colas pesadas)')

    # Outliers IQR
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = s[(s < lower) | (s > upper)]
    n_out = len(outliers)
    pct_out = n_out / len(s) * 100

    if n_out > 0:
        registrar(seccion, 'HALLAZGO',
                  f'Outliers IQR: {n_out} ({pct_out:.1f}%) '
                  f'[límites: {lower:.1f} – {upper:.1f}]')
    else:
        registrar(seccion, 'INFO', 'Sin outliers por IQR')

    # Valores sospechosos: ceros o negativos
    n_ceros = (s == 0).sum()
    n_negativos = (s < 0).sum()
    if n_ceros > 0:
        registrar(seccion, 'HALLAZGO', f'Valores = 0: {n_ceros}')
    if n_negativos > 0:
        registrar(seccion, 'ANOMALÍA', f'Valores negativos: {n_negativos}')

    # Concentración: ¿pocos valores dominan?
    top_val = s.value_counts(normalize=True).head(1)
    if top_val.values[0] > 0.25:
        registrar(seccion, 'HALLAZGO',
                  f'Valor dominante: {top_val.index[0]} concentra '
                  f'{top_val.values[0]*100:.1f}% de los datos')


def auditar_numericas(df):
    print("\n" + "=" * 70)
    print("2. AUDITORÍA DE VARIABLES NUMÉRICAS")
    print("=" * 70)

    numericas, _ = clasificar_variables(df, excluir=['id_vuelo'])
    for i, col in enumerate(numericas, 1):
        auditar_numerica(df, col, i)

    # Gráfico resumen: boxplots normalizados
    numericas_data = df[numericas].copy()
    normalized = (numericas_data - numericas_data.min()) / (numericas_data.max() - numericas_data.min())

    fig, ax = plt.subplots(figsize=(14, 6))
    normalized.boxplot(ax=ax, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='#4C78A8', alpha=0.6),
                       medianprops=dict(color='#E45756', linewidth=2))
    ax.set_title('Boxplots normalizados (0-1) — todas las numéricas', fontweight='bold')
    ax.set_ylabel('Valor normalizado')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{AUDIT_DIR}/boxplots_normalizados.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] boxplots_normalizados.png")


# =============================================================================
# 3. VARIABLES CATEGÓRICAS — una por una
# =============================================================================

def auditar_categorica(df, col):
    seccion = f'Categórica: {col}'
    print(f"\n  --- {col} ---")

    s = df[col].dropna()
    n_unique = s.nunique()
    print(f"    Categorías: {n_unique}")

    # Distribución
    dist = s.value_counts(normalize=True)
    print(f"    Top 5: {dict(dist.head(5).round(3))}")

    # Cardinalidad alta
    if n_unique > 30:
        registrar(seccion, 'HALLAZGO', f'Cardinalidad alta: {n_unique} categorías únicas')

    # Desbalance extremo
    if dist.max() > 0.80:
        registrar(seccion, 'ANOMALÍA',
                  f'Categoría dominante: "{dist.idxmax()}" = {dist.max()*100:.1f}%')

    # Categorías raras (< 1%)
    raras = dist[dist < 0.01]
    if len(raras) > 0:
        registrar(seccion, 'HALLAZGO',
                  f'{len(raras)} categorías con < 1% de representación')

    # Relación con demora
    if 'demora' in df.columns and col != 'demora':
        tasa_demora = df.groupby(col)['demora'].mean() * 100
        rango = tasa_demora.max() - tasa_demora.min()
        if rango > 15:
            registrar(seccion, 'HALLAZGO',
                      f'Variación importante en tasa de demora: '
                      f'{tasa_demora.min():.1f}% – {tasa_demora.max():.1f}% '
                      f'(rango {rango:.1f}pp)')
        else:
            registrar(seccion, 'INFO',
                      f'Tasa de demora uniforme entre categorías '
                      f'(rango {rango:.1f}pp)')


def auditar_categoricas(df):
    print("\n" + "=" * 70)
    print("3. AUDITORÍA DE VARIABLES CATEGÓRICAS")
    print("=" * 70)

    _, categoricas = clasificar_variables(df, excluir=['id_vuelo'])
    for col in categoricas:
        auditar_categorica(df, col)


# =============================================================================
# 4. COHERENCIA CRUZADA (relaciones entre variables)
# =============================================================================

def coherencia_cruzada(df):
    print("\n" + "=" * 70)
    print("4. COHERENCIA CRUZADA")
    print("=" * 70)

    # 4a. Ruta EZE ↔ AEP (ya sabemos que no existe para pasajeros)
    print("\n  --- Rutas imposibles ---")
    if all(c in df.columns for c in ['aeropuerto_origen', 'aeropuerto_destino']):
        eze_aep = df[
            (df['aeropuerto_origen'].isin(['EZE', 'AEP'])) &
            (df['aeropuerto_destino'].isin(['EZE', 'AEP'])) &
            (df['aeropuerto_origen'] != df['aeropuerto_destino'])
        ]
        if len(eze_aep) > 0:
            registrar('Coherencia', 'CRÍTICO',
                      f'Ruta EZE ↔ AEP: {len(eze_aep)} vuelos '
                      f'({len(eze_aep)/len(df)*100:.1f}%) — no existen vuelos '
                      f'comerciales de pasajeros en esta ruta')

        # Vuelos con mismo origen y destino
        misma_ruta = df[df['aeropuerto_origen'] == df['aeropuerto_destino']]
        if len(misma_ruta) > 0:
            registrar('Coherencia', 'ANOMALÍA',
                      f'Vuelos con origen = destino: {len(misma_ruta)}')
        else:
            registrar('Coherencia', 'INFO', 'No hay vuelos con origen = destino')

    # 4b. Velocidad derivada
    print("\n  --- Velocidad derivada ---")
    if all(c in df.columns for c in ['distancia_vuelo', 'tiempo_estimado_vuelo']):
        df_vel = df.copy()
        df_vel['velocidad_kmh'] = (df_vel['distancia_vuelo'] / df_vel['tiempo_estimado_vuelo']) * 60

        # Rangos operativos reales de aviación comercial
        ultra_lentos = df_vel[df_vel['velocidad_kmh'] < 150]
        supersonic = df_vel[df_vel['velocidad_kmh'] > 950]

        if len(ultra_lentos) > 0:
            rutas_lentas = ultra_lentos.groupby(
                ['aeropuerto_origen', 'aeropuerto_destino']
            ).size().reset_index(name='vuelos')
            registrar('Coherencia', 'HALLAZGO',
                      f'Velocidad < 150 km/h: {len(ultra_lentos)} vuelos '
                      f'(rutas: {rutas_lentas.values.tolist()})')

        if len(supersonic) > 0:
            registrar('Coherencia', 'ANOMALÍA',
                      f'Velocidad > 950 km/h (supersónica): {len(supersonic)}')
        else:
            registrar('Coherencia', 'INFO', 'No hay velocidades supersónicas')

        # Distribución de velocidad por tipo de avión
        stats_vel = (
            df_vel.groupby('tipo_avion')['velocidad_kmh']
            .agg(['mean', 'std', 'min', 'max'])
            .round(1)
        )
        print(f"\n    Velocidad media por tipo de avión:")
        for avion, row in stats_vel.iterrows():
            print(f"      {avion:15s}  μ={row['mean']:>6.1f}  σ={row['std']:>5.1f}  "
                  f"[{row['min']:>5.1f} – {row['max']:>5.1f}]")

        # ¿Todos los aviones tienen el mismo rango de velocidad?
        rango_medias = stats_vel['mean'].max() - stats_vel['mean'].min()
        if rango_medias < 30:
            registrar('Coherencia', 'HALLAZGO',
                      f'Velocidad media casi idéntica entre tipos de avión '
                      f'(rango = {rango_medias:.1f} km/h) — '
                      f'tipo_avion probablemente NO influye en velocidad')

    # 4c. Distancia vs Tiempo estimado — correlación esperada
    print("\n  --- Distancia vs Tiempo ---")
    if all(c in df.columns for c in ['distancia_vuelo', 'tiempo_estimado_vuelo']):
        corr = df['distancia_vuelo'].corr(df['tiempo_estimado_vuelo'])
        registrar('Coherencia', 'INFO', f'Correlación distancia ↔ tiempo: {corr:.4f}')
        if corr < 0.90:
            registrar('Coherencia', 'ANOMALÍA',
                      f'Correlación menor a esperada (< 0.90), '
                      f'posibles inconsistencias')

    # 4d. Ocupación del vuelo — ¿valores extremos?
    print("\n  --- Ocupación ---")
    if 'ocupacion_vuelo' in df.columns:
        s = df['ocupacion_vuelo']
        if s.max() > 100:
            registrar('Coherencia', 'ANOMALÍA',
                      f'Ocupación > 100%: {(s > 100).sum()} vuelos (sobreventa?)')
        if s.min() < 0:
            registrar('Coherencia', 'ANOMALÍA', f'Ocupación negativa: {(s < 0).sum()}')
        if (s == 0).sum() > 0:
            registrar('Coherencia', 'HALLAZGO',
                      f'Vuelos con 0% ocupación: {(s == 0).sum()}')

    # 4e. Simetría de rutas
    print("\n  --- Simetría de rutas ---")
    if all(c in df.columns for c in ['aeropuerto_origen', 'aeropuerto_destino']):
        rutas = df.groupby(['aeropuerto_origen', 'aeropuerto_destino']).size().reset_index(name='vuelos')
        rutas['ruta'] = rutas.apply(
            lambda r: tuple(sorted([r['aeropuerto_origen'], r['aeropuerto_destino']])), axis=1
        )
        pares = rutas.groupby('ruta')['vuelos'].agg(['sum', 'count']).reset_index()
        solo_ida = pares[pares['count'] == 1]
        if len(solo_ida) > 0:
            registrar('Coherencia', 'HALLAZGO',
                      f'{len(solo_ida)} rutas operan solo en una dirección')
        else:
            registrar('Coherencia', 'INFO', 'Todas las rutas operan en ambas direcciones')

        # Top 10 rutas por volumen
        top_rutas = rutas.sort_values('vuelos', ascending=False).head(10)
        print(f"\n    Top 10 rutas por volumen:")
        for _, r in top_rutas.iterrows():
            print(f"      {r['aeropuerto_origen']} → {r['aeropuerto_destino']}: {r['vuelos']} vuelos")


# =============================================================================
# 5. BALANCE DE LA VARIABLE OBJETIVO
# =============================================================================

def auditar_target(df, target='demora'):
    print("\n" + "=" * 70)
    print("5. VARIABLE OBJETIVO")
    print("=" * 70)

    if target not in df.columns:
        registrar('Target', 'CRÍTICO', f'Variable "{target}" no encontrada')
        return

    dist = df[target].value_counts(normalize=True)
    print(f"\n  Distribución:")
    for val, pct in dist.items():
        label = 'No demorado' if val == 0 else 'Demorado'
        print(f"    {label} ({val}): {pct*100:.1f}%")

    # Balance
    ratio = dist.min() / dist.max()
    if ratio < 0.25:
        registrar('Target', 'ANOMALÍA',
                  f'Desbalance severo: ratio {ratio:.2f} — considerar SMOTE/undersampling')
    elif ratio < 0.5:
        registrar('Target', 'HALLAZGO',
                  f'Desbalance moderado: ratio {ratio:.2f}')
    else:
        registrar('Target', 'INFO', f'Balance aceptable: ratio {ratio:.2f}')


# =============================================================================
# 6. CORRELACIONES CON DEMORA
# =============================================================================

def auditar_correlaciones(df):
    print("\n" + "=" * 70)
    print("6. CORRELACIONES CON DEMORA")
    print("=" * 70)

    numericas, _ = clasificar_variables(df, excluir=['id_vuelo'])
    if 'demora' not in df.columns:
        return

    cols_corr = [c for c in numericas if c != 'demora'] + ['demora']
    corrs = df[cols_corr].corr()['demora'].drop('demora')
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)

    print(f"\n  {'Variable':<30} {'Correlación':>12} {'Fuerza':>10}")
    print("  " + "-" * 54)
    for var, val in corrs.items():
        if abs(val) > 0.3:
            fuerza = 'FUERTE'
        elif abs(val) > 0.1:
            fuerza = 'DÉBIL'
        else:
            fuerza = 'NULA'
        print(f"  {var:<30} {val:>12.4f} {fuerza:>10}")
        if abs(val) < 0.05:
            registrar('Correlaciones', 'HALLAZGO',
                      f'{var}: correlación {val:.4f} — candidata a eliminación')

    # Multicolinealidad entre features
    print(f"\n  --- Multicolinealidad ---")
    corr_matrix = df[numericas].corr()
    high_corr = []
    for i in range(len(numericas)):
        for j in range(i + 1, len(numericas)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.7:
                high_corr.append((numericas[i], numericas[j], val))
    if high_corr:
        for v1, v2, val in high_corr:
            registrar('Correlaciones', 'HALLAZGO',
                      f'Alta correlación: {v1} ↔ {v2} = {val:.4f} '
                      f'(posible redundancia)')
    else:
        registrar('Correlaciones', 'INFO', 'Sin multicolinealidad alta (> 0.7)')

    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    full_corr = df[numericas + ['demora']].corr()
    mask_full = np.triu(np.ones_like(full_corr, dtype=bool))
    sns.heatmap(full_corr, mask=mask_full, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Matriz de Correlación — Auditoría', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{AUDIT_DIR}/correlaciones_auditoria.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] correlaciones_auditoria.png")


# =============================================================================
# 7. GRÁFICO DE ANOMALÍAS POR VARIABLE
# =============================================================================

def grafico_resumen_anomalias(df):
    print("\n" + "=" * 70)
    print("7. VISUALIZACIONES DE AUDITORÍA")
    print("=" * 70)

    numericas, _ = clasificar_variables(df, excluir=['id_vuelo'])

    # Histogramas de todas las numéricas
    n_cols = 3
    n_rows = (len(numericas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(numericas):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color='#4C78A8', alpha=0.6)
        axes[i].set_title(f'{col}\n(skew={df[col].skew():.2f})', fontsize=10)
        axes[i].axvline(df[col].mean(), color='#E45756', linestyle='--', linewidth=1.5)
        axes[i].axvline(df[col].median(), color='#4CAF50', linestyle='-', linewidth=1.5)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Distribución de variables numéricas (rojo=media, verde=mediana)',
                 fontweight='bold', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{AUDIT_DIR}/distribuciones_numericas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [+] distribuciones_numericas.png")

    # Categóricas: tasa de demora por categoría
    _, categoricas = clasificar_variables(df, excluir=['id_vuelo'])
    cats_graficar = [c for c in categoricas if df[c].nunique() <= 20 and c != 'demora']

    if cats_graficar and 'demora' in df.columns:
        n_rows_cat = (len(cats_graficar) + 1) // 2
        fig, axes = plt.subplots(n_rows_cat, 2, figsize=(16, 5 * n_rows_cat))
        axes = axes.flatten()

        for i, col in enumerate(cats_graficar):
            tasa = df.groupby(col)['demora'].mean().sort_values() * 100
            tasa.plot.barh(ax=axes[i], color='#E45756', alpha=0.8)
            axes[i].set_title(f'{col} → % demora', fontweight='bold', fontsize=11)
            axes[i].set_xlabel('% demorados')
            for idx, (cat, val) in enumerate(tasa.items()):
                axes[i].text(val + 0.3, idx, f'{val:.1f}%', va='center', fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle('Tasa de demora por variable categórica',
                     fontweight='bold', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(f'{AUDIT_DIR}/categoricas_vs_demora.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   [+] categoricas_vs_demora.png")


# =============================================================================
# 8. REPORTE FINAL
# =============================================================================

def imprimir_reporte():
    print("\n" + "=" * 70)
    print("REPORTE FINAL — HALLAZGOS DE AUDITORÍA")
    print("=" * 70)

    # Agrupar por nivel
    for nivel in ['CRÍTICO', 'ANOMALÍA', 'HALLAZGO', 'INFO']:
        items = [h for h in HALLAZGOS if h['nivel'] == nivel]
        if not items:
            continue

        iconos = {'CRÍTICO': '🚨', 'ANOMALÍA': '⚠️', 'HALLAZGO': '🔍', 'INFO': 'ℹ️'}
        print(f"\n  {iconos[nivel]} {nivel} ({len(items)})")
        print("  " + "-" * 60)
        for h in items:
            print(f"    [{h['seccion']}] {h['texto']}")

    # Resumen de accionables
    criticos = [h for h in HALLAZGOS if h['nivel'] == 'CRÍTICO']
    anomalias = [h for h in HALLAZGOS if h['nivel'] == 'ANOMALÍA']
    hallazgos = [h for h in HALLAZGOS if h['nivel'] == 'HALLAZGO']

    print(f"\n  {'=' * 50}")
    print(f"  RESUMEN: {len(criticos)} críticos | {len(anomalias)} anomalías | "
          f"{len(hallazgos)} hallazgos")
    print(f"  {'=' * 50}")


# =============================================================================
# PIPELINE
# =============================================================================

def main():
    print("=" * 70)
    print("AUDITORÍA COMPLETA DEL DATASET DE VUELOS")
    print("=" * 70)

    df = cargar_datos()

    panorama_general(df)
    auditar_numericas(df)
    auditar_categoricas(df)
    coherencia_cruzada(df)
    auditar_target(df)
    auditar_correlaciones(df)
    grafico_resumen_anomalias(df)
    imprimir_reporte()

    print(f"\n  Gráficos de auditoría en: {AUDIT_DIR}/")


if __name__ == '__main__':
    main()
