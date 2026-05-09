"""
Análisis enfocados en relaciones útiles entre variables.

Solo se mantienen análisis donde las variables tienen correlación real
con la variable objetivo (demora) o entre sí (ej: distancia ↔ tiempo).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    OUTPUT_DIR,
    clasificar_variables,
    detectar_columnas_aeropuerto,
    expandir_aeropuertos,
    serie_demorada,
)


# ── 1. Visibilidad vs Demora ─────────────────────────────────────────────────

def visibilidad_vs_demora(df, col_vis='visibilidad', target='demora', output_dir=OUTPUT_DIR):
    """
    Analiza cómo la visibilidad impacta en la tasa de demora.
    Genera boxplot comparativo y curva de tasa de demora por rango de visibilidad.
    """
    print("\n" + "-" * 70)
    print("VISIBILIDAD vs DEMORA")
    print("-" * 70)

    faltantes = [c for c in [col_vis, target] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas: {faltantes}")
        return None

    base = df[[col_vis, target]].dropna()
    base['demorado'] = serie_demorada(base[target])

    # Estadísticos por grupo
    stats_grupo = base.groupby('demorado')[col_vis].describe().round(2)
    print(f"\n  Visibilidad por estado de demora:")
    print(f"  {'Grupo':<15} {'Media':>8} {'Mediana':>8} {'Std':>8}")
    print("  " + "-" * 41)
    for grupo, row in stats_grupo.iterrows():
        etiqueta = "Demorado" if grupo == 1 else "No demorado"
        print(f"  {etiqueta:<15} {row['mean']:>8.2f} {row['50%']:>8.2f} {row['std']:>8.2f}")

    # Correlación puntual
    corr = base[col_vis].corr(base['demorado'])
    print(f"\n  Correlación visibilidad ↔ demora: {corr:.4f}")

    # Tasa de demora por rangos de visibilidad
    base['rango_vis'] = pd.cut(base[col_vis], bins=10)
    tasa = (
        base.groupby('rango_vis', observed=True)
        .agg(
            total=('demorado', 'count'),
            pct_demora=('demorado', lambda x: x.mean() * 100)
        )
        .reset_index()
    )
    tasa['pct_demora'] = tasa['pct_demora'].round(2)

    # --- Gráfico 1: Boxplot comparativo ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    base['estado'] = base['demorado'].map({0: 'No demorado', 1: 'Demorado'})
    sns.boxplot(
        data=base, x='estado', y=col_vis, ax=axes[0],
        hue='estado', palette={'No demorado': '#4CAF50', 'Demorado': '#F44336'},
        legend=False,
    )
    axes[0].set_title('Visibilidad según estado de demora', fontweight='bold')
    axes[0].set_ylabel('Visibilidad')
    axes[0].set_xlabel('')

    # --- Gráfico 2: Tasa de demora por rango ---
    rangos_str = [str(r) for r in tasa['rango_vis']]
    axes[1].bar(rangos_str, tasa['pct_demora'], color='#E45756', alpha=0.85)
    axes[1].set_title('Tasa de demora por rango de visibilidad', fontweight='bold')
    axes[1].set_ylabel('% de vuelos demorados')
    axes[1].set_xlabel('Rango de visibilidad')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/visibilidad_vs_demora.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] visibilidad_vs_demora.png")

    return tasa


# ── 2. Distancia vs Tiempo estimado ──────────────────────────────────────────

def distancia_vs_tiempo(
    df,
    col_dist='distancia_vuelo',
    col_tiempo='tiempo_estimado_vuelo',
    output_dir=OUTPUT_DIR
):
    """
    Valida la relación lineal esperada entre distancia y tiempo estimado.
    Permite detectar anomalías (vuelos con tiempos inconsistentes).
    """
    print("\n" + "-" * 70)
    print("DISTANCIA vs TIEMPO ESTIMADO")
    print("-" * 70)

    faltantes = [c for c in [col_dist, col_tiempo] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas: {faltantes}")
        return None

    base = df[[col_dist, col_tiempo]].dropna()

    corr = base[col_dist].corr(base[col_tiempo])
    print(f"\n  Correlación distancia ↔ tiempo estimado: {corr:.4f}")

    # Velocidad derivada (km/h, asumiendo tiempo en minutos)
    base['velocidad_kmh'] = (base[col_dist] / base[col_tiempo]) * 60
    stats_vel = base['velocidad_kmh'].describe().round(2)
    print(f"\n  Velocidad derivada (km/h):")
    print(f"    Media: {stats_vel['mean']:.0f} | "
          f"Mediana: {stats_vel['50%']:.0f} | "
          f"Std: {stats_vel['std']:.0f} | "
          f"Min: {stats_vel['min']:.0f} | "
          f"Max: {stats_vel['max']:.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter distancia vs tiempo
    sns.scatterplot(
        data=base, x=col_dist, y=col_tiempo,
        alpha=0.3, color='steelblue', ax=axes[0], s=10
    )
    axes[0].set_title(f'Distancia vs Tiempo estimado (r={corr:.3f})', fontweight='bold')
    axes[0].set_xlabel('Distancia (km)')
    axes[0].set_ylabel('Tiempo estimado (min)')

    # Distribución de velocidad
    sns.histplot(base['velocidad_kmh'], kde=True, color='coral', ax=axes[1])
    axes[1].set_title('Distribución de velocidad derivada', fontweight='bold')
    axes[1].set_xlabel('Velocidad (km/h)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/distancia_vs_tiempo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] distancia_vs_tiempo.png")

    return base


# ── 3. Condiciones climáticas vs Demora ──────────────────────────────────────

def clima_vs_demora(
    df,
    col_clima='condiciones_climaticas',
    target='demora',
    output_dir=OUTPUT_DIR
):
    """
    Compara la tasa de demora según las condiciones climáticas.
    """
    print("\n" + "-" * 70)
    print("CONDICIONES CLIMÁTICAS vs DEMORA")
    print("-" * 70)

    faltantes = [c for c in [col_clima, target] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas: {faltantes}")
        return None

    base = df[[col_clima, target]].dropna()
    base['demorado'] = serie_demorada(base[target])

    resumen = (
        base.groupby(col_clima, as_index=False)
        .agg(
            total=('demorado', 'count'),
            pct_demora=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('pct_demora', ascending=False)
    )
    resumen['pct_demora'] = resumen['pct_demora'].round(2)

    print(f"\n  {'Condición':<25} {'Vuelos':>8} {'% Demora':>10}")
    print("  " + "-" * 45)
    for _, row in resumen.iterrows():
        print(f"  {row[col_clima]:<25} {int(row['total']):>8} {row['pct_demora']:>9.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    orden = resumen.sort_values('pct_demora', ascending=True)
    ax.barh(orden[col_clima], orden['pct_demora'], color='#E45756', alpha=0.85)
    for i, (_, row) in enumerate(orden.iterrows()):
        ax.text(row['pct_demora'] + 0.3, i, f"{row['pct_demora']:.1f}%", va='center', fontsize=9)
    ax.set_title('Tasa de demora por condición climática', fontweight='bold')
    ax.set_xlabel('% de vuelos demorados')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/clima_vs_demora.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] clima_vs_demora.png")

    return resumen


# ── 4. Congestión aérea vs Demora ────────────────────────────────────────────

def congestion_vs_demora(
    df,
    col_congestion='congestion_aerea',
    target='demora',
    output_dir=OUTPUT_DIR
):
    """
    Compara la tasa de demora según el nivel de congestión aérea.
    """
    print("\n" + "-" * 70)
    print("CONGESTIÓN AÉREA vs DEMORA")
    print("-" * 70)

    faltantes = [c for c in [col_congestion, target] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas: {faltantes}")
        return None

    base = df[[col_congestion, target]].dropna()
    base['demorado'] = serie_demorada(base[target])

    resumen = (
        base.groupby(col_congestion, as_index=False)
        .agg(
            total=('demorado', 'count'),
            pct_demora=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('pct_demora', ascending=False)
    )
    resumen['pct_demora'] = resumen['pct_demora'].round(2)

    print(f"\n  {'Congestión':<20} {'Vuelos':>8} {'% Demora':>10}")
    print("  " + "-" * 40)
    for _, row in resumen.iterrows():
        print(f"  {row[col_congestion]:<20} {int(row['total']):>8} {row['pct_demora']:>9.2f}%")

    fig, ax = plt.subplots(figsize=(8, 4))
    orden = resumen.sort_values('pct_demora', ascending=True)
    ax.barh(orden[col_congestion], orden['pct_demora'], color='#4C78A8', alpha=0.85)
    for i, (_, row) in enumerate(orden.iterrows()):
        ax.text(row['pct_demora'] + 0.3, i, f"{row['pct_demora']:.1f}%", va='center', fontsize=9)
    ax.set_title('Tasa de demora por nivel de congestión', fontweight='bold')
    ax.set_xlabel('% de vuelos demorados')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/congestion_vs_demora.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] congestion_vs_demora.png")

    return resumen


# ── 5. Hora de salida vs Demora ──────────────────────────────────────────────

def hora_vs_demora(
    df,
    col_hora='hora_salida_programada',
    target='demora',
    output_dir=OUTPUT_DIR
):
    """
    Analiza cómo la hora de salida influye en la tasa de demora.
    """
    print("\n" + "-" * 70)
    print("HORA DE SALIDA vs DEMORA")
    print("-" * 70)

    faltantes = [c for c in [col_hora, target] if c not in df.columns]
    if faltantes:
        print(f"\n  [!] Faltan columnas: {faltantes}")
        return None

    base = df[[col_hora, target]].copy()
    hora_txt = base[col_hora].astype(str).str.extract(r'(\d{1,2})', expand=False)
    base['hora'] = pd.to_numeric(hora_txt, errors='coerce')
    base = base[base['hora'].between(0, 23, inclusive='both')]

    if base.empty:
        print("\n  [!] No se pudieron interpretar horas válidas (0-23).")
        return None

    base['demorado'] = serie_demorada(base[target])

    resumen = (
        base.groupby('hora', as_index=False)
        .agg(
            vuelos=('hora', 'count'),
            pct_demorados=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('hora')
    )
    resumen['pct_demorados'] = resumen['pct_demorados'].round(2)

    corr = resumen['hora'].corr(resumen['pct_demorados'])
    pico = resumen.loc[resumen['pct_demorados'].idxmax()]

    print(f"\n  Correlación hora ↔ % demora: {corr:.3f}")
    print(f"  Hora con mayor demora: {int(pico['hora']):02d}:00 "
          f"({pico['pct_demorados']:.2f}%)")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(resumen['hora'], resumen['vuelos'], color='#4C78A8', alpha=0.5, label='Vuelos')
    ax2 = ax.twinx()
    ax2.plot(resumen['hora'], resumen['pct_demorados'], color='#E45756',
             marker='o', linewidth=2, label='% demorados')

    ax.set_xlabel('Hora de salida')
    ax.set_ylabel('Cantidad de vuelos', color='#4C78A8')
    ax2.set_ylabel('% de vuelos demorados', color='#E45756')
    ax.set_xticks(range(0, 24))
    ax.set_title('Volumen de vuelos y tasa de demora por hora', fontweight='bold')

    lineas = ax.containers + ax2.get_lines()
    ax.legend([ax.patches[0], ax2.get_lines()[0]], ['Vuelos', '% demorados'], loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/hora_vs_demora.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] hora_vs_demora.png")

    return resumen


# ── 6. Demora por aeropuerto (desglosada por clima) ──────────────────────────

def demora_por_aeropuerto(
    df,
    target='demora',
    col_clima='condiciones_climaticas',
    top_n=20,
    output_dir=OUTPUT_DIR
):
    """
    Tasa de demora por aeropuerto, desglosada por condición climática.

    Cada barra muestra el % de vuelos demorados en ese aeropuerto,
    y los segmentos de color indican cuánto aporta cada clima a ese %.
    """
    print("\n" + "-" * 70)
    print("DEMORA POR AEROPUERTO (desglose por clima)")
    print("-" * 70)

    for col in [target, col_clima]:
        if col not in df.columns:
            print(f"\n  [!] No se encontró la columna '{col}'.")
            return None

    columnas_aeropuerto = detectar_columnas_aeropuerto(df)
    if not columnas_aeropuerto:
        print("\n  [!] No se detectaron columnas de aeropuerto.")
        return None

    largo = expandir_aeropuertos(
        df, columnas_aeropuerto, columnas_extra=[target, col_clima]
    )
    largo['demorado'] = serie_demorada(largo[target])

    # Resumen general para ordenar y para la tabla
    resumen = (
        largo.groupby('aeropuerto', as_index=False)
        .agg(
            total_vuelos=('aeropuerto', 'count'),
            pct_demorados=('demorado', lambda x: x.mean() * 100)
        )
        .sort_values('total_vuelos', ascending=False)
    )
    resumen['pct_demorados'] = resumen['pct_demorados'].round(2)

    top_aeropuertos = resumen.head(top_n)['aeropuerto'].tolist()
    largo_top = largo[largo['aeropuerto'].isin(top_aeropuertos)]

    # Tabla consola
    print(f"\n  Top {len(top_aeropuertos)} aeropuertos:")
    print(f"  {'Aeropuerto':<15} {'Vuelos':>10} {'% Demora':>12}")
    print("  " + "-" * 40)
    for _, row in resumen.head(top_n).iterrows():
        print(f"  {row['aeropuerto']:<15} {int(row['total_vuelos']):>10} "
              f"{row['pct_demorados']:>11.2f}%")

    # ── Calcular aporte de cada clima al % de demora por aeropuerto ──
    # Para cada (aeropuerto, clima): demorados_clima / total_vuelos_aeropuerto * 100
    clima_demora = (
        largo_top.groupby(['aeropuerto', col_clima], as_index=False)
        .agg(demorados_clima=('demorado', 'sum'))
    )
    totales = (
        largo_top.groupby('aeropuerto', as_index=False)
        .agg(total=('demorado', 'count'))
    )
    clima_demora = clima_demora.merge(totales, on='aeropuerto')
    clima_demora['pct_aporte'] = (
        clima_demora['demorados_clima'] / clima_demora['total'] * 100
    ).round(2)

    # Pivotar: aeropuerto × clima → pct_aporte
    pivot = clima_demora.pivot(
        index='aeropuerto', columns=col_clima, values='pct_aporte'
    ).fillna(0)

    # Ordenar por total de demora descendente
    orden_total = resumen.set_index('aeropuerto')['pct_demorados']
    pivot = pivot.loc[
        pivot.index.intersection(orden_total.index)
    ]
    pivot['_total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('_total', ascending=True).drop(columns='_total')

    # ── Gráfico: barras horizontales stacked por clima ──
    colores_clima = {
        'Despejado': '#4CAF50',
        'Nublado': '#78909C',
        'Lluvia': '#42A5F5',
        'Tormenta': '#EF5350',
        'Niebla': '#FFA726',
    }
    # Fallback para climas no contemplados
    palette_extra = ['#AB47BC', '#26A69A', '#8D6E63', '#EC407A']
    climas = pivot.columns.tolist()
    colores = []
    idx_extra = 0
    for c in climas:
        if c in colores_clima:
            colores.append(colores_clima[c])
        else:
            colores.append(palette_extra[idx_extra % len(palette_extra)])
            idx_extra += 1

    fig, ax = plt.subplots(figsize=(13, max(7, len(pivot) * 0.5)))

    left = np.zeros(len(pivot))
    for clima, color in zip(climas, colores):
        valores = pivot[clima].values
        ax.barh(pivot.index, valores, left=left, color=color,
                label=clima, edgecolor='white', linewidth=0.5)
        left += valores

    # Etiqueta con % total al final de cada barra
    for i, aeropuerto in enumerate(pivot.index):
        total = left[i]
        ax.text(total + 0.3, i, f'{total:.1f}%', va='center', fontsize=9,
                fontweight='bold')

    ax.set_xlabel('% de vuelos demorados')
    ax.set_ylabel('Aeropuerto')
    ax.set_title('Tasa de demora por aeropuerto (desglose por clima)',
                 fontweight='bold', fontsize=14)
    ax.legend(title='Clima', loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/demora_por_aeropuerto.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] demora_por_aeropuerto.png")

    return resumen


# ── 7. Velocidad por tipo de avión ───────────────────────────────────────────

def velocidad_por_tipo_avion(df, output_dir=OUTPUT_DIR):
    """
    Calcula velocidad derivada y compara distribución por tipo de avión.
    Útil para detectar registros anómalos.
    """
    print("\n" + "-" * 70)
    print("VELOCIDAD POR TIPO DE AVIÓN")
    print("-" * 70)

    for col in ['distancia_vuelo', 'tiempo_estimado_vuelo', 'tipo_avion']:
        if col not in df.columns:
            print(f"\n  [!] Falta columna '{col}'.")
            return None

    base = df[['distancia_vuelo', 'tiempo_estimado_vuelo', 'tipo_avion']].dropna()
    base['velocidad_kmh'] = (base['distancia_vuelo'] / base['tiempo_estimado_vuelo']) * 60

    resumen = (
        base.groupby('tipo_avion')['velocidad_kmh']
        .describe()
        .round(1)
        .sort_values('mean', ascending=False)
    )
    print(f"\n{resumen[['count', 'mean', '50%', 'std', 'min', 'max']].to_string()}")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=base, x='velocidad_kmh', y='tipo_avion', ax=ax,
                hue='tipo_avion', palette='Set2', legend=False)
    ax.set_title('Distribución de velocidad por tipo de avión', fontweight='bold')
    ax.set_xlabel('Velocidad (km/h)')
    ax.set_ylabel('Tipo de avión')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/velocidad_por_tipo_avion.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] velocidad_por_tipo_avion.png")

    return resumen


# ── 8. Balance de clases ─────────────────────────────────────────────────────

def balance_clases(df, target='demora', output_dir=OUTPUT_DIR):
    """Visualiza el balance de clases de la variable objetivo."""
    print("\n" + "-" * 70)
    print("BALANCE DE CLASES (Variable Objetivo)")
    print("-" * 70)

    conteo = df[target].value_counts()
    labels = ['No Demorado', 'Demorado']
    colors = ['#4CAF50', '#F44336']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(labels, conteo.values, color=colors, edgecolor='white', linewidth=2)
    for i, v in enumerate(conteo.values):
        axes[0].text(i, v + 100, f'{v}\n({v/len(df)*100:.1f}%)',
                     ha='center', fontweight='bold')
    axes[0].set_title('Balance de Clases', fontweight='bold')
    axes[0].set_ylabel('Cantidad')

    axes[1].pie(conteo.values, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 12})
    axes[1].set_title('Proporción de Clases', fontweight='bold')

    plt.suptitle(f'Variable Objetivo: {target}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/balance_clases.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] balance_clases.png")


# ── 9. Matriz de correlación (enfocada en demora) ────────────────────────────

def matriz_correlacion(df, excluir=None, output_dir=OUTPUT_DIR):
    """Genera la matriz de correlación con ranking de correlaciones con demora."""
    print("\n" + "-" * 70)
    print("MATRIZ DE CORRELACIÓN")
    print("-" * 70)

    numericas, _ = clasificar_variables(df, excluir=excluir)
    if 'demora' not in (excluir or []) and 'demora' in df.columns:
        if 'demora' not in numericas:
            numericas.append('demora')

    corr = df[numericas].corr()

    if 'demora' in corr.columns:
        print("\n  --- Correlaciones con 'demora' ---")
        corr_target = corr['demora'].drop('demora').sort_values(key=abs, ascending=False)
        for var, val in corr_target.items():
            fuerza = "fuerte" if abs(val) > 0.3 else "débil" if abs(val) > 0.1 else "nula"
            print(f"    {var:<30} {val:>8.4f}  ({fuerza})")

    fig, ax = plt.subplots(figsize=(10, 8))
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
