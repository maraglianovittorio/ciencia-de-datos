"""
Gráficos de torta para cruce_vuelos_clientes.csv
Muestra proporciones de demora y etiqueta_cliente.
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUT_DIR

RUTA_CSV = "TPI/cruces/cruce_vuelos_clientes.csv"
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "torta")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.figsize': (10, 6),
    'font.size': 12,
    'axes.titlesize': 15,
    'axes.labelsize': 13,
})

COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']

df = pd.read_csv(RUTA_CSV)
total = len(df)

def autopct_abs(pct):
    absolute = int(round(pct / 100.0 * total))
    return f"{pct:.1f}%\n({absolute:,})"

# ── 1. Demora ────────────────────────────────────────────────────────────
fig, ax = plt.subplots()
demora_counts = df['demora'].value_counts().sort_index()
labels_demora = {0: 'Sin demora', 1: 'Con demora'}
sizes = demora_counts.values
labels = [labels_demora.get(i, f'Demora {i}') for i in demora_counts.index]
colors = [COLORS[0] if i == 0 else COLORS[1] for i in demora_counts.index]

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=autopct_abs,
    startangle=90, colors=colors, explode=(0.02, 0.02),
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Proporción de vuelos con demora', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "torta_demora.png"), dpi=150)
plt.close(fig)
print("[OK] torta_demora.png")

# ── 2. Etiqueta Cliente ───────────────────────────────────────────────────
fig, ax = plt.subplots()
etq_counts = df['etiqueta_cliente'].value_counts(dropna=False).sort_index()
labels_etq = {0.0: 'Etiqueta 0', 1.0: 'Etiqueta 1', np.nan: 'Sin etiqueta'}
sizes = etq_counts.values
labels = [labels_etq.get(i, str(i)) for i in etq_counts.index]
# Mapa de colores
color_map = {0.0: COLORS[2], 1.0: COLORS[3], np.nan: '#cccccc'}
colors = [color_map.get(i, COLORS[4]) for i in etq_counts.index]

wedges, texts, autotexts = ax.pie(
    sizes, labels=labels, autopct=autopct_abs,
    startangle=90, colors=colors, explode=(0.02, 0.02, 0.02),
    textprops={'fontsize': 11}
)
for at in autotexts:
    at.set_fontsize(10)
ax.set_title('Proporción de etiqueta de cliente', fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "torta_etiqueta_cliente.png"), dpi=150)
plt.close(fig)
print("[OK] torta_etiqueta_cliente.png")

# ── 3. Combinado: demora + etiqueta (subplots) ───────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

wedges1, texts1, autotexts1 = ax1.pie(
    demora_counts.values,
    labels=[labels_demora[i] for i in demora_counts.index],
    autopct=autopct_abs, startangle=90,
    colors=[COLORS[0], COLORS[1]], explode=(0.02, 0.02),
    textprops={'fontsize': 11}
)
ax1.set_title('Demora', fontweight='bold')

wedges2, texts2, autotexts2 = ax2.pie(
    sizes, labels=labels, autopct=autopct_abs,
    startangle=90, colors=colors, explode=(0.02, 0.02, 0.02),
    textprops={'fontsize': 11}
)
ax2.set_title('Etiqueta Cliente', fontweight='bold')

fig.suptitle('Distribución de variables del cruce', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "torta_combinado.png"), dpi=150)
plt.close(fig)
print("[OK] torta_combinado.png")

# ── 4. Demora segmentado por etiqueta_cliente ────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
etiquetas_valores = [0.0, 1.0]
titulos = ['Clientes con etiqueta 0', 'Clientes con etiqueta 1']

for ax, etq, titulo in zip(axes, etiquetas_valores, titulos):
    sub = df[df['etiqueta_cliente'] == etq]
    counts = sub['demora'].value_counts().sort_index()
    sizes_sub = counts.values
    labels_sub = [labels_demora[i] for i in counts.index]
    wedges, texts, autotexts = ax.pie(
        sizes_sub, labels=labels_sub, autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct/100*len(sub))):,})",
        startangle=90, colors=[COLORS[0], COLORS[1]], explode=(0.02, 0.02),
        textprops={'fontsize': 10}
    )
    ax.set_title(titulo, fontweight='bold')

fig.suptitle('Demora segmentada por etiqueta de cliente', fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "torta_demora_por_etiqueta.png"), dpi=150)
plt.close(fig)
print("[OK] torta_demora_por_etiqueta.png")

print(f"\nTodos los gráficos guardados en: {OUTPUT_DIR}/")
