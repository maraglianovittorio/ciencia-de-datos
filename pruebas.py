"""
================================================================================
Pruebas — Evaluación de Modelos Entrenados
================================================================================
Dado un modelo (knn/arboles) y un número de modelo, carga el modelo y la vista
minable correspondiente y ejecuta la evaluación completa.

Permite comparar el rendimiento de distintos modelos sobre distintas vistas.
================================================================================
"""

import os
import json
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)
import joblib

VISTAS_MINABLES_DIR = 'vistas_minables'
MODELOS_BASE = 'models'
TARGET = 'demora'
TEST_SIZE = 0.3
RANDOM_STATE = 42

OUTPUT_DIR = 'TPI/graficos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODELOS_PERMITIDOS = ['knn', 'arbol']
SPINNING = False


def spin(msg="Procesando"):
    global SPINNING
    SPINNING = True
    chars = r'|/-\|'
    idx = 0
    while SPINNING:
        print(f"\r  {msg} {chars[idx % 4]}", end='', flush=True)
        idx += 1
        try:
            threading.Event().wait(0.1)
        except KeyboardInterrupt:
            break
    print(f"\r  {msg} ✓", end='', flush=True)
    print()


def stop_spin():
    global SPINNING
    SPINNING = False


def get_modelo_dir(tipo_modelo, nro_modelo):
    return os.path.join(MODELOS_BASE, tipo_modelo, nro_modelo)


def cargar_modelo(tipo_modelo, nro_modelo):
    archivo = f'model_{tipo_modelo}_{nro_modelo}.joblib'
    ruta = os.path.join(get_modelo_dir(tipo_modelo, nro_modelo), archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el modelo '{archivo}' en '{get_modelo_dir(tipo_modelo, nro_modelo)}/'. "
            f"Ejecutá primero {tipo_modelo}.py para generar el modelo."
        )
    t = threading.Thread(target=spin, args=(f"Cargando modelo {tipo_modelo} #{nro_modelo}",))
    t.start()
    modelo = joblib.load(ruta)
    stop_spin()
    print(f"[INFO] Modelo cargado: {archivo}")
    return modelo


def cargar_vista(nro_vista):
    archivo = f'vista_{nro_vista}.csv'
    ruta = os.path.join(VISTAS_MINABLES_DIR, archivo)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró la vista '{archivo}' en '{VISTAS_MINABLES_DIR}/'."
        )
    t = threading.Thread(target=spin, args=(f"Cargando vista_{nro_vista}.csv",))
    t.start()
    df = pd.read_csv(ruta)
    stop_spin()
    print(f"[INFO] Vista cargada: {archivo} — {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def graficar_matriz_confusion(y_test, y_pred, nombre_modelo, nro_modelo, nro_vista, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['No Demorado', 'Demorado'],
        yticklabels=['No Demorado', 'Demorado'],
        linewidths=1, linecolor='white'
    )
    ax.set_xlabel('Predicción', fontweight='bold')
    ax.set_ylabel('Real', fontweight='bold')
    ax.set_title(f'Matriz de Confusión — {nombre_modelo.upper()} #{nro_modelo}', fontweight='bold', fontsize=14)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    archivo = f'prueba_{nombre_modelo}_{nro_modelo}_vista_{nro_vista}_confusion.png'
    plt.savefig(os.path.join(output_dir, archivo), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   [+] {archivo}")


def evaluar_modelo(modelo, X_train, X_test, y_train, y_test, nombre_modelo, nro_modelo, nro_vista):
    print("\n" + "=" * 70)
    print(f"EVALUACIÓN — {nombre_modelo.upper()} #{nro_modelo} sobre vista_{nro_vista}.csv")
    print("=" * 70)

    t = threading.Thread(target=spin, args=(f"Evaluando modelo {nombre_modelo} #{nro_modelo}",))
    t.start()
    y_pred = modelo.predict(X_test)

    try:
        y_proba = modelo.predict_proba(X_test)[:, 1]
        has_proba = True
    except AttributeError:
        y_proba = None
        has_proba = False
    stop_spin()

    metricas = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if has_proba:
        metricas['auc_roc'] = float(roc_auc_score(y_test, y_proba))

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, target_names=['No Demorado', 'Demorado'], output_dict=True
    )

    print(f"\n  {'Métrica':<15} {'Valor':<10}")
    print("  " + "-" * 27)
    for nombre, valor in metricas.items():
        print(f"  {nombre:<15} {valor:.4f}")

    print(f"\n  Classification Report:")
    print("  " + "-" * 55)
    report_str = classification_report(y_test, y_pred, target_names=['No Demorado', 'Demorado'])
    for linea in report_str.split('\n'):
        print(f"  {linea}")

    modelo_dir = get_modelo_dir(nombre_modelo, nro_modelo)

    if has_proba:
        t = threading.Thread(target=spin, args=("Generando curva ROC",))
        t.start()
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='#4C78A8', linewidth=2,
                label=f'{nombre_modelo.upper()} (AUC = {metricas["auc_roc"]:.4f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
                label='Random (AUC = 0.5)')
        ax.fill_between(fpr, tpr, alpha=0.1, color='#4C78A8')
        ax.set_xlabel('False Positive Rate (1 - Especificidad)')
        ax.set_ylabel('True Positive Rate (Sensibilidad)')
        ax.set_title(f'Curva ROC — {nombre_modelo.upper()} #{nro_modelo}', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        plt.tight_layout()
        archivo = f'prueba_{nombre_modelo}_{nro_modelo}_vista_{nro_vista}_roc.png'
        plt.savefig(os.path.join(modelo_dir, archivo), dpi=150, bbox_inches='tight')
        plt.close()
        stop_spin()
        print(f"\n   [+] {archivo}")

    t = threading.Thread(target=spin, args=("Generando matriz de confusión",))
    t.start()
    graficar_matriz_confusion(y_test, y_pred, nombre_modelo, nro_modelo, nro_vista, modelo_dir)
    stop_spin()

    return metricas, cm, report, y_pred, y_proba


def exportar_resultados_json(metricas, cm, nombre_modelo, nro_modelo, nro_vista, report):
    datos = {
        'modelo': nombre_modelo,
        'nro_modelo': nro_modelo,
        'nro_vista': nro_vista,
        'metricas': metricas,
        'matriz_confusion': cm.tolist(),
        'confusion_matrix_labels': ['No Demorado', 'Demorado'],
        'classification_report': report,
    }
    archivo = f'resultados_{nombre_modelo}_{nro_modelo}_vista_{nro_vista}.json'
    ruta = os.path.join(get_modelo_dir(nombre_modelo, nro_modelo), archivo)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    print(f"\n   [+] {archivo}")
    return ruta


def main():
    print("=" * 70)
    print("PRUEBAS — EVALUACIÓN DE MODELOS")
    print("=" * 70)

    print("\nModelos disponibles: knn, arbol")

    while True:
        tipo = input("\nIngresá el tipo de modelo (knn/arbol): ").strip().lower()
        if tipo in MODELOS_PERMITIDOS:
            break
        print("  Modelo no válido. Intentá de nuevo.")

    while True:
        nro_modelo = input("Ingresá el número de modelo (ej: 001): ").strip()
        if nro_modelo:
            break
        print("  Entrada inválida.")

    while True:
        nro_vista = input("Ingresá el número de vista (ej: 001): ").strip()
        if nro_vista:
            break
        print("  Entrada inválida.")

    modelo = cargar_modelo(tipo, nro_modelo)
    df = cargar_vista(nro_vista)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print(f"\n  Features (X): {X.shape[1]} columnas")
    print(f"  Target (y): '{TARGET}' → 0={y.eq(0).sum()} | 1={y.eq(1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")

    metricas, cm, report, y_pred, y_proba = evaluar_modelo(
        modelo, X_train, X_test, y_train, y_test, tipo, nro_modelo, nro_vista
    )

    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE RESULTADOS")
    print("=" * 70)
    exportar_resultados_json(metricas, cm, tipo, nro_modelo, nro_vista, report)

    modelo_dir = get_modelo_dir(tipo, nro_modelo)
    print("\n" + "=" * 70)
    print("PRUEBA COMPLETA")
    print("=" * 70)
    print(f"\nGráficos y resultados en: {modelo_dir}/")
    print(f"  • prueba_{tipo}_{nro_modelo}_vista_{nro_vista}_confusion.png")
    if y_proba is not None:
        print(f"  • prueba_{tipo}_{nro_modelo}_vista_{nro_vista}_roc.png")
    print(f"  • resultados_{tipo}_{nro_modelo}_vista_{nro_vista}.json")


if __name__ == "__main__":
    main()
