"""
================================================================================
ÁRBOLES — Clasificador de Demoras Aéreas
================================================================================
Consume una vista minable y entrena un modelo de Árbol de Decisión para
predecir demoras.

Incluye:
  - Búsqueda de hiperparámetros con GridSearchCV (validación cruzada)
  - Evaluación completa (accuracy, precision, recall, F1, AUC-ROC)
  - Matriz de confusión y curva ROC
  - Exportación del modelo en .joblib + .json + .md
================================================================================
"""

import os
import json
import glob
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
)
from imblearn.under_sampling import RandomUnderSampler
import joblib

VISTA_MINABLE_DIR = 'vistas_minables'
VISTAS_PATTERN = os.path.join(VISTA_MINABLE_DIR, 'vista_*.csv')
MODELOS_BASE = 'models'
TARGET = 'demora'
TEST_SIZE = 0.3
RANDOM_STATE = 42

OUTPUT_DIR = 'TPI/graficos'
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def listar_vistas():
    archivos = glob.glob(VISTAS_PATTERN)
    vistas = []
    for f in archivos:
        basename = os.path.basename(f)
        if basename.startswith('vista_') and basename.endswith('.csv'):
            vistas.append(basename)
    return sorted(vistas)


def seleccionar_vista():
    vistas = listar_vistas()
    if not vistas:
        raise FileNotFoundError(
            f"No se encontró ninguna vista minable en '{VISTA_MINABLE_DIR}/'. "
            "Ejecutá primero preprocesamiento.py para generar una vista."
        )

    print("\n" + "=" * 70)
    print("SELECCIÓN DE VISTA MINABLE")
    print("=" * 70)
    print("\nVistas disponibles:")
    for i, v in enumerate(vistas, 1):
        print(f"  {i}. {v}")

    while True:
        entrada = input("\nIngresá el número de vista a utilizar: ").strip()
        try:
            idx = int(entrada) - 1
            if 0 <= idx < len(vistas):
                break
            print("  Número fuera de rango. Intentá de nuevo.")
        except ValueError:
            print("  Entrada inválida. Ingresá un número.")

    vista_seleccionada = vistas[idx]
    nro_vista = vista_seleccionada.replace('vista_', '').replace('.csv', '')
    ruta_vista = os.path.join(VISTA_MINABLE_DIR, vista_seleccionada)
    return ruta_vista, nro_vista


def cargar_vista_minable(ruta):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró '{ruta}'. "
            "Ejecutá primero preprocesamiento.py para generar la vista minable."
        )
    df = pd.read_csv(ruta)
    print(f"[INFO] Vista minable cargada: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def balancear_dataset(X, y):
    ros = RandomUnderSampler(random_state=RANDOM_STATE)
    X_res, y_res = ros.fit_resample(X, y)
    print(f"[INFO] Dataset balanceado: {len(X_res)} muestras "
          f"({y_res.sum()} demorados / {(y_res == 0).sum()} no demorados)")
    return X_res, y_res


def separar_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"\n  Features (X): {X.shape[1]} columnas")
    print(f"  Target (y): '{TARGET}' → 0={y.eq(0).sum()} | 1={y.eq(1).sum()}")
    print(f"  Balance: {y.mean()*100:.1f}% demorados")
    return X, y


def buscar_mejores_hiperparametros(X_train, y_train):
    print("\n" + "=" * 70)
    print("BÚSQUEDA DE HIPERPARÁMETROS")
    print("=" * 70)
    print()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
    }

    base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)

    t = threading.Thread(target=spin, args=("Buscando mejores hiperparámetros",))
    t.start()

    grid = GridSearchCV(
        base_tree,
        param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_train, y_train)

    stop_spin()

    mejores = grid.best_params_
    mejor_score = grid.best_score_

    print(f"\n  Mejores hiperparámetros encontrados:")
    for k, v in mejores.items():
        print(f"    • {k}: {v}")
    print(f"\n  Accuracy CV óptimo: {mejor_score:.4f}")

    return grid.best_estimator_, mejores, mejor_score


def entrenar_y_evaluar(X_train, X_test, y_train, y_test, modelo):
    print("\n" + "=" * 70)
    print("ENTRENAMIENTO Y EVALUACIÓN")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Entrenando modelo",))
    t.start()
    modelo.fit(X_train, y_train)
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

    return modelo, y_pred, y_proba, metricas, cm, report


def graficar_matriz_confusion(y_test, y_pred, output_dir=OUTPUT_DIR):
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
    ax.set_title('Matriz de Confusión — Árbol de Decisión', fontweight='bold', fontsize=14)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arbol_confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] arbol_confusion_matrix.png")


def graficar_curva_roc(y_test, y_proba, auc, output_dir=OUTPUT_DIR):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#4C78A8', linewidth=2,
            label=f'Árbol (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4C78A8')
    ax.set_xlabel('False Positive Rate (1 - Especificidad)')
    ax.set_ylabel('True Positive Rate (Sensibilidad)')
    ax.set_title('Curva ROC — Árbol de Decisión', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'arbol_curva_roc.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] arbol_curva_roc.png")


def imprimir_conclusiones(metricas, mejores, n_features, n_muestras):
    print("\n" + "=" * 70)
    print("CONCLUSIONES DEL MODELO ÁRBOL DE DECISIÓN")
    print("=" * 70)

    acc = metricas['accuracy']
    prec = metricas['precision']
    rec = metricas['recall']
    f1 = metricas['f1']
    auc = metricas.get('auc_roc', 0)

    print(f"\n  1. CONFIGURACIÓN")
    print(f"     • Modelo: Árbol de Decisión")
    for k, v in mejores.items():
        print(f"     • {k}: {v}")
    print(f"     • Features: {n_features}")
    print(f"     • Muestras: {n_muestras} ({int(n_muestras*0.7)} train / {int(n_muestras*0.3)} test)")

    print(f"\n  2. RENDIMIENTO GENERAL")
    if acc >= 0.85:
        print(f"     ★ Accuracy de {acc:.1%}: clasifica correctamente la mayoría.")
    elif acc >= 0.70:
        print(f"     • Accuracy de {acc:.1%}: aceptable con margen de mejora.")
    else:
        print(f"     ⚠ Accuracy de {acc:.1%}: rendimiento bajo.")

    print(f"\n  3. CAPACIDAD DE DETECCIÓN DE DEMORAS")
    if rec >= 0.70:
        print(f"     ★ Recall de {rec:.1%}: buena detección de demoras.")
    elif rec >= 0.50:
        print(f"     • Recall de {rec:.1%}: detecta la mitad de las demoras.")
    else:
        print(f"     ⚠ Recall de {rec:.1%}: dificultades para detectar demoras.")

    print(f"\n  4. PRECISIÓN EN ALERTAS")
    if prec >= 0.70:
        print(f"     ★ Precision de {prec:.1%}: cuando alerta, generalmente acierta.")
    elif prec >= 0.50:
        print(f"     • Precision de {prec:.1%}: falsas alarmas frecuentes.")
    else:
        print(f"     ⚠ Precision de {prec:.1%}: la mayoría de las alertas son falsas.")

    print(f"\n  5. BALANCE PRECISION-RECALL")
    print(f"     • F1-Score: {f1:.4f}")
    if abs(prec - rec) > 0.15:
        if prec > rec:
            print(f"       Modelo conservador: alerta con precisión pero deja pasar demoras.")
        else:
            print(f"       Modelo agresivo: detecta más demoras pero con falsas alarmas.")
    else:
        print(f"       Balance equilibrado entre precision y recall.")

    if auc > 0:
        print(f"\n  6. DISCRIMINACIÓN (AUC-ROC)")
        if auc >= 0.80:
            print(f"     ★ AUC-ROC de {auc:.4f}: excelente discriminación.")
        elif auc >= 0.65:
            print(f"     • AUC-ROC de {auc:.4f}: discriminación aceptable.")
        else:
            print(f"     ⚠ AUC-ROC de {auc:.4f}: apenas supera al azar.")


def exportar_modelo(modelo, mejores, nro_vista, modelo_dir):
    ruta = os.path.join(modelo_dir, f'model_arbol_{nro_vista}.joblib')
    joblib.dump(modelo, ruta)
    print(f"\n   [+] model_arbol_{nro_vista}.joblib")
    joblib.dump(mejores, os.path.join(modelo_dir, f'params_arbol_{nro_vista}.joblib'))
    return ruta


def exportar_json(metricas, cm, nro_vista, mejores, n_features, n_muestras, modelo_dir):
    datos = {
        'modelo': 'arbol',
        'nro_vista': nro_vista,
        'hiperparametros': mejores,
        'n_features': n_features,
        'n_muestras': n_muestras,
        'metricas': metricas,
        'matriz_confusion': cm.tolist(),
        'confusion_matrix_labels': ['No Demorado', 'Demorado'],
    }
    ruta = os.path.join(modelo_dir, f'model_arbol_{nro_vista}.json')
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    print(f"   [+] model_arbol_{nro_vista}.json")
    return ruta


def exportar_markdown(nro_vista, mejores, metricas, modelo_dir):
    auc_str = f"{metricas.get('auc_roc', 0):.4f}"
    contenido = f"""# Modelo Árbol de Decisión — {nro_vista}

## Información del Modelo

- **Vista Minable:** vista_{nro_vista}.csv
- **Modelo:** Árbol de Decisión
- **Hiperparámetros:**

| Hiperparámetro | Valor |
|----------------|-------|
| max_depth | {mejores.get('max_depth', 'None')} |
| min_samples_split | {mejores.get('min_samples_split', '-')} |
| min_samples_leaf | {mejores.get('min_samples_leaf', '-')} |
| criterion | {mejores.get('criterion', '-')} |

## Métricas

| Métrica   | Valor   |
|-----------|---------|
| Accuracy  | {metricas['accuracy']:.4f} |
| Precision | {metricas['precision']:.4f} |
| Recall    | {metricas['recall']:.4f} |
| F1        | {metricas['f1']:.4f} |
| AUC-ROC   | {auc_str} |

## Notas y Anotaciones

[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]
"""
    ruta = os.path.join(modelo_dir, f'model_arbol_{nro_vista}.md')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(contenido)
    print(f"   [+] model_arbol_{nro_vista}.md")
    return ruta


def main():
    print("=" * 70)
    print("ÁRBOL DE DECISIÓN — CLASIFICADOR DE DEMORAS AÉREAS")
    print("=" * 70)

    ruta_vista, nro_vista = seleccionar_vista()

    modelo_dir = os.path.join(MODELOS_BASE, f'arbol/{nro_vista}')
    os.makedirs(modelo_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Cargando vista minable",))
    t.start()
    df = cargar_vista_minable(ruta_vista)
    stop_spin()

    X, y = separar_features_target(df)
    X, y = balancear_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")

    modelo, mejores, mejor_score = buscar_mejores_hiperparametros(X_train, y_train)

    modelo, y_pred, y_proba, metricas, cm, report = entrenar_y_evaluar(
        X_train, X_test, y_train, y_test, modelo
    )

    print("\n" + "=" * 70)
    print("VISUALIZACIONES")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Generando matriz de confusión",))
    t.start()
    graficar_matriz_confusion(y_test, y_pred, output_dir=modelo_dir)
    stop_spin()

    if y_proba is not None:
        t = threading.Thread(target=spin, args=("Generando curva ROC",))
        t.start()
        graficar_curva_roc(y_test, y_proba, metricas['auc_roc'], output_dir=modelo_dir)
        stop_spin()

    imprimir_conclusiones(metricas, mejores, X.shape[1], len(df))

    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE MODELO")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Guardando modelo .joblib",))
    t.start()
    exportar_modelo(modelo, mejores, nro_vista, modelo_dir)
    stop_spin()

    t = threading.Thread(target=spin, args=("Guardando métricas .json",))
    t.start()
    exportar_json(metricas, cm, nro_vista, mejores, X.shape[1], len(df), modelo_dir)
    stop_spin()

    t = threading.Thread(target=spin, args=("Generando documentación .md",))
    t.start()
    exportar_markdown(nro_vista, mejores, metricas, modelo_dir)
    stop_spin()

    print("\n" + "=" * 70)
    print("PROCESO COMPLETO")
    print("=" * 70)
    print(f"\nGráficos del modelo en: {modelo_dir}/")
    print(f"  • arbol_confusion_matrix.png")
    if y_proba is not None:
        print(f"  • arbol_curva_roc.png")
    print(f"\nModelo y documentación en: {modelo_dir}/")
    print(f"  • model_arbol_{nro_vista}.joblib")
    print(f"  • model_arbol_{nro_vista}.json")
    print(f"  • model_arbol_{nro_vista}.md")


if __name__ == "__main__":
    main()
