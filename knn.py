"""
================================================================================
KNN — Clasificador de Demoras Aéreas
================================================================================
Consume una vista minable y entrena un modelo K-Nearest Neighbors para
predecir demoras.

Incluye:
  - Búsqueda del K óptimo por validación cruzada
  - Evaluación completa (accuracy, precision, recall, F1, AUC-ROC)
  - Matriz de confusión y curva ROC
  - Conclusiones automáticas basadas en los resultados
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

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
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
import joblib

VISTA_MINABLE_DIR = 'vistas_minables'
VISTAS_PATTERN = os.path.join(VISTA_MINABLE_DIR, 'vista_*.csv')
MODELOS_BASE = 'models'
TARGET = 'demora'
TEST_SIZE = 0.3
RANDOM_STATE = 42
K_RANGE = range(1, 31)

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


def separar_features_target(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"\n  Features (X): {X.shape[1]} columnas")
    print(f"  Target (y): '{TARGET}' → 0={y.eq(0).sum()} | 1={y.eq(1).sum()}")
    print(f"  Balance: {y.mean()*100:.1f}% demorados")
    return X, y


def buscar_k_optimo(X_train, y_train, k_range=K_RANGE):
    print("\n" + "=" * 70)
    print("BÚSQUEDA DEL K ÓPTIMO")
    print("=" * 70)
    print()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    resultados = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='accuracy')
        resultados.append({
            'k': k,
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std(),
        })

    stop_spin()

    df_resultados = pd.DataFrame(resultados)
    mejor = df_resultados.loc[df_resultados['accuracy_mean'].idxmax()]
    k_optimo = int(mejor['k'])

    print(f"\n  {'K':<5} {'Accuracy (CV)':<18} {'± Std':<10}")
    print("  " + "-" * 35)
    for _, row in df_resultados.iterrows():
        marca = " ◀ MEJOR" if int(row['k']) == k_optimo else ""
        print(f"  {int(row['k']):<5} {row['accuracy_mean']:.4f}           "
              f"± {row['accuracy_std']:.4f}{marca}")

    print(f"\n  ★ K óptimo: {k_optimo} "
          f"(accuracy CV: {mejor['accuracy_mean']:.4f} ± {mejor['accuracy_std']:.4f})")

    t = threading.Thread(target=spin, args=("Generando gráfico K óptimo",))
    t.start()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_resultados['k'], df_resultados['accuracy_mean'],
            marker='o', color='#4C78A8', linewidth=2, markersize=5)
    ax.fill_between(
        df_resultados['k'],
        df_resultados['accuracy_mean'] - df_resultados['accuracy_std'],
        df_resultados['accuracy_mean'] + df_resultados['accuracy_std'],
        alpha=0.2, color='#4C78A8'
    )
    ax.axvline(k_optimo, color='#E45756', linestyle='--', linewidth=1.5,
               label=f'K óptimo = {k_optimo}')
    ax.set_xlabel('K (número de vecinos)')
    ax.set_ylabel('Accuracy (validación cruzada)')
    ax.set_title('Búsqueda del K óptimo para KNN', fontweight='bold')
    ax.legend()
    ax.set_xticks(list(k_range))
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/knn_k_optimo.png', dpi=150, bbox_inches='tight')
    plt.close()
    stop_spin()
    print(f"   [+] knn_k_optimo.png")

    return k_optimo, df_resultados


def entrenar_y_evaluar(X_train, X_test, y_train, y_test, k):
    print("\n" + "=" * 70)
    print(f"ENTRENAMIENTO Y EVALUACIÓN (K={k})")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Entrenando modelo",))
    t.start()
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    stop_spin()

    metricas = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba)),
    }

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
    ax.set_title('Matriz de Confusión — KNN', fontweight='bold', fontsize=14)
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j + 0.5, i + 0.7, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/knn_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] knn_confusion_matrix.png")


def graficar_curva_roc(y_test, y_proba, auc, output_dir=OUTPUT_DIR):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#4C78A8', linewidth=2,
            label=f'KNN (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1,
            label='Random (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#4C78A8')
    ax.set_xlabel('False Positive Rate (1 - Especificidad)')
    ax.set_ylabel('True Positive Rate (Sensibilidad)')
    ax.set_title('Curva ROC — KNN', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/knn_curva_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   [+] knn_curva_roc.png")


def imprimir_conclusiones(metricas, k_optimo, n_features, n_muestras):
    print("\n" + "=" * 70)
    print("CONCLUSIONES DEL MODELO KNN")
    print("=" * 70)

    acc = metricas['accuracy']
    prec = metricas['precision']
    rec = metricas['recall']
    f1 = metricas['f1']
    auc = metricas['auc_roc']

    print(f"\n  1. CONFIGURACIÓN")
    print(f"     • Modelo: K-Nearest Neighbors (K={k_optimo})")
    print(f"     • Features: {n_features} (normalizadas + one-hot encoded)")
    print(f"     • Muestras: {n_muestras} ({int(n_muestras*0.7)} train / "
          f"{int(n_muestras*0.3)} test)")

    print(f"\n  2. RENDIMIENTO GENERAL")
    if acc >= 0.85:
        print(f"     ★ Accuracy de {acc:.1%}: el modelo clasifica correctamente")
        print(f"       la gran mayoría de los vuelos.")
    elif acc >= 0.70:
        print(f"     • Accuracy de {acc:.1%}: rendimiento aceptable, pero con")
        print(f"       margen de mejora significativo.")
    else:
        print(f"     ⚠ Accuracy de {acc:.1%}: rendimiento bajo. El modelo tiene")
        print(f"       dificultades para distinguir las clases.")

    print(f"\n  3. CAPACIDAD DE DETECCIÓN DE DEMORAS")
    if rec >= 0.70:
        print(f"     ★ Recall de {rec:.1%}: buena capacidad para detectar vuelos")
        print(f"       que realmente van a demorarse.")
    elif rec >= 0.50:
        print(f"     • Recall de {rec:.1%}: detecta la mitad de las demoras reales.")
        print(f"       Hay falsos negativos (demoras no detectadas).")
    else:
        print(f"     ⚠ Recall de {rec:.1%}: el modelo falla en detectar demoras.")
        print(f"       Muchos vuelos demorados se clasifican como no demorados.")

    print(f"\n  4. PRECISIÓN EN ALERTAS")
    if prec >= 0.70:
        print(f"     ★ Precision de {prec:.1%}: cuando el modelo predice demora,")
        print(f"       generalmente acierta.")
    elif prec >= 0.50:
        print(f"     • Precision de {prec:.1%}: hay falsas alarmas frecuentes.")
    else:
        print(f"     ⚠ Precision de {prec:.1%}: la mayoría de las alertas de")
        print(f"       demora son falsas alarmas.")

    print(f"\n  5. BALANCE PRECISION-RECALL")
    print(f"     • F1-Score: {f1:.4f}")
    if abs(prec - rec) > 0.15:
        if prec > rec:
            print(f"       El modelo es más conservador: cuando alerta, acierta,")
            print(f"       pero deja pasar demoras sin detectar.")
        else:
            print(f"       El modelo es más agresivo: detecta muchas demoras")
            print(f"       pero también genera falsas alarmas.")
    else:
        print(f"       Balance equilibrado entre precision y recall.")

    print(f"\n  6. DISCRIMINACIÓN (AUC-ROC)")
    if auc >= 0.80:
        print(f"     ★ AUC-ROC de {auc:.4f}: excelente capacidad de discriminación")
        print(f"       entre vuelos demorados y no demorados.")
    elif auc >= 0.65:
        print(f"     • AUC-ROC de {auc:.4f}: capacidad de discriminación aceptable.")
    else:
        print(f"     ⚠ AUC-ROC de {auc:.4f}: el modelo apenas supera al azar")
        print(f"       en su capacidad de discriminar clases.")

    print(f"\n  7. RECOMENDACIONES")
    recomendaciones = []
    if rec < 0.60:
        recomendaciones.append(
            "Considerar técnicas de balanceo de clases (SMOTE, undersampling)"
            " para mejorar la detección de demoras."
        )
    if acc < 0.75:
        recomendaciones.append(
            "Evaluar otros algoritmos (Random Forest, Gradient Boosting)"
            " que podrían capturar mejor las relaciones no lineales."
        )
    if n_features > 30:
        recomendaciones.append(
            "Con muchas features dummy, considerar selección de features"
            " o PCA para reducir dimensionalidad."
        )
    recomendaciones.append(
        "Validar el modelo con nuevos datos (holdout temporal)"
        " para confirmar generalización."
    )
    for i, rec_texto in enumerate(recomendaciones, 1):
        print(f"     {i}. {rec_texto}")


def exportar_modelo(modelo, k_optimo, nro_vista, modelo_dir):
    ruta = os.path.join(modelo_dir, f'model_knn_{nro_vista}.joblib')
    joblib.dump(modelo, ruta)
    print(f"\n   [+] model_knn_{nro_vista}.joblib")
    return ruta


def exportar_json(metricas, cm, nro_vista, k_optimo, n_features, n_muestras, modelo_dir):
    datos = {
        'modelo': 'knn',
        'nro_vista': nro_vista,
        'k_optimo': k_optimo,
        'n_features': n_features,
        'n_muestras': n_muestras,
        'metricas': metricas,
        'matriz_confusion': cm.tolist(),
        'confusion_matrix_labels': ['No Demorado', 'Demorado'],
    }
    ruta = os.path.join(modelo_dir, f'model_knn_{nro_vista}.json')
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)
    print(f"   [+] model_knn_{nro_vista}.json")
    return ruta


def exportar_markdown(nro_vista, k_optimo, metricas, modelo_dir):
    contenido = f"""# Modelo KNN — {nro_vista}

## Información del Modelo

- **Vista Minable:** vista_{nro_vista}.csv
- **K óptimo:** {k_optimo}
- **Modelo:** K-Nearest Neighbors

## Métricas

| Métrica   | Valor   |
|-----------|---------|
| Accuracy  | {metricas['accuracy']:.4f} |
| Precision | {metricas['precision']:.4f} |
| Recall    | {metricas['recall']:.4f} |
| F1        | {metricas['f1']:.4f} |
| AUC-ROC   | {metricas['auc_roc']:.4f} |

## Notas y Anotaciones

[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]
"""
    ruta = os.path.join(modelo_dir, f'model_knn_{nro_vista}.md')
    with open(ruta, 'w', encoding='utf-8') as f:
        f.write(contenido)
    print(f"   [+] model_knn_{nro_vista}.md")
    return ruta


def main():
    global SPINNING
    print("=" * 70)
    print("KNN — CLASIFICADOR DE DEMORAS AÉREAS")
    print("=" * 70)

    ruta_vista, nro_vista = seleccionar_vista()

    modelo_dir = os.path.join(MODELOS_BASE, f'knn/{nro_vista}')
    os.makedirs(modelo_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PREPROCESAMIENTO")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Cargando vista minable",))
    t.start()
    df = cargar_vista_minable(ruta_vista)
    stop_spin()

    X, y = separar_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} muestras")
    print(f"  Test:  {X_test.shape[0]} muestras")

    t = threading.Thread(target=spin, args=("Buscando K óptimo (30 iteraciones)",))
    t.start()
    k_optimo, _ = buscar_k_optimo(X_train, y_train)

    modelo, y_pred, y_proba, metricas, cm, report = entrenar_y_evaluar(
        X_train, X_test, y_train, y_test, k_optimo
    )

    print("\n" + "=" * 70)
    print("VISUALIZACIONES")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Generando matriz de confusión",))
    t.start()
    graficar_matriz_confusion(y_test, y_pred, output_dir=modelo_dir)
    stop_spin()

    t = threading.Thread(target=spin, args=("Generando curva ROC",))
    t.start()
    graficar_curva_roc(y_test, y_proba, metricas['auc_roc'], output_dir=modelo_dir)
    stop_spin()

    imprimir_conclusiones(metricas, k_optimo, X.shape[1], len(df))

    print("\n" + "=" * 70)
    print("EXPORTACIÓN DE MODELO")
    print("=" * 70)

    t = threading.Thread(target=spin, args=("Guardando modelo .joblib",))
    t.start()
    exportar_modelo(modelo, k_optimo, nro_vista, modelo_dir)
    stop_spin()

    t = threading.Thread(target=spin, args=("Guardando métricas .json",))
    t.start()
    exportar_json(metricas, cm, nro_vista, k_optimo, X.shape[1], len(df), modelo_dir)
    stop_spin()

    t = threading.Thread(target=spin, args=("Generando documentación .md",))
    t.start()
    exportar_markdown(nro_vista, k_optimo, metricas, modelo_dir)
    stop_spin()

    print("\n" + "=" * 70)
    print("PROCESO COMPLETO")
    print("=" * 70)
    print(f"\nGráficos del modelo en: {modelo_dir}/")
    print(f"  • knn_confusion_matrix.png")
    print(f"  • knn_curva_roc.png")
    print(f"\nModelo y documentación en: {modelo_dir}/")
    print(f"  • model_knn_{nro_vista}.joblib")
    print(f"  • model_knn_{nro_vista}.json")
    print(f"  • model_knn_{nro_vista}.md")
    print(f"\nGráfico general en: {OUTPUT_DIR}/")
    print(f"  • knn_k_optimo.png")


if __name__ == "__main__":
    main()
