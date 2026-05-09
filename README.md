# TP Integrador 2026 — DS Airlines: Gestión Proactiva de Demoras Aéreas

## Entorno

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Estructura del Proyecto

```
.
├── Vuelos.xlsx                    # Dataset bruto
├── preprocesamiento.py            # Paso 1: genera vistas minables
├── knn.py                         # Paso 2: entrena modelo KNN
├── arboles.py                     # Paso 2: entrena modelo Árbol de Decisión
├── pruebas.py                     # Paso 3: evaluar modelos cargados
├── utils.py                       # Funciones compartidas (carga, clasificación)
├── requirements.txt               # Dependencias
├── entendimiento/
│   ├── main.py                    # Ejecuta todo el análisis exploratorio
│   ├── limpieza.py                # Detección de nulos, duplicados, outliers
│   └── analisis.py                # Gráficos y relaciones entre variables
├── vistas_minables/               # Salida de preprocesamiento.py
│   ├── vista_001.csv
│   └── vista_001.md
├── models/                        # Salida de knn.py y arboles.py
│   ├── knn/001/
│   └── arbol/001/
└── TPI/graficos/                  # Gráficos generados automáticamente
```

## pipeline de trabajo

### 1. Entendimiento de los datos (`entendimiento/main.py`)

Ejecuta análisis exploratorio completo sobre el dataset bruto. Muestra:
- Descripción general del dataset (tipos, nulos, únicos)
- Estadísticos descriptivos
- Detección de nulos, duplicados y outliers
- Análisis de relaciones entre variables (visibilidad vs demora, clima vs demora, etc.)
- Balance de clases de la variable objetivo
- Matriz de correlación

Todos los gráficos se guardan en `TPI/graficos/`.

```bash
python entendimiento/main.py
```

---

### 2. Preprocesamiento (`preprocesamiento.py`)

Toma `Vuelos.xlsx`, aplica limpieza y transformación, y genera una **vista minable** en `vistas_minables/`. Es el paso obligatorio previo a entrenar cualquier modelo.

**Qué hace:**
- Filtros custom: función `filtrado_custom(df)` con lógica adaptable. Por defecto elimina vuelos con velocidad calculada < 100 km/h y elimina EZE↔AEP. Estos filtros son **editables** según lo que quieras probar (ej: distinto umbral de velocidad, eliminar otras rutas, etc.).
- Elimina columnas irrelevantes según la lista `EXCLUIR` (ej: id_vuelo, hora_salida_programada, aeropuerto_origen, aeropuerto_destino). Agregar o quitar campos ahí es suffisant para cambiar el dataset.
- Elimina columnas irrelevantes para modelado (id_vuelo, hora_salida_programada, etc.)
- Elimina duplicados
- Imputa nulos: mediana para numéricas, moda para categóricas
- Normaliza numéricas con Min-Max (0-1), excepto la variable objetivo
- Aplica One-Hot Encoding a categóricas (con `drop_first=True`)
- Exporta `vista_XXX.csv` + `vista_XXX.md` con documentación del proceso

```bash
python preprocesamiento.py
```

Si ya existen vistas, genera la siguiente numeración automáticamente (`vista_002.csv`, etc.).

---

### 3. Modelado

#### KNN (`knn.py`)

Clasificador K-Nearest Neighbors. Busca el K óptimo por validación cruzada (5-fold StratifiedKFold), entrena y evalúa con métricas completas.

**Qué hace:**
- Solicita seleccionar una vista minable
- Busca el mejor K en rango 1-30 por accuracy en CV
- Entrena con el K óptimo
- Evalúa: accuracy, precision, recall, F1, AUC-ROC
- Genera matriz de confusión y curva ROC
- Exporta modelo en `.joblib` + `.json` + `.md` en `models/knn/XXX/`

```bash
python knn.py
```

#### Árbol de Decisión (`arboles.py`)

Clasificador Decision Tree con GridSearchCV. Busca hiperparámetros óptimos (max_depth, min_samples_split, etc.).

**Qué hace:**
- Solicita seleccionar una vista minable
- GridSearchCV con validación cruzada (5-fold)
- Entrena con mejores hiperparámetros
- Evalúa: accuracy, precision, recall, F1, AUC-ROC (si hay probabilidades)
- Genera matriz de confusión y curva ROC
- Exporta modelo + hiperparámetros en `.joblib` + `.json` + `.md` en `models/arbol/XXX/`

```bash
python arboles.py
```

---

### 4. Pruebas (`pruebas.py`)

Permite evaluar un modelo ya guardado (de knn o arbol) sobre una vista minable, sin reentrenar. Útil para comparar rendimiento entre modelos y vistas.

```bash
python pruebas.py
```

---

## Archivos principales

| Archivo | Qué hace |
|---------|----------|
| `preprocesamiento.py` | Limpia y transforma los datos, genera vistas minables |
| `knn.py` | Entrena modelo KNN con búsqueda de K óptimo |
| `arboles.py` | Entrena modelo Árbol con GridSearchCV |
| `pruebas.py` | Evalúa modelos ya entrenados |
| `utils.py` | Funciones compartidas: carga de datos, clasificación de variables |
| `entendimiento/main.py` | Análisis exploratorio completo |
| `entendimiento/limpieza.py` | Detección de nulos, duplicados, outliers |
| `entendimiento/analisis.py` | Análisis de relaciones (clima, hora, aeropuerto, etc.) |

## Flujo de ejecución

```
Vuelos.xlsx
    │
    ▼
entendimiento/main.py     ← análisis exploratorio (opcional, para entender los datos)
    │
    ▼
preprocesamiento.py        ← genera vistas_minables/vista_XXX.csv
    │
    ▼
knn.py  ──→  models/knn/XXX/
arboles.py  ──→  models/arbol/XXX/
    │
    ▼
pruebas.py                 ← evaluar modelos guardados
```

## Notas

- Los modelos consumen **exclusivamente** vistas generadas por `preprocesamiento.py`.
- Cada ejecución de preprocesamiento genera una vista numerada secuencialmente.
- Para cambiar la vista usada por knn.py o arboles.py, se selecciona interactivamente al ejecutar.