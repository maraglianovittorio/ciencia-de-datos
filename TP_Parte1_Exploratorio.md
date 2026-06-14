# Trabajo Práctico: K-Medias para Segmentación de Clientes (Parte 1)

Este informe detalla exhaustivamente el proceso técnico llevado a cabo para la limpieza, preprocesamiento y entrenamiento de un modelo de *Machine Learning* no supervisado (K-Medias) sobre un dataset de 50.000 clientes de una aerolínea / agencia de turismo.

---

## 1. Análisis Exploratorio de Datos (EDA)

El dataset inicial consta de **50.000 registros** e incluye tanto variables demográficas como información comportamental y financiera. 

### 1.1 Análisis Univariante

#### Variables Numéricas
Se estudiaron 6 variables cuantitativas principales: `edad`, `cant_vuelos`, `gasto_acumulado`, `cantidad_millas`, `ingreso_mensual` y `anticipacion_compra_promedio`.
- Al generar boxplots y calcular medidas de dispersión, se observó un patrón claro en las variables financieras (`gasto_acumulado`, `ingreso_mensual`) y de fidelidad (`cantidad_millas`): todas presentaban **asimetría positiva severa (cola a la derecha)**.
- Un grupo reducido de clientes (aproximadamente el 10%) concentraba métricas de vuelo y gasto que superaban hasta en 5 veces la media poblacional. Matemáticamente, estos valores actúan como *outliers* pesados que distorsionan el centro de gravedad geométrico.

#### Variables Categóricas
El dataset contiene variables cualitativas como `sexo`, `provincia`, `ocupacion`, `clase_preferida`, `programaMillas` y `canal_compra`. 
> [!IMPORTANT]
> **Decisión Arquitectónica:** Se decidió descartar completamente las variables categóricas del entrenamiento del modelo. ¿Por qué? K-Medias utiliza la **distancia euclidiana**. Aplicar *One-Hot Encoding* sobre variables categóricas para convertirlas en 0 y 1 carece de sentido geométrico cuando se mezclan con distancias continuas masivas (ej: 40.000 millas). Su inclusión hubiera deformado la forma esférica de los clusters.

### 1.2 Análisis Multivariante
Se calculó la Matriz de Correlación de Pearson para todas las variables numéricas.
- **Hallazgo Crítico:** Se detectó una altísima correlación lineal (**0.91**) entre `gasto_acumulado` y `gasto_acumulado_extra`.
- **Resolución:** Esta **multicolinealidad** es tóxica para K-Medias porque el modelo no entiende de conceptos, sino de distancias. Dejar ambas variables hubiera provocado que el "peso del dinero" valga el doble frente a variables como "cantidad de vuelos". Se procedió a eliminar `gasto_acumulado_extra`.

---

## 2. Fase de Preparación y Limpieza de Datos

### 2.1 Proceso de Limpieza (Tratamiento de Outliers)
En base a los sesgos detectados en el análisis univariante, se procedió a limpiar la base de datos dividiéndola empíricamente en dos partes.
- Se calculó el **Percentil 95 (P95)** para las variables financieras crudas. Todo cliente que superara el P95 en gasto, millas o ingreso fue catalogado como una **"Ballena" o Segmento VIP**.
- **Resultados:** De los 50.000 clientes, **5.180 (10.3%)** fueron etiquetados como VIPs y apartados. 

Esta limpieza garantizó que el algoritmo K-Medias operara sobre una base de 44.820 clientes "terrenales", asegurando que los centroides se ubiquen en el núcleo real de la población en lugar de ser arrastrados infinitamente hacia la derecha por los millonarios.

### 2.2 Especificación de Transformaciones (Normalización)
Para que las distancias geométricas sean equitativas, se deben escalar los datos. Un cliente no puede ser "empujado" de cluster sólo porque el ingreso se mide en miles (pesos) y la edad en decenas (años).
- Se aplicó un **StandardScaler (Z-Score)** a las 6 variables numéricas de los clientes normales. Esto centró todas las variables en media 0 y desviación estándar 1, homologando las escalas dimensionales.

---

## 3. Especificación de la Vista Minable

Como resultado de todo el proceso anterior, se construyeron las siguientes **Vistas Minables**:

1. `clientes_premium.csv`: Vista exclusiva conteniendo la data original (no escalada) de los 5.180 clientes VIP aislados por el Percentil 95.
2. `vista_clientes_kmeans.csv`: Vista conteniendo los 44.820 clientes estándar, sin la variable colineal `gasto_acumulado_extra`, y sin el `id_cliente`.
3. **`vista_clientes_kmeans_normalizada.csv` (Target del Cluster)**: Esta fue la vista minable oficial que consumió el modelo. Contenía exclusivamente a los 44.820 clientes estándar, con sus 6 variables numéricas transformadas bajo Z-Score y libre de cualquier ruido categórico.

*(Continúa en la Parte 2: Entrenamiento, Gráficos PCA y Conclusiones)*
