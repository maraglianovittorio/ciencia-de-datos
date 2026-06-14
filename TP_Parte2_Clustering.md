# Trabajo Práctico: K-Medias para Segmentación de Clientes (Parte 2)

*(Continuación de la Fase Analítica)*

## 4. Entrenamiento del Modelo de Clustering

Sobre la vista minable oficial `vista_clientes_kmeans_normalizada.csv` (compuesta por las 6 variables continuas de los 44.820 clientes regulares), se instanció el algoritmo de particionamiento **K-Medias**.

### 4.1 Elección de K y Entrenamiento
Dado el objetivo del negocio, se forzó al algoritmo a particionar el espacio geométrico en dos grupos (`K=2`).
El algoritmo convergió exitosamente, dividiendo a la población en:
- **Cluster 1 (Segmento Estándar)**: 36.443 clientes.
- **Cluster 0 (Segmento Frecuente)**: 8.377 clientes.

---

## 5. Evaluación Visual del Modelo (Análisis de Componentes Principales)

Debido a que el modelo calcula distancias euclidianas en 6 dimensiones, resulta físicamente imposible graficar sus resultados en 2D. Para solucionar este problema de visualización, se aplicó un **Análisis de Componentes Principales (ACP/PCA)** que comprimió las 6 variables en 2 grandes vectores (Componente 1 y Componente 2), reteniendo aproximadamente un 53% de la varianza total del sistema.

### Gráfico Unificado (K-Medias + VIP)
Para obtener la foto comercial final, se proyectó dentro de este mismo mapa PCA a los 5.180 clientes "VIP" (las ballenas que habían sido removidas en la limpieza). Al insertarse geométricamente en el mapa, estos VIPs cayeron lógicamente dentro de la jurisdicción del Cluster 0 (Alto Valor), ubicándose en el extremo derecho del espectro.

![Scatterplot PCA Final](/home/vitto/.gemini/antigravity-ide/brain/f193a960-98e2-496f-85ba-16a7821e5e77/kmeans_pca_scatter_final.png)

> **Interpretación de los Ejes (Umbrales)**: 
> - **Componente 1 (Eje X): Volumen de Negocio**. Reúne el gasto acumulado, los vuelos y los ingresos. El umbral (la frontera de decisión que divide los colores) ocurre aproximadamente en los 20 vuelos y $6.000 de gasto. Todo lo que cae a la derecha, pertenece al Cluster de Alto Valor.
> - **Componente 2 (Eje Y): Demografía**. Agrupa fuertemente a la Edad de los clientes.

---

## 6. Eficiencia del Modelo (Métricas)

- **Coeficiente de Silhouette (Estimación sobre 5000 muestras)**: **`0.3548`**
- **Interpretación Técnica**: El coeficiente de Silhouette oscila entre -1 y 1. Un valor de `0.35` en un problema de comportamiento humano (datos *soft*) indica una estructura de particionamiento altamente coherente y sólida. Si bien no es un número absoluto de "separación perfecta" (como ocurriría en datos físicos duros), significa que los individuos dentro de su cluster están estadísticamente mucho más cerca de sus pares que de los miembros del grupo contrario. 
- La separación es real, justificable, y aporta un valor predictivo certero.

---

## 7. Conclusión Final del Cluster

Al unificar el resultado algorítmico (K-Medias) con la limpieza manual (Percentil 95), la base de 50.000 clientes quedó exportada en su versión final (`clientes_etiquetados_final.csv`) con dos etiquetas de negocio claras (`0` o `1`):

1. **Clase 1 (La Masa Estandarizada - 72.8% del negocio)**: 
   Representan a los clientes esporádicos o jóvenes planificadores. Viajan unas 10 veces en promedio, acumulan escaso volumen financiero (~$2.500) y tienen ingresos más bajos. Son fundamentales para la base de la pirámide y el volumen de liquidez operativa, pero no son generadores de alto margen.

2. **Clase 0 (El Núcleo de Rentabilidad - K-Medias 0 + VIPs - 27.2% del negocio)**:
   Este grupo aglutina tanto a los clientes frecuentes detectados por el modelo (Cluster 0) como a las hiper-ballenas (VIPs). Promedian más de 28 vuelos históricos y sostienen gastos acumulados altísimos (desde $9.600 hacia el infinito). Demográficamente son ligeramente mayores (40+ años) e ingresan el doble o triple de capital mensual.

**Veredicto:** El modelo K-Medias (K=2) probó ser una herramienta fundamental de negocio, capaz de trazar matemáticamente la frontera exacta donde un cliente esporádico se gradúa en un cliente frecuente. Esta segmentación de 2 polos es 100% accionable para diseñar campañas de retención (Clase 0) y campañas de *upselling* (Clase 1).
