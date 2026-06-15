# Evaluación de Modelo K-Medias (K=2) - Boxplots

Se entrenó el algoritmo de K-Medias con `K=2` basándonos exclusivamente en el dataset 100% cuantitativo, estandarizado y libre de outliers (*whales*).

Acá tenés el análisis gráfico de cómo quedó particionado cada cluster variable por variable:

![Boxplots de Variables por Cluster](/home/vitto/.gemini/antigravity-ide/brain/f193a960-98e2-496f-85ba-16a7821e5e77/kmeans_boxplots_variables.png)

## Análisis Técnico del Resultado

1. **Coeficiente de Silhouette (0.35)**: 
   Un score de 0.35 indica una separación aceptable. En problemas de datos de comportamiento humano esto es un resultado completamente normal. Indica que los grupos están estadísticamente diferenciados, aunque lógicamente hay una "zona gris" de clientes frontera que comparten similitudes con ambos mundos.

2. **Interpretación de los Boxplots**: 
   Cada subgráfico representa una de tus 6 variables. En el Eje X tenés los dos clusters (0 y 1), y en el Eje Y tenés los valores de la variable escalados (Z-Scores, es decir, cuántos desvíos estándar se alejan de la media general).
   - Observando las cajas vas a poder dictaminar inmediatamente el "perfil" del cluster. Por ejemplo, podés notar si un cluster agrupa exclusivamente a la gente que gasta por arriba del promedio, o a los que vuelan menos.

> [!TIP]
> **Dataset Clasificado**
> El dataset con las etiquetas ya está guardado en `vistas_minables/clientes_con_clusters.csv`. Tené en cuenta que la columna `Cluster` es la que te indica a qué grupo (0 o 1) pertenece cada registro para tus análisis de negocio.
