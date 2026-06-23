# Comparacion de Algoritmos de Clustering

Se ejecutaron 3 algoritmos (K-Medias, Jerarquico-Ward, Bietapico-Birch) con k=2 y k=3 sobre la base de clientes (whales P95 excluidos).

> **Entradas:** K-Medias y Jerarquico clusterizan sobre `vista_clientes_kmeans_normalizada.csv` (6 numericas Z-score). Bietapico clusteriza sobre una vista mixta (6 numericas std + OHE de 6 categoricas) construida in-memory desde `Clientes.xlsx`.
> **Advertencia:** los Silhouette del Bietapico no son estrictamente comparables con los de K-Medias/Jerarquico por calcularse en espacios de dimension distinta. La metrica de **separacion de medias (F)** si es comparable (se calcula sobre las mismas 6 numericas en los 3 casos).

## Tabla comparativa

| Algoritmo | k | Silhouette global | Silhouette por cluster | Separacion medias (F) | Score combinado | Tamanios |
|-----------|---|-------------------|------------------------|------------------------|-----------------|----------|
| kmeans | 2 | **0.3659** | C0=0.0393, C1=0.4378 | **15721.8329** | 1.0000 | C0=8377, C1=36443 |
| kmeans | 3 | **0.1739** | C0=0.0231, C1=0.2125, C2=0.1855 | **12538.2820** | 0.1990 | C0=6048, C1=16623, C2=22149 |
| jerarquico | 2 | **0.3099** | C0=-0.0005, C1=0.3885 | **12512.5514** | 0.5507 | C0=9329, C1=35491 |
| jerarquico | 3 | **0.3062** | C0=0.3485, C1=0.1068, C2=0.1975 | **15714.5917** | 0.8438 | C0=35491, C1=5908, C2=3421 |
| bietapico | 2 | **0.1920** | C0=-0.0109, C1=0.2515 | **10433.9016** | 0.0471 | C0=10479, C1=34341 |
| bietapico | 3 | **0.1895** | C0=0.22, C1=0.0713, C2=0.1143 | **12792.9775** | 0.2637 | C0=34341, C1=6852, C2=3627 |

## Veredicto

- **Mejor Silhouette global:** `kmeans` con k=2 -> 0.3659
- **Mejor separacion de medias (F):** `kmeans` con k=2 -> 15721.8329
- **Score combinado (50% Silhouette + 50% Separacion, normalizados):** ganador `kmeans` con k=2 -> score=1.0000

### Recomendacion para la fase de evaluacion

Se recomienda continuar la fase de evaluacion con **kmeans (k=2)**, por presentar el mejor balance entre coherencia interna (Silhouette) y separacion entre grupos (F between/within).

Archivo de asignaciones correspondiente: `clientes_clusters_kmeans.csv`.
