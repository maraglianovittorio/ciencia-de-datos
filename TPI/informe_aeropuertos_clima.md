# ✈️ Análisis: Influencia del Aeropuerto en las Demoras — Descomposición Clima vs Efecto Propio

---

## 1. Metodología

Se analizó la tasa de demora de cada aeropuerto bajo 4 condiciones progresivas para aislar el **efecto propio** del aeropuerto del **efecto climático/congestión**:

| Escenario | Filtro aplicado | Demora media | Std entre aeropuertos |
|---|---|---|---|
| **Sin filtro** (crudo) | Ninguno | 31.9% | 2.39pp |
| **Sin clima** | Solo Despejado | 26.6% | 4.14pp |
| **Sin congestión** | Solo Baja | — | 3.58pp |
| **Doble control** | Despejado + Baja | **18.2%** | **4.75pp** |

> La desviación estándar entre aeropuertos **aumenta** al controlar por clima (de 2.39pp a 4.75pp), lo que indica que las condiciones climáticas actúan como factor igualador, enmascarando diferencias operativas reales entre aeropuertos.

**Fuente de datos**: dataset `Vuelos.xlsx`, 15.000 registros. Para el análisis por aeropuerto se combinaron las columnas `aeropuerto_origen` y `aeropuerto_destino`, generando 30.000 observaciones aeropuerto-vuelo.

---

## 2. Tabla Maestra — 40 Aeropuertos

| Aeropuerto | Vuelos | Demora cruda | Demora pura¹ | Δpp | % explicado por clima |
|---|---|---|---|---|---|
| EZE | 4,328 | 32.5% | 17.6% | +14.9 | 46% |
| BOG | 3,764 | 31.7% | 17.0% | +14.7 | 46% |
| MDE | 2,975 | 32.7% | 19.2% | +13.5 | 41% |
| GRU | 2,958 | 30.8% | 18.4% | +12.4 | 40% |
| PTY | 2,592 | 31.2% | 19.2% | +12.0 | 38% |
| LIM | 816 | 33.6% | 20.0% | +13.6 | 40% |
| UIO | 809 | 33.1% | 20.1% | +13.0 | 39% |
| PEI | 717 | 31.9% | 18.2% | +13.7 | 43% |
| MIA | 712 | 34.0% | 22.7% | +11.3 | 33% |
| CTG | 694 | 32.4% | 16.3% | +16.2 | 50% |
| BAQ | 673 | 31.6% | 17.9% | +13.7 | 43% |
| CLO | 624 | 32.7% | 24.5% | +8.2 | 25% |
| BGA | 617 | 29.8% | 13.5% | +16.3 | 55% |
| JFK | 611 | 35.0% | 20.5% | +14.5 | 42% |
| CCS | 572 | 29.9% | 11.8% | +18.1 | 61% |
| AEP² | 562 | 30.4% | 15.8% | +14.6 | 48% |
| MVD | 511 | 27.2% | 9.9% | +17.3 | 64% |
| MEX | 418 | 29.2% | 16.2% | +12.9 | 44% |
| GIG | 414 | 29.2% | 15.5% | +13.8 | 47% |
| SCL | 382 | 35.9% | 26.3% | +9.5 | 27% |
| MAD | 322 | 33.2% | 15.0% | +18.2 | 55% |
| ROS | 302 | 34.4% | 16.7% | +17.8 | 52% |
| SJO | 254 | 33.5% | 22.9% | +10.5 | 32% |
| CWB | 253 | 26.9% | 14.6% | +12.2 | 46% |
| FLN | 249 | 28.5% | 17.0% | +11.5 | 40% |
| COR | 245 | 33.1% | 17.1% | +16.0 | 48% |
| GUA | 239 | 31.0% | 16.3% | +14.6 | 47% |
| POA | 228 | 31.1% | 20.0% | +11.1 | 36% |
| IGR | 219 | 30.6% | 8.7% | +21.9 | 72% |
| MDZ | 215 | 32.6% | 18.2% | +14.4 | 44% |
| ASU | 213 | 33.8% | 25.0% | +8.8 | 26% |
| BRC | 211 | 32.7% | 20.0% | +12.7 | 39% |
| BSB | 209 | 37.3% | 27.0% | +10.3 | 28% |
| NQN | 207 | 32.4% | 15.8% | +16.6 | 51% |
| TUC | 196 | 29.1% | 19.4% | +9.7 | 34% |
| CLT | 165 | 32.1% | 11.1% | +21.0 | 65% |
| IAH | 162 | 35.2% | **32.4%** | +2.8 | **8%** |
| SSA | 150 | 27.3% | 16.1% | +11.2 | 41% |
| AMS | 110 | 30.0% | 16.7% | +13.3 | 44% |
| CDG | 102 | 21.6% | 13.3% | +8.2 | 38% |

> ¹ *Demora pura*: tasa de demora filtrando solo vuelos con `condiciones_climaticas = Despejado` y `congestion_aerea = Baja`.
>
> ² *AEP*: Aeroparque aparece en el dataset pero la ruta EZE↔AEP no opera vuelos comerciales de pasajeros (ver sección de limpieza).

---

## 3. Volumen vs Demora

![Volumen vs Demora](/home/vitto/Documents/GitHub/ciencia-de-datos/TPI/graficos/auditoria/aeropuerto_volumen_vs_demora.png)

| Correlación | r | p-valor | Interpretación |
|---|---|---|---|
| Volumen ↔ Demora cruda | **0.084** | 0.607 | Nula — no significativa |
| Volumen ↔ Demora pura | **0.006** | 0.973 | Literalmente cero |

**La cantidad de vuelos NO influye en la tasa de demora.** Aeropuertos con alto tráfico (EZE: 4,328 vuelos) tienen tasas similares a aeropuertos medianos (CLO: 624 vuelos).

### Efecto del tamaño muestral en la varianza

| Grupo | Aeropuertos | Std demora cruda |
|---|---|---|
| Grandes (≥1,000 vuelos) | 5 (EZE, BOG, MDE, GRU, PTY) | **0.82pp** |
| Chicos (<500 vuelos) | 23 | **3.43pp** |

Los aeropuertos con pocos vuelos muestran **4x más varianza** en su tasa de demora. Esto no significa que sean más inestables operativamente — es un artefacto estadístico: con muestras chicas, las estimaciones oscilan más. Es la **ley de los grandes números** en acción.

> Caso extremo: **IAH** tiene solo 162 vuelos totales y muestra una demora pura de 32.4%, pero su muestra de control (Despejado + Baja) es de apenas 37 vuelos. Con n=37, el intervalo de confianza al 95% es ≈ ±15pp, lo que hace que su valor extremo no sea confiable.

---

## 4. Demora Cruda vs Pura por Aeropuerto

![Cruda vs Pura](/home/vitto/Documents/GitHub/ciencia-de-datos/TPI/graficos/auditoria/aeropuerto_cruda_vs_pura_volumen.png)

La barra roja (cruda) siempre es mayor que la azul (pura). La diferencia entre ambas es lo que **explica el clima + congestión**. En promedio, clima y congestión explican **~13.7pp** de la demora (de 31.9% bajan a 18.2%).

### Casos destacados

| Aeropuerto | Demora pura | Observación |
|---|---|---|
| **IAH** (Houston) | 32.4% | Apenas baja 2.8pp al controlar → la demora es casi 100% estructural. Pero ojo: n=37 en control, dato poco confiable |
| **SCL** (Santiago) | 26.3% | Solo baja 9.5pp → demora alta independiente del clima |
| **CLT** (Charlotte) | 11.1% | Baja 21pp → casi toda su demora era climática |
| **IGR** (Iguazú) | 8.7% | Baja 21.9pp → aeropuerto muy eficiente en condiciones normales. Pero n=23 en control |
| **MVD** (Montevideo) | 9.9% | Baja 17.3pp con n=101 → dato confiable, aeropuerto eficiente |

---

## 5. Impacto del Clima por Aeropuerto

![Delta por clima](/home/vitto/Documents/GitHub/ciencia-de-datos/TPI/graficos/auditoria/aeropuerto_delta_clima.png)

Este gráfico muestra cuántos puntos porcentuales de demora se eliminan al quitar el efecto del mal clima. Todos los aeropuertos se benefician de buen clima, pero en proporciones muy diferentes.

---

## 6. Composición Climática

![Composición climática](/home/vitto/Documents/GitHub/ciencia-de-datos/TPI/graficos/auditoria/aeropuerto_composicion_clima.png)

La distribución climática es **uniforme** entre aeropuertos (~45% Despejado, ~25% Nublado, ~15% Lluvia en todos). Esto descarta que las diferencias de demora se deban a que ciertos aeropuertos experimentan más tormentas.

---

## 7. Aeropuerto Origen vs Aeropuerto Destino — ¿Son la misma variable?

Un aeropuerto no se comporta igual cuando es punto de **salida** que cuando es punto de **llegada**. Para verificarlo, se comparó la tasa de demora de cada aeropuerto en ambos roles.

### Resultado global

| Métrica | Valor |
|---|---|
| Correlación tasa demora (origen ↔ destino) | **r = -0.047** (p = 0.774) |
| Std entre aeropuertos (como origen) | 4.22pp |
| Std entre aeropuertos (como destino) | 4.08pp |
| Rango como origen | 22.9% – 41.2% |
| Rango como destino | 15.7% – 36.5% |
| Media absoluta de diferencia | 4.47pp |
| Aeropuertos con diferencia > 4pp | **15 de 40** (37.5%) |

> La correlación entre la tasa de demora de un aeropuerto como origen y como destino es **prácticamente cero** (r = -0.047). Esto demuestra que `aeropuerto_origen` y `aeropuerto_destino` son **variables independientes**, no redundantes.

### Mayores asimetrías origen/destino

| Aeropuerto | % demora como origen | % demora como destino | Diferencia |
|---|---|---|---|
| **MAD** (Madrid) | 41.2% | 26.0% | **+15.1pp** |
| **POA** (Porto Alegre) | 37.2% | 23.2% | +14.0pp |
| **CDG** (París) | 27.4% | 15.7% | +11.8pp |
| **MDZ** (Mendoza) | 38.3% | 26.9% | +11.5pp |
| **MIA** (Miami) | 39.8% | 28.4% | +11.5pp |
| **TUC** (Tucumán) | 23.4% | 34.3% | **-10.9pp** |
| **CWB** (Curitiba) | 22.9% | 31.1% | -8.2pp |
| **EZE** (Ezeiza) | 32.5% | 32.5% | **0.0pp** |

### Interpretación operativa

La asimetría tiene sentido desde el punto de vista aeronáutico. Las demoras en la **salida** dependen de factores del aeropuerto de origen: gestión de puertas de embarque, carga de equipaje, despacho, espera en pista, y slot de salida. Las demoras en la **llegada** dependen del aeropuerto de destino: disponibilidad de pista, tráfico en la zona de aproximación, y capacidad de desembarque.

Un aeropuerto puede ser eficiente despachando vuelos pero ineficiente recibiéndolos (o viceversa), lo que explica que las tasas de demora como origen y destino sean estadísticamente independientes.

**Casos notables:**
- **MAD y MIA** tienen demoras mucho más altas como origen (~40%) que como destino (~27%). Esto sugiere problemas operativos en la salida (despacho, gate management).
- **TUC y CWB** muestran el patrón inverso: baja demora de salida pero alta de llegada, posiblemente por limitaciones de pista o aproximación.
- **EZE** es perfectamente simétrico (32.5% en ambos roles), lo que sugiere un comportamiento operativo estable.

---

## 8. Conclusiones

### 8.1. El clima es el principal predictor de demora, no el aeropuerto

Las condiciones climáticas y la congestión aérea explican en promedio el **43% de la demora cruda** de cada aeropuerto (rango: 8% en IAH hasta 72% en IGR). Al eliminar estos factores, la demora media cae de 31.9% a 18.2% (−13.7pp).

### 8.2. Existe un efecto propio del aeropuerto, pero es secundario

Controlando por clima y congestión, los aeropuertos aún muestran diferencias (rango: 8.7% – 32.4%, std: 4.75pp). Este efecto residual probablemente refleja factores no presentes en el dataset: infraestructura aeroportuaria, eficiencia de los controladores de tránsito aéreo, políticas operativas locales, y capacidad de pistas.

### 8.3. El volumen de vuelos NO explica la demora

La correlación entre cantidad de vuelos y tasa de demora es estadísticamente nula (r=0.084, p=0.607 para demora cruda; r=0.006, p=0.973 para demora pura). Esto significa que tener más tráfico no implica más demoras; la congestión puntual (ya capturada en `congestion_aerea`) es más relevante que el volumen total.

### 8.4. Los aeropuertos con pocos vuelos tienen estimaciones ruidosas

Los aeropuertos con menos de 500 vuelos muestran una varianza 4 veces mayor (std=3.43pp vs 0.82pp en aeropuertos grandes). En particular, aeropuertos con valores extremos (IAH, IGR) tienen muestras de control de n=23–37, insuficientes para conclusiones firmes.

### 8.5. Origen y destino son variables independientes

La correlación entre la tasa de demora como origen y como destino es r = -0.047 (no significativa). Un 37.5% de los aeropuertos muestra diferencias superiores a 4pp entre ambos roles. Esto implica que `aeropuerto_origen` y `aeropuerto_destino` capturan información distinta y no deben tratarse como una sola variable.

### 8.6. Para el modelo KNN: señal real pero costo dimensional excesivo

El aeropuerto aporta información genuina y origen/destino son independientes entre sí, lo que potencialmente duplica el valor informativo. Sin embargo, incluirlos en KNN implica generar hasta 78 columnas dummy vía One-Hot Encoding (40 aeropuertos × 2 columnas, menos referencia). Dado que KNN calcula distancias euclidianas, esta explosión dimensional diluiría la señal de variables más fuertes como `condiciones_climaticas` (rango de 37.2pp) y `congestion_aerea` (rango de 23.2pp). La recomendación es **no incluirlos en KNN**, pero documentar su efecto en el análisis exploratorio.

---

## Fuente de Datos

| Campo | Detalle |
|---|---|
| **Dataset** | `Vuelos.xlsx` |
| **Registros** | 15,000 vuelos |
| **Aeropuertos** | 40 (códigos IATA) |
| **Variables utilizadas** | `aeropuerto_origen`, `aeropuerto_destino`, `demora`, `condiciones_climaticas`, `congestion_aerea` |
| **Método de control** | Filtrado estratificado: se aislaron subgrupos con clima=Despejado y congestión=Baja para eliminar confounders |
| **Herramientas** | Python 3, pandas, matplotlib, seaborn, scipy |
| **Métrica de evaluación** | Tasa de demora (% de vuelos con `demora=1`) por aeropuerto |
| **Combinación origen/destino** | Para las secciones 1–6, cada vuelo aporta 2 observaciones (una por origen, una por destino), totalizando 30,000 observaciones aeropuerto-vuelo. La sección 7 analiza origen y destino por separado |

> **Nota metodológica**: el doble conteo (origen + destino) puede sobrerrepresentar aeropuertos hub. Sin embargo, dado que el análisis es descriptivo y comparativo (no predictivo), este enfoque es válido para estimar la "exposición" de cada aeropuerto a demoras. La sección 7 complementa este enfoque al demostrar que ambos roles son estadísticamente independientes.
