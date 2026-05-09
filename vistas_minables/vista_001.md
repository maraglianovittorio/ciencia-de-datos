# Vista Minable — 001

## Descripción

[ _Descripción general del dataset y propósito_ ]

---

## Propiedades (Features)

### Variables Numéricas (normalizadas Min-Max)

| Propiedad | Descripción |
|-----------|-------------|
| distancia_vuelo | Distancia de la ruta en kilómetros |
| ocupacion_vuelo | Porcentaje de ocupación del vuelo |
| puerta_embarque | Identificador de puerta de embarque |
| visibilidad | Visibilidad en metros |
| tiempo_estimado_vuelo | Tiempo estimado de vuelo en minutos |

### Variables Categóricas (One-Hot Encoding)

#### Aeropuerto Origen
[ Lista de aeropuertos de origen codificados como dummies ]

#### Aeropuerto Destino
[ Lista de aeropuertos de destino codificados como dummies ]

#### Hora de Salida Programada
[ Franjas horarias cada 15 minutos codificadas como dummies ]

#### Día de la Semana
- dia_semana_Lunes
- dia_semana_Martes
- dia_semana_Miércoles
- dia_semana_Jueves
- dia_semana_Viernes
- dia_semana_Sábado

#### Condiciones Climáticas
- condiciones_climaticas_Lluvia
- condiciones_climaticas_Niebla
- condiciones_climaticas_Nublado
- condiciones_climaticas_Tormenta

#### Congestión Aérea
- congestion_aerea_Baja
- congestion_aerea_Media

#### Tipo de Avión
- tipo_avion_Airbus A320
- tipo_avion_Airbus A320neo
- tipo_avion_Airbus A330
- tipo_avion_Airbus A350
- tipo_avion_Boeing 737
- tipo_avion_Boeing 737 MAX
- tipo_avion_Boeing 757
- tipo_avion_Boeing 777
- tipo_avion_Boeing 787
- tipo_avion_Embraer E190
- tipo_avion_Embraer E195

#### Temporada Alta
- temporada_alta_True

### Variable Objetivo

- **demora** (0 = No demorado, 1 = Demorado)

---

## Transformaciones Aplicadas

1. **Limpieza**: Eliminación de `id_vuelo`, filas duplicadas, imputación de nulos (mediana para numéricas, moda para categóricas)
2. **Normalización Min-Max**: Escala (0-1) para todas las variables numéricas
3. **One-Hot Encoding**: Conversión de variables categóricas a dummies con `drop_first=True`

---

## Notas y Anotaciones

[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]