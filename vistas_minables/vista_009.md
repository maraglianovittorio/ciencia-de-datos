# Vista Minable — 009

## Descripción

[ _Descripción general del dataset y propósito_ ]

---

## Propiedades (Features)

### Variables Numéricas (normalizadas Min-Max)

| Propiedad | Descripción |
|-----------|-------------|
| distancia_vuelo | |
| ocupacion_vuelo | |
| visibilidad | |
| tiempo_estimado_vuelo | |
| timestamp_vuelo | |

### Variables Categóricas (One-Hot Encoding)

#### aeropuerto_origen
- aeropuerto_origen_ASU
- aeropuerto_origen_BAQ
- aeropuerto_origen_BGA
- aeropuerto_origen_BOG
- aeropuerto_origen_BRC
- aeropuerto_origen_BSB
- aeropuerto_origen_CCS
- aeropuerto_origen_CDG
- aeropuerto_origen_CLO
- aeropuerto_origen_CLT
- aeropuerto_origen_COR
- aeropuerto_origen_CTG
- aeropuerto_origen_CWB
- aeropuerto_origen_EZE
- aeropuerto_origen_FLN
- aeropuerto_origen_GIG
- aeropuerto_origen_GRU
- aeropuerto_origen_GUA
- aeropuerto_origen_IAH
- aeropuerto_origen_IGR
- aeropuerto_origen_JFK
- aeropuerto_origen_LIM
- aeropuerto_origen_MAD
- aeropuerto_origen_MDE
- aeropuerto_origen_MDZ
- aeropuerto_origen_MEX
- aeropuerto_origen_MIA
- aeropuerto_origen_MVD
- aeropuerto_origen_NQN
- aeropuerto_origen_PEI
- aeropuerto_origen_POA
- aeropuerto_origen_PTY
- aeropuerto_origen_ROS
- aeropuerto_origen_SCL
- aeropuerto_origen_SJO
- aeropuerto_origen_SSA
- aeropuerto_origen_TUC
- aeropuerto_origen_UIO

#### aeropuerto_destino
- aeropuerto_destino_ASU
- aeropuerto_destino_BAQ
- aeropuerto_destino_BGA
- aeropuerto_destino_BOG
- aeropuerto_destino_BRC
- aeropuerto_destino_BSB
- aeropuerto_destino_CCS
- aeropuerto_destino_CDG
- aeropuerto_destino_CLO
- aeropuerto_destino_CLT
- aeropuerto_destino_COR
- aeropuerto_destino_CTG
- aeropuerto_destino_CWB
- aeropuerto_destino_EZE
- aeropuerto_destino_FLN
- aeropuerto_destino_GIG
- aeropuerto_destino_GRU
- aeropuerto_destino_GUA
- aeropuerto_destino_IAH
- aeropuerto_destino_IGR
- aeropuerto_destino_JFK
- aeropuerto_destino_LIM
- aeropuerto_destino_MAD
- aeropuerto_destino_MDE
- aeropuerto_destino_MDZ
- aeropuerto_destino_MEX
- aeropuerto_destino_MIA
- aeropuerto_destino_MVD
- aeropuerto_destino_NQN
- aeropuerto_destino_PEI
- aeropuerto_destino_POA
- aeropuerto_destino_PTY
- aeropuerto_destino_ROS
- aeropuerto_destino_SCL
- aeropuerto_destino_SJO
- aeropuerto_destino_SSA
- aeropuerto_destino_TUC
- aeropuerto_destino_UIO

#### condiciones_climaticas
- condiciones_climaticas_Lluvia
- condiciones_climaticas_Niebla
- condiciones_climaticas_Nublado
- condiciones_climaticas_Tormenta

#### congestion_aerea
- congestion_aerea_Baja
- congestion_aerea_Media

#### tipo_avion
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

#### temporada_alta
- temporada_alta_True

#### franja_horaria
- franja_horaria_manana
- franja_horaria_mediodia
- franja_horaria_noche
- franja_horaria_tarde

### Variable Objetivo

- **demora** (0 = No demorado, 1 = Demorado)

---

## Transformaciones Aplicadas

1. **Filtrado Custom**:
   - Vuelos con velocidad < 100 km/h eliminados: 562
   - Vuelos EZE↔AEP eliminados: 562
2. **Limpieza**: Columnas eliminadas: `id_vuelo, hora_salida_programada, dia_semana, puerta_embarque`
3. **Normalización Min-Max**: Escala (0-1) para todas las variables numéricas
4. **One-Hot Encoding**: Conversión de variables categóricas a dummies con `drop_first=True`

---

## Resumen

- Filas: 14438
- Columnas (features): 103
- Variable objetivo: `demora` (0=9831, 1=4607)

---

## Notas y Anotaciones

[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]