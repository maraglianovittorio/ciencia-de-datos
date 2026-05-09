# Vista Minable — 004

## Descripción

[ _Descripción general del dataset y propósito_ ]

---

## Propiedades (Features)

### Variables Numéricas (normalizadas Min-Max)

| Propiedad | Descripción |
|-----------|-------------|
| distancia_vuelo | |
| ocupacion_vuelo | |
| puerta_embarque | |
| visibilidad | |
| tiempo_estimado_vuelo | |

### Variables Categóricas (One-Hot Encoding)

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

### Variable Objetivo

- **demora** (0 = No demorado, 1 = Demorado)

---

## Transformaciones Aplicadas

1. **Filtrado Custom**:
   - Vuelos con velocidad < 100 km/h eliminados: 562
   - Vuelos EZE↔AEP eliminados: 562
2. **Limpieza**: Columnas eliminadas: `id_vuelo, hora_salida_programada, dia_semana, aeropuerto_origen, aeropuerto_destino`
3. **Normalización Min-Max**: Escala (0-1) para todas las variables numéricas
4. **One-Hot Encoding**: Conversión de variables categóricas a dummies con `drop_first=True`

---

## Resumen

- Filas: 14438
- Columnas (features): 23
- Variable objetivo: `demora` (0=9831, 1=4607)

---

## Notas y Anotaciones

[ _Espacio para agregar observaciones, insights o anotaciones adicionales_ ]