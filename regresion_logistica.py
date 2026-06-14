import csv
import math
import os


COEFICIENTES = {
    "distancia_vuelo": 0.564,
    "ocupacion_vuelo": 0.161,
    "visibilidad": 0.018,
    "tiempo_estimado_vuelo": -0.749,
    "timestamp_vuelo": 0.409,
    "aeropuerto_origen_ASU": -0.090,
    "aeropuerto_origen_BAQ": -0.093,
    "aeropuerto_origen_BGA": -0.360,
    "aeropuerto_origen_BOG": -0.184,
    "aeropuerto_origen_BRC": -0.362,
    "aeropuerto_origen_BSB": 0.101,
    "aeropuerto_origen_CCS": -0.462,
    "aeropuerto_origen_CDG": -0.297,
    "aeropuerto_origen_CLO": -0.077,
    "aeropuerto_origen_CLT": -0.254,
    "aeropuerto_origen_COR": -0.256,
    "aeropuerto_origen_CTG": -0.210,
    "aeropuerto_origen_CWB": -0.522,
    "aeropuerto_origen_EZE": -0.042,
    "aeropuerto_origen_FLN": -0.018,
    "aeropuerto_origen_GIG": -0.157,
    "aeropuerto_origen_GRU": -0.086,
    "aeropuerto_origen_GUA": -0.063,
    "aeropuerto_origen_IAH": 0.250,
    "aeropuerto_origen_IGR": -0.528,
    "aeropuerto_origen_JFK": -0.060,
    "aeropuerto_origen_LIM": -0.105,
    "aeropuerto_origen_MAD": 0.262,
    "aeropuerto_origen_MDE": -0.213,
    "aeropuerto_origen_MDZ": -0.043,
    "aeropuerto_origen_MEX": -0.416,
    "aeropuerto_origen_MIA": 0.169,
    "aeropuerto_origen_MVD": -0.419,
    "aeropuerto_origen_NQN": -0.481,
    "aeropuerto_origen_PEI": -0.178,
    "aeropuerto_origen_POA": 0.037,
    "aeropuerto_origen_PTY": -0.320,
    "aeropuerto_origen_ROS": -0.226,
    "aeropuerto_origen_SCL": -0.055,
    "aeropuerto_origen_SJO": -0.012,
    "aeropuerto_origen_SSA": -0.415,
    "aeropuerto_origen_TUC": -0.656,
    "aeropuerto_origen_UIO": -0.237,
    "aeropuerto_destino_ASU": 0.125,
    "aeropuerto_destino_BAQ": 0.259,
    "aeropuerto_destino_BGA": 0.265,
    "aeropuerto_destino_BOG": 0.294,
    "aeropuerto_destino_BRC": 0.112,
    "aeropuerto_destino_BSB": 0.410,
    "aeropuerto_destino_CCS": 0.247,
    "aeropuerto_destino_CDG": -0.559,
    "aeropuerto_destino_CLO": 0.248,
    "aeropuerto_destino_CLT": 0.418,
    "aeropuerto_destino_COR": 0.257,
    "aeropuerto_destino_CTG": 0.293,
    "aeropuerto_destino_CWB": 0.100,
    "aeropuerto_destino_EZE": 0.314,
    "aeropuerto_destino_FLN": -0.152,
    "aeropuerto_destino_GIG": -0.080,
    "aeropuerto_destino_GRU": 0.240,
    "aeropuerto_destino_GUA": 0.108,
    "aeropuerto_destino_IAH": 0.375,
    "aeropuerto_destino_IGR": 0.055,
    "aeropuerto_destino_JFK": 0.378,
    "aeropuerto_destino_LIM": 0.287,
    "aeropuerto_destino_MAD": -0.125,
    "aeropuerto_destino_MDE": 0.351,
    "aeropuerto_destino_MDZ": -0.202,
    "aeropuerto_destino_MEX": 0.257,
    "aeropuerto_destino_MIA": 0.022,
    "aeropuerto_destino_MVD": -0.056,
    "aeropuerto_destino_NQN": 0.315,
    "aeropuerto_destino_PEI": 0.256,
    "aeropuerto_destino_POA": -0.275,
    "aeropuerto_destino_PTY": 0.266,
    "aeropuerto_destino_ROS": 0.273,
    "aeropuerto_destino_SCL": 0.337,
    "aeropuerto_destino_SJO": 0.388,
    "aeropuerto_destino_SSA": -0.085,
    "aeropuerto_destino_TUC": 0.271,
    "aeropuerto_destino_UIO": 0.384,
    "condiciones_climaticas_Lluvia": 0.490,
    "condiciones_climaticas_Niebla": 0.942,
    "condiciones_climaticas_Nublado": -0.088,
    "condiciones_climaticas_Tormenta": 1.599,
    "congestion_aerea_Baja": -1.124,
    "congestion_aerea_Media": -0.644,
    "tipo_avion_Airbus A320": -0.024,
    "tipo_avion_Airbus A320neo": 0.032,
    "tipo_avion_Airbus A330": 0.262,
    "tipo_avion_Airbus A350": 0.091,
    "tipo_avion_Boeing 737": 0.027,
    "tipo_avion_Boeing 737 MAX": 0.077,
    "tipo_avion_Boeing 757": 0.086,
    "tipo_avion_Boeing 777": 0.033,
    "tipo_avion_Boeing 787": 0.081,
    "tipo_avion_Embraer E190": -0.047,
    "tipo_avion_Embraer E195": 0.071,
    "temporada_alta_True": 0.472,
    "franja_horaria_manana": -0.113,
    "franja_horaria_mediodia": -0.218,
    "franja_horaria_noche": -0.301,
    "franja_horaria_tarde": -0.240,
}

B0 = -0.785
CUT_VALUE = 0.305
VISTA_MINABLE = os.path.join("vistas_minables", "vista_011.csv")


def sigmoide(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def calcular_z(fila, coefs, b0):
    z = b0
    for variable, beta in coefs.items():
        valor = fila.get(variable, "0")
        if valor == "" or valor is None:
            valor = "0"
        try:
            xi = float(valor)
        except ValueError:
            xi = 0.0
        z += beta * xi
    return z


def predecir_probabilidad(fila, coefs, b0):
    return sigmoide(calcular_z(fila, coefs, b0))


def clasificar(p, cut=CUT_VALUE):
    return 1 if p >= cut else 0


def predecir(fila, coefs=None, b0=None, cut=None):
    """
    Predice probabilidad y clase para una fila (dict).
    Si no se pasan coefs/b0/cut, usa los del módulo (modelo de vista 011).
    Devuelve (probabilidad, clase).
    """
    if coefs is None:
        coefs = COEFICIENTES
    if b0 is None:
        b0 = B0
    if cut is None:
        cut = CUT_VALUE
    p = predecir_probabilidad(fila, coefs, b0)
    return p, clasificar(p, cut)


def predecir_lote(filas, coefs=None, b0=None, cut=None):
    """
    Predice probabilidad y clase para una lista de filas (dict).
    Devuelve una lista de tuplas (probabilidad, clase).
    """
    if coefs is None:
        coefs = COEFICIENTES
    if b0 is None:
        b0 = B0
    if cut is None:
        cut = CUT_VALUE
    return [predecir(f, coefs, b0, cut) for f in filas]


def cargar_filas(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evaluar(filas, coefs, b0, cut):
    tp = fp = tn = fn = 0
    probs = []
    correctos = 0
    for fila in filas:
        y_real = int(fila["demora"])
        p = predecir_probabilidad(fila, coefs, b0)
        y_hat = clasificar(p, cut)
        probs.append(p)
        if y_real == 1 and y_hat == 1:
            tp += 1
            correctos += 1
        elif y_real == 0 and y_hat == 0:
            tn += 1
            correctos += 1
        elif y_real == 0 and y_hat == 1:
            fp += 1
        else:
            fn += 1
    total = len(filas)
    accuracy = correctos / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    p_min = min(probs)
    p_max = max(probs)
    p_media = sum(probs) / total if total else 0.0
    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "p_min": p_min,
        "p_max": p_max,
        "p_media": p_media,
    }


def imprimir_reporte(r):
    print("=" * 60)
    print("Regresion Logistica desde cero (vista 011)")
    print("=" * 60)
    print(f"Filas evaluadas: {r['total']}")
    print(f"Cut value: {CUT_VALUE}")
    print()
    print("Matriz de confusion:")
    print("                 Predicho 0    Predicho 1")
    print(f"Real 0           {r['tn']:>10}    {r['fp']:>10}")
    print(f"Real 1           {r['fn']:>10}    {r['tp']:>10}")
    print()
    print(f"Accuracy : {r['accuracy']:.4f}  (SPSS = 0.6360)")
    print(f"Precision: {r['precision']:.4f}")
    print(f"Recall   : {r['recall']:.4f}")
    print(f"F1       : {r['f1']:.4f}")
    print()
    print(f"Probabilidad - min: {r['p_min']:.4f}  max: {r['p_max']:.4f}  media: {r['p_media']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    filas = cargar_filas(VISTA_MINABLE)
    resultado = evaluar(filas, COEFICIENTES, B0, CUT_VALUE)
    imprimir_reporte(resultado)

    print("\nEjemplo: primera fila")
    f0 = filas[0]
    p0 = predecir_probabilidad(f0, COEFICIENTES, B0)
    y0 = clasificar(p0)
    print(f"  P(demora=1) = {p0:.4f}  ->  clase {y0}  (real: {f0['demora']})")
