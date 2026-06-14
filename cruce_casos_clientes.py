"""
Cruce final entre vuelos clasificados y clientes clasificados.

Estructura esperada:
    - tabla de vuelos clasificados: id_vuelo + demora/no demora
    - tabla de clientes clasificados: id_cliente + cluster o etiqueta VIP
    - tabla puente: id_vuelo -> id_cliente

Salida:
    - tabla cruce con id_vuelo, demora, id_cliente y etiqueta del cliente
    - CSV y Excel en la carpeta de salida
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("TPI") / "cruces"


def _resolver_ruta(ruta: str) -> Path:
    candidata = Path(ruta)
    if candidata.exists():
        return candidata

    script_dir = Path(__file__).resolve().parent
    alternativas = [
        script_dir / ruta,
        script_dir / "TPI" / ruta,
        script_dir / "entendimiento" / ruta,
        script_dir.parent / ruta,
    ]
    for alternativa in alternativas:
        if alternativa.exists():
            return alternativa

    raise FileNotFoundError(f"No se encontró el archivo: {ruta}")


def _leer_excel(ruta: str, sheet=None) -> pd.DataFrame:
    ruta_resuelta = _resolver_ruta(ruta)
    extension = ruta_resuelta.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(ruta_resuelta)

    if extension in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(ruta_resuelta, sheet_name=sheet)

    raise ValueError(f"Formato no soportado para {ruta_resuelta.name}: {extension}")


def _normalizar_nombre(columna: str) -> str:
    return columna.strip().lower().replace(" ", "_")


def _buscar_columna(df: pd.DataFrame, candidatos: list[str]) -> str:
    mapa = {_normalizar_nombre(col): col for col in df.columns}
    for candidato in candidatos:
        candidato_norm = _normalizar_nombre(candidato)
        if candidato_norm in mapa:
            return mapa[candidato_norm]

    for col in df.columns:
        col_norm = _normalizar_nombre(col)
        if any(token in col_norm for token in candidatos):
            return col

    raise KeyError("No se encontró ninguna columna válida entre: " + ", ".join(candidatos))


def construir_cruce(
    archivo_vuelos_clasificados: str,
    archivo_clientes_clasificados: str,
    archivo_casos_nuevos: str,
    sheet_casos=1,
    sheet_vuelos_clasificados=0,
    sheet_clientes_clasificados=0,
) -> pd.DataFrame:
    vuelos = _leer_excel(archivo_vuelos_clasificados, sheet=sheet_vuelos_clasificados)
    clientes = _leer_excel(archivo_clientes_clasificados, sheet=sheet_clientes_clasificados)
    casos = _leer_excel(archivo_casos_nuevos, sheet=sheet_casos)

    col_vuelo_vuelos = _buscar_columna(vuelos, ["id_vuelo", "vuelo", "flight"])
    col_demora = _buscar_columna(vuelos, ["demora", "demorado", "no_demora", "estado"])
    col_cliente_master = _buscar_columna(clientes, ["id_cliente", "cliente", "customer"])
    col_etiqueta_cliente = _buscar_columna(
        clientes,
        ["cluster", "etiqueta", "vip", "categoria", "label", "segmento", "grupo"],
    )
    col_vuelo_casos = _buscar_columna(casos, ["id_vuelo", "vuelo", "flight"])
    col_cliente_casos = _buscar_columna(casos, ["id_cliente", "cliente", "customer"])

    vuelos_limpio = vuelos.rename(
        columns={
            col_vuelo_vuelos: "id_vuelo",
            col_demora: "demora",
        }
    )[["id_vuelo", "demora"]].dropna(subset=["id_vuelo"])

    clientes_limpio = clientes.rename(
        columns={
            col_cliente_master: "id_cliente",
            col_etiqueta_cliente: "etiqueta_cliente",
        }
    )[["id_cliente", "etiqueta_cliente"]].dropna(subset=["id_cliente"])

    casos_limpio = casos.rename(
        columns={
            col_vuelo_casos: "id_vuelo",
            col_cliente_casos: "id_cliente",
        }
    )[["id_vuelo", "id_cliente"]].dropna(subset=["id_vuelo", "id_cliente"])

    cruce = vuelos_limpio.merge(casos_limpio, on="id_vuelo", how="left")
    cruce = cruce.merge(clientes_limpio, on="id_cliente", how="left")

    columnas_frente = ["id_vuelo", "demora", "id_cliente", "etiqueta_cliente"]
    restantes = [c for c in cruce.columns if c not in columnas_frente]
    return cruce[columnas_frente + restantes]


def guardar_salida(df: pd.DataFrame, output_dir: Path, nombre_base: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{nombre_base}.csv"
    xlsx_path = output_dir / f"{nombre_base}.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path, xlsx_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cruza vuelos clasificados con clientes clasificados usando la hoja puente de casos nuevos."
    )
    parser.add_argument(
        "--vuelos-clasificados",
        default="predicciones_casos_nuevos.csv",
        help="Archivo Excel con los vuelos clasificados (id_vuelo + demora/no demora).",
    )
    parser.add_argument(
        "--clientes-clasificados",
        default="vistas_minables/clientes_etiquetados_final.csv",
        help="Archivo con los clientes clasificados (id_cliente + cluster o etiqueta VIP).",
    )
    parser.add_argument(
        "--casos-nuevos",
        default="casosNuevos.xlsx - VuelosNuevos_Clientes.csv",
        help="Archivo Excel con la hoja puente id_vuelo -> id_cliente.",
    )
    parser.add_argument(
        "--sheet-casos",
        default=0,
        help="Hoja puente del archivo de casos nuevos. Por defecto usa la primera hoja (índice 0).",
    )
    parser.add_argument(
        "--sheet-vuelos-clasificados",
        default=0,
        help="Hoja del archivo de vuelos clasificados. Por defecto usa la primera hoja (índice 0).",
    )
    parser.add_argument(
        "--sheet-clientes-clasificados",
        default=0,
        help="Hoja del archivo de clientes clasificados. Por defecto usa la primera hoja (índice 0).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directorio de salida para la tabla cruce.",
    )
    parser.add_argument(
        "--output-name",
        default="cruce_vuelos_clientes",
        help="Nombre base del archivo de salida.",
    )

    args = parser.parse_args()

    try:
        sheet_casos = int(args.sheet_casos)
    except (TypeError, ValueError):
        sheet_casos = args.sheet_casos

    try:
        sheet_vuelos_clasificados = int(args.sheet_vuelos_clasificados)
    except (TypeError, ValueError):
        sheet_vuelos_clasificados = args.sheet_vuelos_clasificados

    try:
        sheet_clientes_clasificados = int(args.sheet_clientes_clasificados)
    except (TypeError, ValueError):
        sheet_clientes_clasificados = args.sheet_clientes_clasificados

    cruce = construir_cruce(
        archivo_vuelos_clasificados=args.vuelos_clasificados,
        archivo_clientes_clasificados=args.clientes_clasificados,
        archivo_casos_nuevos=args.casos_nuevos,
        sheet_casos=sheet_casos,
        sheet_vuelos_clasificados=sheet_vuelos_clasificados,
        sheet_clientes_clasificados=sheet_clientes_clasificados,
    )

    salida_dir = Path(args.output_dir)
    csv_path, xlsx_path = guardar_salida(cruce, salida_dir, args.output_name)

    print("=" * 70)
    print("CRUCE CASOS NUEVOS + CLIENTES")
    print("=" * 70)
    print(f"[INFO] Filas resultantes: {len(cruce)}")
    print(f"[INFO] Columnas resultantes: {len(cruce.columns)}")
    print(f"[INFO] CSV generado: {csv_path}")
    print(f"[INFO] Excel generado: {xlsx_path}")


if __name__ == "__main__":
    main()