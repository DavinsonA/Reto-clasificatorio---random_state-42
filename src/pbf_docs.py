"""Parser de los PBF geográficos del corpus de ADL.

Una entidad geográfica -> un bloque.

Los atributos descriptivos de cada entidad se convierten a texto
estructurado. La geometría y los identificadores técnicos se
excluyen del cuerpo porque no aportan valor textual directo.

Los PBF pueden tener columnas diferentes entre sí, por lo que
el parser conserva dinámicamente los atributos disponibles en
cada entidad.

Si un PBF no contiene entidades, se genera un bloque mínimo
"""

from __future__ import annotations
from typing import Any
import geopandas as gpd
import pyogrio
from .core import CatalogEntry, RawDoc, clean


# Campos técnicos que no entran al texto, para reducir tokens.
_SKIP_COLUMNS = {
    "geometry", #Muchas coordenadas que generan ruido al corpus
    "fid", #Identificador sin aporte
    "mvt_id" #Identificador sin aporte
}

def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un PBF del corpus en `RawDoc`."""

    capas = pyogrio.list_layers(str(entry.path))

    blocks: list[str] = []
    columnas: dict[str, list[str]] = {}
    capas_extra: list[dict[str, Any]] = []

    # Los PBF revisados tienen una sola capa, pero mantenemos
    # soporte para varias capas por seguridad.
    multi_capa = len(capas) > 1

    for capa in capas:
        nombre_capa = str(capa[0])

        gdf = gpd.read_file(
            str(entry.path),
            layer=nombre_capa,
        )

        columnas[nombre_capa] = [
            str(col)
            for col in gdf.columns
        ]

        geometry_types: dict[str, int] = {}

        if "geometry" in gdf.columns and not gdf.empty:
            geometry_types = {
                str(tipo): int(cantidad)
                for tipo, cantidad
                in gdf.geometry.geom_type.value_counts().items()
            }

        capas_extra.append(
            {
                "nombre": nombre_capa,
                "num_entidades": len(gdf),
                "geometry_types": geometry_types,
                "crs": str(gdf.crs) if gdf.crs else None,
            }
        )

        # Una entidad geográfica -> un bloque.
        for _, row in gdf.iterrows():
            block = _row_to_block(row)

            if not block:
                continue

            if multi_capa:
                block = f"[{nombre_capa}] {block}"

            blocks.append(block)

    extra: dict[str, Any] = {
        "observatorio": entry.observatory,
        "tabular": True,
        "columnas": columnas,
        "capas": capas_extra,
        "num_entidades": len(blocks),
    }

    # Un PBF puede existir correctamente pero estar vacío.
    if not blocks:
        blocks = [
            f"{entry.path.stem}. Observatorio: "
            f"{entry.observatory}"
        ]
        extra["contenido_minimo"] = True

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title=clean(entry.path.stem),
        blocks=tuple(blocks),
        extra=extra,
    )


def _row_to_block(row: Any) -> str:
    """Convierte una entidad geográfica en un bloque."""

    pares: list[str] = []

    for columna, valor in row.items():
        columna = str(columna)

        # No incluir geometría ni identificadores técnicos.
        if columna.lower() in _SKIP_COLUMNS:
            continue

        valor_limpio = _clean_value(valor)

        if not valor_limpio:
            continue

        pares.append(
            f"{columna}: {valor_limpio}"
        )

    return " | ".join(pares)


def _clean_value(value: Any) -> str:
    """Convierte un atributo geográfico a texto limpio."""

    if value is None:
        return ""

    # Manejo de NaN
    try:
        if value != value:
            return ""
    except Exception:
        pass

    if isinstance(value, (list, tuple)):
        values = [
            clean(str(item))
            for item in value
            if item not in (None, "")
        ]
        return ", ".join(
            value for value in values if value
        )

    return clean(str(value))