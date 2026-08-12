"""Contratos del chunking: configuracion, borrador de chunk y utilidades comunes.

Este modulo no importa a sus hermanos: es la base sobre la que se apoyan
`sentence`, `units`, `pack` y `evidence`. La composicion vive en `__init__.py`,
igual que en `src/extract/`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

# Formatos reales del corpus (ver `src/extract/__init__.py::PARSERS`). No hay
# HTML: el indice de ADL no trae ninguno. Un formato fuera de estas dos listas
# no tiene politica y debe fallar, no elegirse en silencio.
NARRATIVE_FORMATS = frozenset({"json", "pdf", "txt", "jpg", "jpeg", "png", "avif"})
TABULAR_FORMATS = frozenset({"csv", "xlsx", "pbf"})

# Separador entre unidades dentro de un chunk. Es invertible: `clean()` colapsa
# todo `\s+` a un espacio, asi que ningun bloque de `src/extract/` contiene un
# salto de linea y `texto.split("\n")` recupera las unidades originales.
UNIT_SEPARATOR = "\n"

CHUNK_ID_TEMPLATE = "{doc_id}__chunk_{posicion:06d}"


class UnknownFormatError(ValueError):
    """El formato no tiene politica de chunking registrada."""


class TokenCounter(Protocol):
    """Cuenta tokens con el tokenizador real del encoder, inyectado desde fuera.

    El encoder todavia no esta elegido: el chunker nunca cuenta tokens por su
    cuenta ni rellena `num_tokens` con un sustituto.
    """

    def __call__(self, text: str) -> int:  # pragma: no cover - protocolo
        ...


def count_words(text: str) -> int:
    """Cuenta palabras como lo hace el validador del comite (`CLAUDE.md` §2.3)."""
    return len(text.split())


def make_chunk_id(doc_id: str, posicion: int) -> str:
    """Identificador determinista del chunk: mismo documento y config -> mismo id."""
    return CHUNK_ID_TEMPLATE.format(doc_id=doc_id, posicion=posicion)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Presupuestos del chunker. Baseline de implementacion, no optimo demostrado.

    `target_words` es un objetivo blando y `max_words` el techo de un chunk
    empaquetado; una unidad indivisible que ya supere el techo se emite entera
    (ver `pack.py`). Los presupuestos de salida (`output_*`) rigen la evidencia
    entregada, que es una unidad distinta de la indexada.
    """

    target_words: int = 200
    soft_min_words: int = 120
    max_words: int = 250
    overlap_units: int = 0
    # False cierra el chunk en cada frontera de bloque. Es la forma explicita de
    # correr E1 (packing solo dentro del bloque) y, con `max_words` por encima
    # del bloque mas largo, de reproducir E0 (bloque = chunk). Ver
    # `block_as_chunk_config`.
    cross_block_packing: bool = True
    output_target_words: int = 240
    output_max_words: int = 250
    # Presupuesto de tokens del encoder. Sin encoder elegido va en None y el
    # baseline trabaja solo con palabras.
    max_tokens: int | None = None
    # pysbd no tiene reglas de portugues (108 documentos del corpus). El
    # romance mas cercano disponible es el espanol: es una aproximacion
    # declarada, no una verdad linguistica.
    portuguese_ruleset: str = "es"
    # Idioma cuando la deteccion falla o devuelve algo que pysbd no soporta.
    fallback_ruleset: str = "en"

    def __post_init__(self) -> None:
        """Valida los presupuestos; una config incoherente falla al construirse."""
        if self.target_words <= 0 or self.soft_min_words <= 0 or self.max_words <= 0:
            raise ValueError("los presupuestos en palabras deben ser positivos")
        if self.soft_min_words > self.target_words:
            raise ValueError("soft_min_words no puede superar target_words")
        if self.target_words > self.max_words:
            raise ValueError("target_words no puede superar max_words")
        if self.output_target_words <= 0 or self.output_max_words <= 0:
            raise ValueError("los presupuestos de salida deben ser positivos")
        if self.output_target_words > self.output_max_words:
            raise ValueError("output_target_words no puede superar output_max_words")
        if self.max_tokens is not None and self.max_tokens <= 0:
            raise ValueError("max_tokens debe ser positivo o None")
        if self.overlap_units < 0:
            raise ValueError("overlap_units no puede ser negativo")
        if self.overlap_units != 0:
            # La interfaz queda, el comportamiento no: E3 se implementa cuando
            # se mida, no antes (docs/research §7).
            raise NotImplementedError("el baseline solo soporta overlap_units=0")


DEFAULT_CONFIG = ChunkingConfig()

# Techo por encima del bloque mas largo del corpus (9.235 palabras, `F1-AIINDEX-041`):
# con el no se segmenta ningun bloque y ninguna unidad queda marcada oversized.
NO_SPLIT_WORDS = 100_000


def block_as_chunk_config(**overrides: Any) -> ChunkingConfig:
    """Config de la politica E0 del research: **un bloque, un chunk**.

    Es la referencia contra la que se comparan las demas politicas, asi que
    tiene que ser explicita y reproducible, no el efecto lateral de unos
    presupuestos numericos extremos.

    Combina las dos condiciones necesarias: sin packing entre bloques y sin
    segmentacion por oraciones (ningun bloque supera el techo).
    """
    parameters: dict[str, Any] = {
        "target_words": NO_SPLIT_WORDS,
        "soft_min_words": 1,
        "max_words": NO_SPLIT_WORDS,
        "cross_block_packing": False,
    }
    parameters.update(overrides)
    return ChunkingConfig(**parameters)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """Chunk antes del embedding: todo salvo `num_tokens`, que exige el encoder.

    `texto` es el contenido extraido tal cual, concatenando unidades adyacentes
    con `UNIT_SEPARATOR`. No lleva prefijos de contexto: eso es la ablacion E4 y
    se materializaria en un `embedding_text` aparte, sin mutar `texto`.
    """

    doc_id: str
    chunk_id: str
    fuente: str
    formato: str
    fenomeno: int
    posicion: int
    texto: str
    num_words: int
    block_start: int
    block_end: int
    unit_count: int
    group_key: str | None
    oversized_atomic: bool

    def as_dict(self) -> dict[str, Any]:
        """Vuelca el borrador completo (artefacto de investigacion, no Tabla 1)."""
        return asdict(self)


def materialize_metadata(chunk: ChunkDraft, token_counter: TokenCounter) -> dict[str, Any]:
    """Convierte un borrador en la metadata de la Tabla 1 con el tokenizador real.

    Args:
        chunk: borrador producido por el chunker.
        token_counter: contador del encoder ya elegido. No hay valor por defecto
            a proposito: inventar `num_tokens` es peor que no tenerlo.

    Returns:
        Los ocho campos obligatorios de la Tabla 1 de la especificacion.
    """
    return {
        "doc_id": chunk.doc_id,
        "chunk_id": chunk.chunk_id,
        "fuente": chunk.fuente,
        "formato": chunk.formato,
        "fenomeno": chunk.fenomeno,
        "posicion": chunk.posicion,
        "num_tokens": token_counter(chunk.texto),
        "texto": chunk.texto,
    }
