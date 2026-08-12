"""Contrato de encoders: formato de texto, tokenizador y modelo, desacoplado del chunker.

`src/chunking` inyecta un `TokenCounter` (`Callable[[str], int]`) sin conocer nada
del encoder que lo produce (ver `TokenCounter` en `src/chunking/core.py`); este
modulo es quien lo produce. `EncoderModel` separa a proposito el tokenizador
(liviano, basta para el audit de tokens) del modelo completo de
`sentence-transformers` (pesado, exige los pesos y GPU/CPU real): auditar tokens
no debe exigir descargar ni cargar ningun modelo completo.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# HuggingFace usa enteros centinela (del orden de 10**30) en `model_max_length`
# para representar "sin limite conocido". No son un contexto efectivo: compararlos
# con el declarado en el registro daria una advertencia sin sentido.
_SANE_MODEL_MAX_LENGTH = 1_000_000


class EncoderConfigError(ValueError):
    """La configuracion declarada de un encoder es incoherente."""


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Configuracion declarativa de un encoder candidato (CLAUDE.md S4.1).

    `max_sequence_length` es el contexto efectivo que el equipo decide usar en
    retrieval, no el que reporte el tokenizador: varios checkpoints multilingues
    declaran limites de HuggingFace poco fiables (ver `_SANE_MODEL_MAX_LENGTH`).
    Ningun encoder de este registro esta elegido todavia; esto solo fija su
    configuracion para poder auditarlos con el mismo contrato.
    """

    name: str
    model_id: str
    embedding_dimension: int
    max_sequence_length: int
    query_prefix: str = ""
    document_prefix: str = ""
    trust_remote_code: bool = False
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        """Valida la configuracion; un encoder mal declarado falla al registrarse."""
        if not self.name or not self.model_id:
            raise EncoderConfigError("name y model_id son obligatorios")
        if self.embedding_dimension <= 0:
            raise EncoderConfigError("embedding_dimension debe ser positiva")
        if self.max_sequence_length <= 0:
            raise EncoderConfigError("max_sequence_length debe ser positiva")

    def format_query(self, text: str) -> str:
        """Texto exacto que recibira el tokenizador/modelo para una consulta."""
        return f"{self.query_prefix}{text}"

    def format_document(self, text: str) -> str:
        """Texto exacto que recibira el tokenizador/modelo para un chunk."""
        return f"{self.document_prefix}{text}"


def _load_tokenizer(spec: EncoderSpec) -> PreTrainedTokenizerBase:
    """Carga solo el tokenizador: el audit de tokens no necesita los pesos del modelo."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, trust_remote_code=spec.trust_remote_code
    )
    reported = getattr(tokenizer, "model_max_length", None)
    if (
        isinstance(reported, int)
        and reported < _SANE_MODEL_MAX_LENGTH
        and reported != spec.max_sequence_length
    ):
        logger.warning(
            "model_max_length del tokenizer difiere del contexto declarado | %s | "
            "tokenizer=%d declarado=%d",
            spec.model_id,
            reported,
            spec.max_sequence_length,
        )
    return tokenizer


def _load_sentence_transformer(spec: EncoderSpec) -> SentenceTransformer:
    """Carga el modelo completo de `sentence-transformers`, pesos incluidos."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(spec.model_id, trust_remote_code=spec.trust_remote_code)


class EncoderModel:
    """Encoder de un candidato, cargado de forma perezosa.

    Tokenizador y modelo se piden por separado: `count_document_tokens` solo
    dispara la carga del tokenizador, nunca la de los pesos. `_tokenizer` y
    `_model` son la costura de inyeccion para los tests (se asignan directo, sin
    mockear las funciones de carga).
    """

    def __init__(self, spec: EncoderSpec) -> None:
        self.spec = spec
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Tokenizador real del checkpoint, cargado la primera vez que se usa."""
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.spec)
        return self._tokenizer

    @property
    def model(self) -> SentenceTransformer:
        """Modelo completo de `sentence-transformers`. No se usa en el audit de tokens."""
        if self._model is None:
            self._model = _load_sentence_transformer(self.spec)
        return self._model

    def count_query_tokens(self, text: str) -> int:
        """Tokens de una consulta con el formato exacto que veria el modelo."""
        return self._count(self.spec.format_query(text))

    def count_document_tokens(self, text: str) -> int:
        """Tokens de un chunk con el formato exacto que veria el modelo.

        Para E5 esto incluye el prefijo `passage: `: es el `TokenCounter` que
        `src.chunking.pack_units` debe recibir para respetar el presupuesto real.
        """
        return self._count(self.spec.format_document(text))

    def count_document_tokens_batch(self, texts: list[str]) -> list[int]:
        """Version por lote de `count_document_tokens`, para auditar ~172k chunks."""
        formatted = [self.spec.format_document(text) for text in texts]
        encoded = self.tokenizer(formatted, add_special_tokens=True)["input_ids"]
        return [len(ids) for ids in encoded]

    def _count(self, formatted_text: str) -> int:
        return len(self.tokenizer(formatted_text, add_special_tokens=True)["input_ids"])

    def encode_query(self, text: str) -> Any:
        """Embedding de una consulta. No se usa en esta fase (solo tokenizador)."""
        return self.model.encode(
            self.spec.format_query(text),
            normalize_embeddings=self.spec.normalize_embeddings,
        )

    def encode_document(self, text: str) -> Any:
        """Embedding de un chunk. No se usa en esta fase (solo tokenizador)."""
        return self.model.encode(
            self.spec.format_document(text),
            normalize_embeddings=self.spec.normalize_embeddings,
        )

    def as_token_counter(self) -> Any:
        """El `TokenCounter` que `src.chunking.pack_units` espera inyectar."""
        return self.count_document_tokens
