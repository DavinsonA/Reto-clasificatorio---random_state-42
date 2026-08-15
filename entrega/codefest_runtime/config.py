"""Constantes CONGELADAS de la arquitectura productiva. Ninguna es ajustable en evaluacion.

Todas provienen de la fase experimental cerrada (ver `informe_tecnico`) y estan replicadas aqui
para que la entrega no dependa del repositorio de desarrollo. El test de equivalencia del repo
comprueba que estos valores coinciden uno a uno con los de `src/`: si alguien cambia la
arquitectura alli y no aqui, ese test falla.

Compatibilidad: Python 3.9.5+ (el entorno de evaluacion declarado por la especificacion).
"""

from __future__ import annotations

# --- encoder ---------------------------------------------------------------------------------

ENCODER_NAME = "bge-m3"
MODEL_ID = "BAAI/bge-m3"
# Commit SHA concreto, nunca `main`: sin el, dos ejecuciones en fechas distintas podrian usar
# pesos diferentes sin que nada lo delate.
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024
MAX_SEQUENCE_LENGTH = 8192
NORMALIZE_EMBEDDINGS = True
QUERY_PREFIX = ""  # BGE-M3 no lleva prefijo de consulta
TRUST_REMOTE_CODE = False  # BGE-M3 es XLM-R estandar: no carga codigo remoto

# --- indice ----------------------------------------------------------------------------------

INDEX_DIR_NAME = "encoder_bge_m3"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "manifest.json"
EXPECTED_INDEX_TYPE = "IndexFlatIP"  # coseno exacto sobre vectores L2-normalizados

# --- recuperacion ----------------------------------------------------------------------------

CANDIDATE_K = 100
OFFICIAL_FRAGMENTS = 10
OFFICIAL_DOCUMENTS = 3
MAX_WORDS = 250

MATERIALIZATION_POLICY = "best_bge_similarity_adjacent_if_fits"
DOCUMENT_AGGREGATION = "max_pooling_over_legal_fragments"

# --- consultas -------------------------------------------------------------------------------

OFFICIAL_QUERY_COUNT = 50
QUERY_ID_FIELD = "query_id"
QUERY_TEXT_FIELD = "query"

# --- presupuestos de la normalizacion de salida ------------------------------------------------
# Derivados de `FORMAT_AWARE_V2_CONFIG` (chunking congelado). El empaquetador de SALIDA usa
# `target=240, soft_min=72, max=250`, sin solapamiento y sin presupuesto de tokens: es exactamente
# lo que produce `src/chunking/evidence.py::_output_config` sobre la config productiva.
OUTPUT_TARGET_WORDS = 240
OUTPUT_SOFT_MIN_WORDS = 72
OUTPUT_MAX_WORDS = 250

# Separador entre unidades dentro de un chunk. Invertible por construccion del chunker: la
# extraccion colapsa todo `\s+` a un espacio, asi que ninguna unidad contiene un salto de linea.
UNIT_SEPARATOR = "\n"

# Formatos cuya unidad (fila, feature) es atomica: no se parte por `|` ni por columnas.
TABULAR_FORMATS = frozenset({"csv", "xlsx", "pbf"})

# pysbd no trae reglas de portugues; el romance mas cercano disponible es el espanol. Aproximacion
# declarada, identica a la del chunking.
PORTUGUESE_RULESET = "es"
FALLBACK_RULESET = "en"

# Semilla de `langdetect`, para que la deteccion de idioma sea reproducible.
LANGDETECT_SEED = 42

# --- salida ----------------------------------------------------------------------------------

DEFAULT_QUERIES_FILENAME = "consultas.jsonl"
DEFAULT_BASE_VECTORIAL_DIRNAME = "base_vectorial"
DEFAULT_OUTPUT_FILENAME = "resultados.jsonl"

# Ubicacion del modelo empaquetado, si viaja con la entrega (local-first).
LOCAL_MODEL_DIRNAME = "modelos"
LOCAL_MODEL_SUBDIR = "bge_m3"
