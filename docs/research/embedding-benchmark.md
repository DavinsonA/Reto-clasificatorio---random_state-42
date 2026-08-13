# Benchmark de embeddings: BGE-M3 vs GTE multilingual

- **Fecha**: 2026-08-12
- **Responsable**: Daniela Castaño (RAG & LLM Engineer)
- **Corpus**: `data/interim/chunking/format_aware_v1.jsonl` (171.780 chunks, baseline ADR-007,
  sin modificar)
- **Codigo**: `src/encoders/benchmark.py` (microbenchmark), `src/encoders/build.py` (build completo)

Continua el token audit de `docs/decisions/007-chunking-format-aware.md` (§ Interaccion medida) y
la auditoria de `data/interim/encoder_audit/`. Esta fase genera embeddings reales y dos indices
FAISS — solo uno se completo; ver § GTE-multilingual: bloqueante.

## Hardware

| Campo | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM total | 12.226 MiB |
| Torch | 2.11.0+cu128 |
| CUDA (build de torch) | 12.8 |
| `torch.cuda.is_available()` | `True` (gate `hardware.require_cuda` pasa) |

El entorno partia con torch CPU (`uv sync --extra cpu` fue lo ultimo corrido en la fase anterior).
Se re-sincronizo con `uv sync --extra gpu` antes de arrancar; sin ese paso `require_cuda()` frena
la ejecucion explicitamente, sin fallback silencioso a CPU.

## Checkpoints y revision pinning

`EncoderSpec.revision` ahora es obligatoria (`src/encoders/core.py`). Resuelta el 2026-08-12 contra
el Hub con `HfApi().model_info(model_id).sha`:

| Modelo | `model_id` | `revision` (commit SHA) | Dimension | Contexto declarado |
|---|---|---|---|---|
| bge-m3 | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | 1024 | 8192 |
| gte-multilingual | `Alibaba-NLP/gte-multilingual-base` | `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` | 768 | 8192 |
| multilingual-e5-large *(fuera de esta fase)* | `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | 1024 | 512 |

Tokenizer y `SentenceTransformer` reciben `revision=` explicitamente
(`src/encoders/core.py::_load_tokenizer`, `_load_sentence_transformer`); ninguno usa `main`.

`SentenceTransformer.max_seq_length` se fuerza al contexto declarado en el registro tras cargar el
modelo (`_load_sentence_transformer`), con warning si difiere de lo que trae el checkpoint. Para GTE
esto es real: el `config.json` del checkpoint declara `max_position_embeddings: 8192`, pero el
**tokenizer** reporta `model_max_length = 32768` (ya detectado en la fase anterior). El contexto
efectivamente usado para truncar es 8192 en ambos casos, no el valor del tokenizer.

## GTE-multilingual: bloqueante

**No se genero el indice de GTE.** El modelo se carga sin error, pero **toda** llamada a
`.encode()` — incluso un unico texto corto — falla con un error de indexacion dentro de su codigo
custom (`trust_remote_code=True`, repo `Alibaba-NLP/new-impl`).

Diagnostico (4 pruebas acotadas, todas con el mismo resultado):

| Prueba | Resultado |
|---|---|
| GPU, lote de 20 textos reales, FP32 (antes de tocar FP16) | `torch.AcceleratorError: CUDA error: unknown error`, con asserts de kernel `index out of bounds` en `apply_rotary_pos_emb` / `rotate_half` |
| GPU, un solo texto, dtype auto (FP16, el que trae el `config.json` del checkpoint) | Mismo fallo |
| GPU, un solo texto, forzando `torch_dtype=torch.float32` | Mismo fallo — descarta que sea un problema de precision |
| **CPU**, un solo texto, `attn_implementation` por defecto | `IndexError: index 3707664269312 is out of bounds for dimension 0 with size 12` — reproduce sin CUDA, con un indice claramente corrupto (numero del orden de 10^12) |
| CPU, un solo texto, `attn_implementation="eager"` | Mismo `IndexError` con otro indice igual de absurdo — descarta que sea el path de atencion |

**Causa raiz identificada**: el `config.json` del checkpoint declara
`"transformers_version": "4.39.1"`; el entorno del proyecto tiene `transformers==5.14.1` instalado
(major version distinta). El codigo remoto de `Alibaba-NLP/new-impl` (RoPE con `rope_scaling`
`ntk` factor 8.0, `pack_qkv=True`) no es compatible con la API interna de `transformers` 5.x — el
fallo reproduce identico en GPU y CPU, en FP16 y FP32, con distintas implementaciones de atencion,
lo que descarta hardware, precision y configuracion de atencion como causa.

**Decision** (confirmada con el equipo): no downgradear `transformers` en el entorno compartido
para intentar arreglarlo — el blast radius alcanzaria chunking y el audit de tokens, que ya
dependen de la version actual y estan verificados. Se entrega **solo el indice de bge-m3** en esta
fase; GTE queda documentado como bloqueado, no descartado. Vias de seguimiento no exploradas
todavia: version fijada de `transformers` en un entorno aislado solo para GTE, o esperar una
actualizacion del codigo remoto en el Hub.

## Sanity check de precision (FP32 vs FP16)

`src/encoders/benchmark.py::sanity_check_precision`, 20 chunks reales, cosine entre el embedding
FP32 y el FP16 del mismo texto:

| Modelo | cosine medio | cosine minimo | NaN/Inf | FP16 seguro |
|---|---|---|---|---|
| bge-m3 | 1.00008 | 0.99976 | No | **Si** |

(GTE no llego a esta etapa: el fallo ocurre en la primera llamada a `.encode()`, antes de convertir
a FP16.)

Con un solo candidato viable, la politica "misma precision para todos" se decide trivialmente:
**FP16** para bge-m3, salida siempre convertida a `float32` antes de tocar FAISS
(`EncoderModel._encode`).

## Microbenchmark (muestra determinista de 5.000 chunks, primeros N del baseline)

| batch_size | chunks/s | tokens/s | peak VRAM (MiB) | mean batch (s) | p95 batch (s) | OOM |
|---:|---:|---:|---:|---:|---:|:---:|
| 8 | **147.89** | 39.988 | 1.393 | 0.054 | 0.087 | No |
| 16 | 140.73 | 38.051 | 1.694 | 0.114 | 0.196 | No |
| 32 | 128.36 | 34.707 | 2.296 | 0.248 | 0.486 | No |
| 64 | 114.49 | 30.958 | 3.500 | 0.553 | 1.190 | No |
| 128 | 101.34 | 27.402 | 5.909 | 1.233 | 2.377 | No |

**Hallazgo contraintuitivo**: el throughput *baja* al subir el batch size, en vez de subir. Sin
length bucketing (excluido explicitamente de esta fase), una muestra con longitudes muy dispares
paga mas *padding* desperdiciado cuanto mas grande es el lote — el modelo (568M parametros) no es
lo bastante pesado para que el paralelismo extra compense ese desperdicio. Ningun batch se acerco a
OOM (el mas grande, 128, uso 5.9 GiB de 12.2 GiB disponibles).

**Batch seleccionado: 8** — mejor throughput y menor VRAM a la vez, sin tener que elegir entre
ambos.

## Full build

| Campo | Valor |
|---|---|
| Modelo | bge-m3 |
| Revision | `5617a9f61b028005a4858fdac845db406aefb181` |
| dtype | float16 (salida siempre float32) |
| Device | `cuda:0` |
| batch_size | 8 |
| Chunks procesados | 171.780 |
| Truncados (> 8192 tokens) | 18 (0,0105%) — coincide con lo medido en el audit de tokens |
| Tiempo real | 1.459,76 s (24 min 20 s) |
| Throughput real | 117,68 chunks/s |
| Throughput proyectado (microbenchmark) | 147,89 chunks/s → estimado 1.161 s |
| Desviacion proyeccion vs real | +26% de tiempo real sobre lo proyectado |
| `index.faiss` | 703.610.925 bytes (671 MiB) — coincide con la aproximacion teorica (`171780 × 1024 × 4`) |
| `metadata.jsonl` | 261.315.648 bytes (249 MiB) |

**Por que el tiempo real fue mayor que la proyeccion**: el microbenchmark corre sobre los primeros
5.000 chunks, con menos cola larga que el corpus completo (los `oversized_atomic` — filas de CSV y
paginas de PDF de hasta 18.191 tokens tras el prefijo — estan distribuidos por todo el corpus, no
concentrados al principio). Procesarlos empuja el promedio real por debajo del proyectado. Es una
diferencia esperable, no una anomalia: la proyeccion es una cota optimista por diseño (usa
`chunks_per_second` de una muestra sin la cola completa).

## Integridad

| Chequeo | Resultado |
|---|---|
| `index.ntotal` | 171.780 |
| `metadata.jsonl` (lineas) | 171.780 |
| `ntotal == metadata_rows == chunks del corpus` | Si |
| Dimension del indice | 1024 (esperada: 1024) |
| NaN en muestra (200 ids uniformes) | No |
| Inf en muestra | No |
| Norma L2 media de la muestra | 1,000098 |
| Desviacion maxima de norma vs 1.0 | 0,000492 (tolerancia: 0,02, por redondeo FP16) |
| `faiss.write_index` → `faiss.read_index`: `ntotal` se preserva | Si |
| `faiss.write_index` → `faiss.read_index`: dimension se preserva | Si |
| **Veredicto** | **`ok = true`** |

## Smoke retrieval (prueba estructural, no evaluacion de calidad)

5 consultas del devset interno (`data/interim/benchmarks/prechunk/devset.jsonl`), top-5 cada una.
Ninguna puntuacion es NaN/Inf, todos los ids caen dentro de rango, todo `doc_id`/`chunk_id` resuelve
contra `metadata.jsonl` y ningun texto vino vacio. Ejemplo (query `f1-2`):

| score | doc_id | chunk_id |
|---:|---|---|
| 0,6377 | F1-DAIO-032 | F1-DAIO-032\_\_chunk\_000073 |
| 0,6330 | F2-CSIS-065 | F2-CSIS-065\_\_chunk\_000000 |
| 0,6302 | F2-CSIS-019 | F2-CSIS-019\_\_chunk\_000006 |

No se compara contra `relevant_documents` del devset: eso es evaluacion formal (NDCG@10/F1@3),
explicitamente fuera de esta fase.

## Artefactos

```
data/interim/encoder_benchmark/microbenchmark.json          # microbenchmark completo (gitignored)
data/interim/faiss_experimental/encoder_bge_m3/
    index.faiss                                              # IndexFlatIP, 171.780 x 1024, L2-normalizado
    metadata.jsonl                                            # Tabla 1, 171.780 lineas, orden == ids FAISS
    build_report.json                                         # metricas + integridad + smoke retrieval
```

Ubicacion intermedia deliberada (`data/interim/faiss_experimental/`, no `entrega/base_vectorial/`):
el encoder ganador no esta decidido (falta NDCG@10/F1@3 y, cuando GTE se desbloquee, su propio
build). Materializar en `entrega/base_vectorial/encoder_<nombre>/` es una decision de la siguiente
fase.

## Riesgos y deuda abierta

1. **GTE-multilingual sigue sin evaluar.** Bloqueante documentado arriba. La comparacion BGE vs GTE
   de la fase anterior (tokenizacion identica, ver `docs/decisions/007-chunking-format-aware.md` y
   memoria `encoder-token-audit-2026-08-11`) sigue siendo valida para tokens, pero no hay todavia
   evidencia de embeddings ni de retrieval para GTE.
2. **El microbenchmark subestima el tiempo real en ~26%** por no incluir la cola de chunks
   `oversized_atomic` en la muestra de los primeros 5.000. Para proyecciones futuras, usar una
   muestra estratificada o el mismo corpus completo si el tiempo lo permite.
3. **`multilingual-e5-large` sigue fuera de esta fase** (excluido explicitamente): su incompatibilidad
   de contexto (512 tokens, 4,90% over-context medido en la fase anterior) no se toco aqui.
4. **No se genero embedding de las 50 consultas oficiales** (q001-q050): esta fase solo corrio el
   devset interno de 5 preguntas como smoke test estructural.
