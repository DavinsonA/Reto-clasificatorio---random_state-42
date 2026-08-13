# Benchmark de embeddings: BGE-M3 vs GTE multilingual

- **Fecha**: 2026-08-12 (build inicial de bge-m3) — actualizado 2026-08-13 (desbloqueo de GTE)
- **Responsable**: Daniela Castaño (RAG & LLM Engineer)
- **Corpus**: `data/interim/chunking/format_aware_v1.jsonl` (171.780 chunks, baseline ADR-007,
  sin modificar)
- **Codigo**: `src/encoders/benchmark.py` (microbenchmark), `src/encoders/build.py` (build
  completo), `src/encoders/gte_compat.py` (compatibility fix de GTE), `scripts/gte_reference_check.py`
  (comparacion cross-version)

Continua el token audit de `docs/decisions/007-chunking-format-aware.md` (§ Interaccion medida) y
la auditoria de `data/interim/encoder_audit/`. Esta fase genera embeddings reales y dos indices
FAISS. **Ambos estan completos**: bge-m3 (2026-08-12) y gte-multilingual (2026-08-13, tras
desbloquear un blocker de compatibilidad — ver § GTE-multilingual).

## Hardware

| Campo | Valor |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM total | 12.226 MiB |
| Python | 3.13.3 |
| Torch | 2.11.0+cu128 |
| CUDA (build de torch) | 12.8 |
| sentence-transformers | 5.6.1 |
| transformers | 5.14.1 |
| huggingface-hub | 1.24.0 |
| faiss | 1.14.3 |
| `torch.cuda.is_available()` | `True` (gate `hardware.require_cuda` pasa) |

El entorno partia con torch CPU (`uv sync --extra cpu` fue lo ultimo corrido en la fase anterior).
Se re-sincronizo con `uv sync --extra gpu` antes de arrancar; sin ese paso `require_cuda()` frena
la ejecucion explicitamente, sin fallback silencioso a CPU.

## Checkpoints y revision pinning

`EncoderSpec.revision` es obligatoria (`src/encoders/core.py`). Resuelta el 2026-08-12 contra
el Hub con `HfApi().model_info(model_id).sha`:

| Modelo | `model_id` | `revision` (commit SHA) | Dimension | Contexto declarado |
|---|---|---|---|---|
| bge-m3 | `BAAI/bge-m3` | `5617a9f61b028005a4858fdac845db406aefb181` | 1024 | 8192 |
| gte-multilingual | `Alibaba-NLP/gte-multilingual-base` | `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` | 768 | 8192 |
| multilingual-e5-large *(fuera de esta fase)* | `intfloat/multilingual-e5-large` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` | 1024 | 512 |

Tokenizer y `SentenceTransformer` reciben `revision=` explicitamente
(`src/encoders/core.py::_load_tokenizer`, `_load_sentence_transformer`); ninguno usa `main`.

**Desde 2026-08-13, `EncoderSpec` tambien fija `code_revision`** para encoders cuyo
`trust_remote_code=True` carga codigo de un repo *distinto* del checkpoint:

| Modelo | `code_revision` (repo de codigo remoto) |
|---|---|
| gte-multilingual | `40ced75c3017eb27626c9d4ea981bde21a2662f4` (`Alibaba-NLP/new-impl`) |
| bge-m3, multilingual-e5-large | `None` (no usan codigo remoto) |

`SentenceTransformer.max_seq_length` se fuerza al contexto declarado en el registro tras cargar el
modelo (`_load_sentence_transformer`), con warning si difiere de lo que trae el checkpoint. Para GTE
esto es real: el `config.json` del checkpoint declara `max_position_embeddings: 8192`, pero el
**tokenizer** reporta `model_max_length = 32768` (ya detectado en la fase anterior). El contexto
efectivamente usado para truncar es 8192 en ambos casos, no el valor del tokenizer.

## GTE-multilingual: diagnostico y compatibility fix

### Diagnostico inicial (2026-08-12)

**No se genero el indice de GTE.** El modelo se carga sin error, pero **toda** llamada a
`.encode()` — incluso un unico texto corto — falla con un error de indexacion dentro de su codigo
custom (`trust_remote_code=True`, repo `Alibaba-NLP/new-impl`).

Diagnostico de esa sesion (4 pruebas acotadas, todas con el mismo resultado):

| Prueba | Resultado |
|---|---|
| GPU, lote de 20 textos reales, FP32 (antes de tocar FP16) | `torch.AcceleratorError: CUDA error: unknown error`, con asserts de kernel `index out of bounds` en `apply_rotary_pos_emb` / `rotate_half` |
| GPU, un solo texto, dtype auto (FP16, el que trae el `config.json` del checkpoint) | Mismo fallo |
| GPU, un solo texto, forzando `torch_dtype=torch.float32` | Mismo fallo — descarta que sea un problema de precision |
| **CPU**, un solo texto, `attn_implementation` por defecto | `IndexError: index 3707664269312 is out of bounds for dimension 0 with size 12` — reproduce sin CUDA, con un indice claramente corrupto (numero del orden de 10^12) |
| CPU, un solo texto, `attn_implementation="eager"` | Mismo `IndexError` con otro indice igual de absurdo — descarta que sea el path de atencion |

Conclusion de esa sesion: incompatibilidad general `transformers` 5.x / `Alibaba-NLP/new-impl`
(el `config.json` del checkpoint declara `"transformers_version": "4.39.1"`). Se decidio no
downgradear `transformers` en el entorno compartido y dejar GTE bloqueado.

### Diagnostico refinado (2026-08-13)

La causa exacta no era "incompatibilidad general": es un buffer especifico mal inicializado.
Inspeccionando `named_buffers()` del `AutoModel` cargado (`Alibaba-NLP/gte-multilingual-base`,
clase `NewModel` de `Alibaba-NLP/new-impl`):

| Buffer | Esperado | Observado bajo `transformers==5.14.1` |
|---|---|---|
| `embeddings.position_ids` | `[0, 1, 2, ..., 8191]` | `[4403616743424, 0, 0, ..., 0]` — un entero del orden de 10^12 en la posicion 0, ceros despues |
| `embeddings.rotary_emb.inv_freq` | secuencia geometrica de 32 frecuencias, finita | `[36176896.0, 1.44e-42, 0.0, ..., 0.0]` — basura de memoria |
| `embeddings.rotary_emb.cos_cached` / `sin_cached` | tabla `(65536, 64)` de valores trigonometricos | **enteramente en cero** |

Los cuatro son buffers registrados con `persistent=False` en el codigo de `Alibaba-NLP/new-impl`
(`modeling.py`, clases `RotaryEmbedding`/`NTKScalingRotaryEmbedding`/`NewEmbeddings`), calculados
en su `__init__` **solo a partir de `config`** (nunca de pesos entrenados). Como no son
persistentes, no viven en el `state_dict()` del checkpoint. La hipotesis operacional (buffer no
persistente que no sobrevive el `device=meta` de la carga rapida de `transformers>=5`) se confirmo
observando los valores directamente, no se dio por buena a priori.

Se probo tambien `model_kwargs={"low_cpu_mem_usage": False}` (que en teoria evita la carga rapida
sobre `meta`): **no lo corrigio** — los mismos cuatro buffers siguieron en el mismo estado corrupto.
La causa exacta dentro de `transformers` no se investigo mas alla de esto (no era necesario para
resolverlo, ver el fix).

### Compatibility fix

`src/encoders/gte_compat.py`. **No reimplementa la formula de RoPE de Alibaba-NLP/new-impl ni
modifica su `modeling.py`**: construye una instancia fresca de las propias clases del checkpoint
(`type(embeddings.rotary_emb)`, ya cargadas en memoria via `trust_remote_code=True`) fuera del
pipeline de carga de pesos. Como esas clases calculan sus buffers solo desde `config`, una
instancia nueva (no cargada via `from_pretrained`) los tiene correctos — se copian al modulo
cargado. `position_ids` es literalmente la misma linea que usa el propio `NewEmbeddings.__init__`
(`torch.arange`).

Condicion de activacion (`needs_gte_rope_fix`): **solo** `encoder_name == "gte-multilingual"` **y**
`transformers` mayor **>= 5**. Nunca toca bge-m3 ni multilingual-e5-large; nunca se aplica si el
entorno tuviera `transformers` 4.x (el checkpoint no lo necesita ahi). Se loggea siempre que se
aplica (`WARNING`, nivel visible por defecto), nunca en silencio:

```
WARNING src.encoders.gte_compat: Applied GTE transformers>=5 position_ids compatibility fix |
transformers=5.14.1 max_position_embeddings=8192 device=cuda:0
```

La reconstruccion se verifica inmediatamente (`_verify_gte_rope_buffers`) y **falla explicitamente**
(`RuntimeError`) si no cumple: forma y dtype de `position_ids`, secuencia exacta `0..N-1`,
`persistent=False`, finitud de `inv_freq`/`cos_cached`/`sin_cached`, y que `cos_cached` no siga en
cero. Nunca continua con un buffer todavia corrupto.

### Gate 1 — el GTE parcheado codifica (entorno principal, `transformers==5.14.1`)

Via `src/encoders/registry.get_model("gte-multilingual")` (el pipeline real, no un script ad-hoc):

| Configuracion | Resultado |
|---|---|
| CPU, FP32, 1 texto | `(1, 768)` float32, finito, norma 1.0002 |
| CPU, FP32, 3 textos | `(3, 768)` float32, finito, normas 1.00005–1.00008 |
| GPU, FP32, 1 texto | `(1, 768)` float32, finito, norma 1.0003 |
| GPU, FP32, 20 textos reales del corpus | `(20, 768)` float32, finito, normas 0.9998–1.0004 |

**Gate 1: PASS.**

### Gate 2 — equivalencia contra un entorno de referencia aislado (`transformers==4.39.1`)

Entorno temporal via `uv run --no-project --python 3.11 --with "transformers==4.39.1" --with torch
--with huggingface_hub --with numpy --with einops` (Python 3.13 no tiene wheel de `tokenizers`
compatible con esa version de `transformers`; se pidio Python 3.11 solo para este entorno efimero).
**Nunca toco `.venv`, `uv.lock` ni `pyproject.toml`.** Mismo `revision` de checkpoint y mismo
`code_revision` que el entorno principal. Referencia via `AutoModel`/`AutoTokenizer` directos (sin
`sentence-transformers`, no instalado ahi a proposito), respetando el contrato oficial dense de
`1_Pooling/config.json` y `modules.json` del checkpoint: CLS pooling (`last_hidden_state[:, 0]`) +
`2_Normalize` (L2).

Muestra: 80 chunks reales (stride uniforme sobre los 171.780, mismo criterio que
`build._sample_indices` — indices explicitos, deterministas, sin curaduria manual) + 5 consultas
del devset interno. Confirmacion cruzada: `inv_freq` reconstruido por el fix (entorno principal) y
`inv_freq` calculado nativamente en el entorno de referencia son **identicos** en los primeros 5
decimales observados (`[0.93708378, 0.64439130, 0.44311956, 0.30471382, 0.20953830]` en ambos).

`scripts/gte_reference_check.py --mode compare`:

| Metrica | Documentos (80) | Consultas (5) |
|---|---:|---:|
| shapes coinciden | Si | Si |
| finito en ambos lados | Si | Si |
| cosine medio | 1,0001175 | 1,0000834 |
| cosine minimo | 0,9997630 | 0,9997690 |
| diferencia absoluta maxima | 0,0002706 | 0,0001469 |
| diferencia absoluta media | 0,0000294 | 0,0000270 |
| norma media (current / reference) | 1,0001180 / 1,0000000 | 1,0000838 / 1,0000000 |

**Ranking** (documentos de la muestra ordenados por similitud contra cada consulta, `current` vs
`reference`): **orden identico en el top-5 y overlap 10/10 en el top-10, en las 5 consultas.**

**Gate 2: PASS**, sin ambigüedad — cosine ~1.0, diferencias del orden de ruido numerico FP32 entre
versiones de biblioteca (no del fix), ranking preservado exacto.

### Resultado A — fix validado, GTE desbloqueado

Segun los criterios de decision de esta tarea: `.encode()` funciona (Gate 1) + equivalencia
numerica con la referencia (Gate 2) + `code_revision` fijado + tests pasando ⇒ **GTE desbloqueado**.
Se continua con microbenchmark y full build.

## Sanity check de precision (FP32 vs FP16)

`src/encoders/benchmark.py::sanity_check_precision`, 20 chunks reales, cosine entre el embedding
FP32 y el FP16 del mismo texto:

| Modelo | cosine medio | cosine minimo | NaN/Inf | FP16 seguro |
|---|---|---|---|---|
| bge-m3 | 1,00008 | 0,99976 | No | **Si** |
| gte-multilingual | 1,00011 | 0,99958 | No | **Si** |

Cada modelo se evaluo con su propia corrida de `benchmark.py` (una vez GTE dejo de ser un blocker,
ya no hace falta forzar la misma politica de precision para ambos — la comparacion de retrieval es
cosa de la siguiente fase, esto solo mide viabilidad numerica de FP16 por candidato). Ambos:
**FP16** para el computo, salida siempre convertida a `float32` antes de tocar FAISS
(`EncoderModel._encode`).

## Microbenchmark (muestra determinista de 5.000 chunks, primeros N del baseline)

### bge-m3

| batch_size | chunks/s | tokens/s | peak VRAM (MiB) | mean batch (s) | p95 batch (s) | OOM |
|---:|---:|---:|---:|---:|---:|:---:|
| 8 | **147.89** | 39.988 | 1.393 | 0.054 | 0.087 | No |
| 16 | 140.73 | 38.051 | 1.694 | 0.114 | 0.196 | No |
| 32 | 128.36 | 34.707 | 2.296 | 0.248 | 0.486 | No |
| 64 | 114.49 | 30.958 | 3.500 | 0.553 | 1.190 | No |
| 128 | 101.34 | 27.402 | 5.909 | 1.233 | 2.377 | No |

### gte-multilingual

| batch_size | chunks/s | tokens/s | peak VRAM (MiB) | OOM |
|---:|---:|---:|---:|:---:|
| 8 | **338.48** | 91.520 | 1.020 | No |
| 16 | 325.92 | 88.124 | 1.425 | No |
| 32 | 292.91 | 79.199 | 2.234 | No |
| 64 | 260.15 | 70.341 | 3.854 | No |
| 128 | 235.11 | 63.572 | 7.094 | No |

**Mismo hallazgo contraintuitivo en ambos modelos**: el throughput *baja* al subir el batch size.
Sin length bucketing (excluido explicitamente de esta fase), una muestra con longitudes muy
dispares paga mas *padding* desperdiciado cuanto mas grande es el lote. Ningun batch se acerco a
OOM en ninguno de los dos modelos (el mas grande probado, 128, uso 5,9 GiB en bge-m3 y 7,1 GiB en
GTE, de 12,2 GiB disponibles). GTE es ~2,3x mas rapido que bge-m3 al mismo batch — consistente con
ser un modelo mas chico (~305M parametros vs los ~568M de bge-m3).

**Batch seleccionado: 8 para ambos** — mejor throughput y menor VRAM a la vez en los dos casos, sin
tener que elegir entre ambos.

## Full build

| Campo | bge-m3 | gte-multilingual |
|---|---|---|
| Revision | `5617a9f6...` | `9bbca17d...` |
| dtype | float16 (salida float32) | float16 (salida float32) |
| Device | `cuda:0` | `cuda:0` |
| batch_size | 8 | 8 |
| Chunks procesados | 171.780 | 171.780 |
| Truncados (> 8192 tokens) | 18 (0,0105%) | 18 (0,0105%) — mismo tokenizer subyacente (fase anterior: bge-m3/gte tokenizan identico) |
| Tiempo real | 1.459,76 s (24 min 20 s) | 781,80 s (13 min 2 s) |
| Throughput real | 117,68 chunks/s | 219,72 chunks/s |
| Throughput proyectado (microbenchmark) | 147,89 chunks/s → estimado 1.161 s | 338,48 chunks/s → estimado 508 s |
| Desviacion proyeccion vs real | +26% | +54% |
| `index.faiss` | 703.610.925 bytes (671 MiB) | 527.708.205 bytes (503 MiB) — coincide con la aproximacion teorica (`171780 × 768 × 4`) |
| `metadata.jsonl` | 261.315.648 bytes | 261.315.648 bytes (identico: mismo `texto`/`num_tokens`, el tokenizer es el mismo) |

**Por que el tiempo real fue mayor que la proyeccion en ambos casos** (y mas pronunciado en GTE):
el microbenchmark corre sobre los primeros 5.000 chunks, con menos cola larga que el corpus
completo (los `oversized_atomic` — filas de CSV y paginas de PDF de hasta ~18.200 tokens tras el
prefijo — estan distribuidos por todo el corpus, no concentrados al principio). GTE, al ser mas
rapido en la parte "normal" del corpus, sufre proporcionalmente mas cuando llega a esa cola: la
desviacion relativa (+54%) es mayor que la de bge-m3 (+26%) aunque el tiempo absoluto de GTE siga
siendo menor. No es una anomalia, es la consecuencia aritmetica de proyectar desde una muestra sin
la cola completa.

## Integridad

| Chequeo | bge-m3 | gte-multilingual |
|---|---|---|
| `index.ntotal` | 171.780 | 171.780 |
| `metadata.jsonl` (lineas) | 171.780 | 171.780 |
| `ntotal == metadata_rows == chunks del corpus` | Si | Si |
| Dimension del indice | 1024 (esperada: 1024) | 768 (esperada: 768) |
| NaN en muestra (200 ids uniformes) | No | No |
| Inf en muestra | No | No |
| Norma L2 media de la muestra | 1,000098 | 1,000094 |
| Desviacion maxima de norma vs 1.0 | 0,000492 (tolerancia: 0,02, por redondeo FP16) | 0,000480 |
| `faiss.write_index` → `faiss.read_index`: `ntotal` se preserva | Si | Si |
| `faiss.write_index` → `faiss.read_index`: dimension se preserva | Si | Si |
| **Veredicto** | **`ok = true`** | **`ok = true`** |

## Smoke retrieval (prueba estructural, no evaluacion de calidad)

5 consultas del devset interno (`data/interim/benchmarks/prechunk/devset.jsonl`), top-5 cada una,
para los dos indices. Ninguna puntuacion es NaN/Inf, todos los ids caen dentro de rango, todo
`doc_id`/`chunk_id` resuelve contra su `metadata.jsonl` y ningun texto vino vacio. Ejemplo (query
`f1-2`):

| Modelo | score | doc_id | chunk_id |
|---|---:|---|---|
| bge-m3 | 0,6377 | F1-DAIO-032 | F1-DAIO-032\_\_chunk\_000073 |
| bge-m3 | 0,6330 | F2-CSIS-065 | F2-CSIS-065\_\_chunk\_000000 |
| gte-multilingual | 0,8070 | F3-SIPRI-016 | F3-SIPRI-016\_\_chunk\_000142 |
| gte-multilingual | 0,8039 | F3-SIPRI-016 | F3-SIPRI-016\_\_chunk\_000276 |

No se compara contra `relevant_documents` del devset ni entre modelos: eso es evaluacion formal
(NDCG@10/F1@3), explicitamente fuera de esta fase. Las puntuaciones de GTE son sistematicamente mas
altas que las de bge-m3 (rango tipico 0,75–0,87 vs 0,5–0,66) — es una diferencia de escala entre
modelos, no evidencia de calidad; no es comparable sin normalizar y esta fuera del alcance de un
smoke test estructural.

## Artefactos

```
data/interim/encoder_benchmark/microbenchmark.json               # microbenchmark bge-m3 (gitignored)
data/interim/encoder_benchmark_gte/microbenchmark.json            # microbenchmark gte-multilingual
data/interim/encoder_benchmark/gte_current.json                   # embeddings Gate 2, entorno principal
data/interim/encoder_benchmark/gte_reference.json                 # embeddings Gate 2, entorno aislado
data/interim/encoder_benchmark/gte_reference_comparison.json      # metricas de equivalencia Gate 2
data/interim/faiss_experimental/encoder_bge_m3/
    index.faiss                                              # IndexFlatIP, 171.780 x 1024, L2-normalizado
    metadata.jsonl                                            # Tabla 1, 171.780 lineas, orden == ids FAISS
    build_report.json                                         # metricas + integridad + smoke retrieval
data/interim/faiss_experimental/encoder_gte_multilingual/
    index.faiss                                              # IndexFlatIP, 171.780 x 768, L2-normalizado
    metadata.jsonl
    build_report.json
```

Ubicacion intermedia deliberada (`data/interim/faiss_experimental/`, no `entrega/base_vectorial/`):
el encoder ganador todavia no esta decidido (falta NDCG@10/F1@3). Materializar en
`entrega/base_vectorial/encoder_<nombre>/` es una decision de la siguiente fase.

## Riesgos y deuda abierta

1. **La causa exacta dentro de `transformers` que deja estos buffers sin inicializar no se
   investigo mas alla de confirmarla y esquivarla.** El fix es correcto y verificado (Gate 1 + Gate
   2), pero si `transformers` cambia de nuevo su pipeline de carga, o si `Alibaba-NLP/new-impl`
   actualiza su codigo, hay que re-verificar `needs_gte_rope_fix`/`fix_gte_rope_buffers` contra la
   nueva combinacion antes de asumir que sigue haciendo falta (o que sigue siendo suficiente).
2. **El microbenchmark subestima el tiempo real** (+26% bge-m3, +54% gte-multilingual) por no
   incluir la cola completa de chunks `oversized_atomic` en la muestra de los primeros 5.000. Para
   proyecciones futuras, usar una muestra estratificada o el corpus completo si el tiempo lo
   permite.
3. **`multilingual-e5-large` sigue fuera de esta fase** (excluido explicitamente): su
   incompatibilidad de contexto (512 tokens, 4,90% over-context medido en la fase anterior) no se
   toco aqui.
4. **No se genero embedding de las 50 consultas oficiales** (q001-q050): esta fase solo corrio el
   devset interno de 5 preguntas como smoke test estructural, para los dos indices.
5. **Las puntuaciones de similitud de GTE y bge-m3 no son comparables directamente** (escalas
   distintas observadas en el smoke retrieval): cualquier fusion futura (RRF, CombSUM/CombMNZ)
   tiene que normalizar antes de combinar puntuaciones de ambos.
