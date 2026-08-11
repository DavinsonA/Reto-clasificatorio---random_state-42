# Bitácora de fuentes — investigación de chunking

**Fecha:** 2026-08-10 · **Branch:** `dev` · **HEAD:** `b75993b`
**Informe que las consume:** [`chunking-best-practices-codefest.md`](chunking-best-practices-codefest.md)

Registro de las fuentes externas revisadas para diseñar la batería de ablaciones de chunking.
Resumidas, no copiadas. **Ninguna entrada de esta lista es una decisión del proyecto**: una
fuente puede describir una práctica excelente en RAG general y ser inadmisible bajo la §8.3 de
CODEFEST (prohibición de decoders), o admisible pero irrelevante para este corpus.

Convención de la columna *Relación con CODEFEST*:

- **Admisible** — la técnica no usa modelos generativos en indexación ni recuperación.
- **No admisible** — requiere un decoder en el pipeline; descalifica (§2.1 de `CLAUDE.md`).
- **Admisible con reservas** — la técnica es legal pero cambia una fase distinta del
  experimento (encoder, pooling, agregación), no el chunking puro.

---

## 1. Evidencia empírica comparativa (el núcleo)

### 1.1 Is Semantic Chunking Worth the Computational Cost?

- **Autores:** Renyi Qu, Ruixuan Tu, Forrest Bao · **Año:** 2024 (arXiv, oct.)
- **URL:** <https://arxiv.org/abs/2410.13070>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** fixed-size (por número de oraciones, con y sin solapamiento de 1
  oración) vs. semantic breakpoint-based (4 umbrales relativos + 2 absolutos) vs. semantic
  clustering-based (aglomerativo *single-linkage* y DBSCAN, con distancia híbrida
  posicional+semántica).
- **Hallazgo relevante:** sobre 10 datasets de recuperación documental, 5 de RAGBench para
  recuperación de evidencia, y generación de respuesta. En recuperación de **evidencia** —el
  caso más cercano al nuestro— fixed-size gana en 3 de 5 datasets y las diferencias son
  mínimas (ExpertQA F1@5: fixed 47,11 % vs. breakpoint 47,08 % vs. clustering 46,87 %). Donde
  el semantic sí gana claramente es en los datasets *stitched* (documentos sintéticos cosidos
  a partir de BEIR, p. ej. Miracl 81,89 % vs. 69,45 %). Conclusión de los autores: *"los
  costes computacionales del semantic chunking no se justifican con ganancias consistentes"*.
- **Limitaciones:** los propios autores señalan que los documentos *stitched* tienen diversidad
  temática artificialmente alta (por eso el semantic brilla ahí), que no existe *ground truth*
  de calidad de chunk, y que los embeddings a nivel oración son "context-free".
- **Relación con CODEFEST:** **admisible**. Es la evidencia más fuerte a favor de no empezar
  por semantic chunking. El matiz de los documentos *stitched* importa: nuestro corpus **no**
  es sintético ni cosido.

### 1.2 Evaluating Chunking Strategies for Retrieval (Chroma Technical Report)

- **Autores/proyecto:** Chroma (Brandon Smith, Anton Troynikov) · **Año:** 2024 (3 julio)
- **URL:** <https://www.trychroma.com/research/evaluating-chunking>
- **Tipo:** reporte técnico de proyecto, con experimento reproducible y código publicado
- **Estrategia estudiada:** `RecursiveCharacterTextSplitter` y `TokenTextSplitter` (200–800
  tokens, solapamiento 0–400), `KamradtSemanticChunker`, `KamradtModifiedChunker`,
  `ClusterSemanticChunker` (200/400), `LLMSemanticChunker`.
- **Hallazgo relevante:** evalúa **a nivel de token**, no de chunk: recall, precisión,
  Precisión_Ω (techo de precisión bajo recall perfecto) e IoU. Resultados: (a) los chunks
  **pequeños suben mucho la precisión** — 200 tokens da 7,0 ± 5,6 de precisión frente a
  1,5 ± 1,3 con 800 tokens; (b) el **solapamiento castiga el IoU** por tokens redundantes;
  (c) el default documentado de 800 tokens con 400 de solapamiento produce *"las peores
  puntuaciones en el resto de métricas"*; (d) `ClusterSemanticChunker` a 200 tokens gana en
  precisión (8,0 ± 6,0) e IoU, y a 400 queda segundo en recall (91,3 %).
- **Limitaciones:** las consultas y los pasajes relevantes se **generan y filtran con un LLM**
  (el ground truth es sintético); 5 corpus en inglés únicamente; el LLMSemanticChunker
  depende de un decoder.
- **Relación con CODEFEST:** **admisible** como evidencia (nosotros no reproducimos su
  generación de queries). Es la fuente que más se parece a cómo nos evalúan: precisión sobre
  **texto recuperado**, no sobre identificadores de chunk.

### 1.3 A Systematic Investigation of Document Chunking Strategies and Embedding Sensitivity

- **Año:** 2026 · **URL:** <https://arxiv.org/abs/2603.06976>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** 36 métodos de segmentación (fixed-size por carácter y por token,
  oración, grupo de párrafos, recursivo, semántico, *structure-aware*, jerárquico, adaptativo,
  late chunking, asistido por LLM) × 6 dominios (biología, física, salud, legal, matemáticas,
  agricultura) × 5 encoders (BGE-M3, all-MiniLM-L6-v2, tres variantes POTION).
- **Hallazgo relevante:** **Paragraph Group Chunking** es la mejor estrategia global
  (nDCG@5 ≈ 0,459; P@1 ≈ 24 %), muy por encima del fixed-size por carácter
  (nDCG@5 < 0,244; P@1 ≈ 2–3 %). *Dynamic token size* domina en biología/física/salud; el
  agrupamiento de párrafos domina en legal y matemáticas. Las jerarquías de rendimiento se
  mantienen entre encoders, pero la sensibilidad al chunking varía por modelo.
- **Limitaciones:** los juicios de relevancia los produce **un LLM**, no anotadores humanos;
  subconjunto fijo de dominios de UltraDomain; los autores reconocen que el chunking cambia el
  número de chunks y eso confunde el efecto semántico.
- **Relación con CODEFEST:** **admisible**. Es la evidencia positiva más fuerte a favor de
  respetar fronteras de párrafo en vez de cortar por carácter.

### 1.4 A Systematic Analysis of Chunking Strategies for Reliable Question Answering

- **Año:** 2026 (enero) · **URL:** <https://arxiv.org/abs/2601.14123>
- **Tipo:** paper (preprint arXiv; también en Springer LNCS)
- **Estrategia estudiada:** chunking por token / oración / semántico / código, tamaños de 50 a
  500 tokens en pasos de 50, solapamiento 0 % o 20 %, sobre Natural Questions con retriever
  SPLADE.
- **Hallazgo relevante:** **el solapamiento del 10–20 % no mejora ni BERTScore ni Exact
  Match**, con diferencias dentro del margen estadístico, mientras infla el número de chunks
  por un factor `1/(1−r)`. Los autores recomiendan explícitamente solapamiento **0 %**. El
  chunking por oración y el semántico rinden parecido entre sí y por encima del chunking por
  token puro. Para QA general, 150–300 tokens equilibra recall y abstención.
- **Limitaciones:** un solo dataset (NQ, inglés, Wikipedia), un solo generador
  (Ministral-8B), sin rerankers ni modelos de interacción tardía; los autores advierten que no
  generaliza a dominios especializados.
- **Relación con CODEFEST:** **admisible**. Es la fuente principal para tratar
  `overlap = 0` como el baseline obligatorio y no como una carencia.

### 1.5 Chunking Methods on RAG — Effectiveness Evaluation Against Computational Cost

- **Año:** 2026 · **URL:** <https://arxiv.org/abs/2606.00881>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** fixed-size, TextTiling, recursive character, clustering (HAC
  secuencial, Max-Min), GraphSeg, LumberChunker (LLM), DenseX; 9 datasets (GutenQA,
  LiteraryQA, NovelQA, Qasper, SQuAD, PoQuAD, TriviaQA, NQ…).
- **Hallazgo relevante:** fixed-size logra 87,71 de Accuracy@5 en **menos de 1 segundo** de
  procesamiento; recursive semantic sube a 89,36 con ~4,9 minutos; DenseX necesita **más de 15
  horas** de media y LumberChunker no terminó en varios datasets bajo un límite de 48 h.
  Conclusión: *"los métodos de chunking computacionalmente más caros no producen mejoras de
  efectividad significativas mientras introducen un coste sustancialmente mayor"*.
- **Limitaciones:** parte de la evaluación es *LLM-as-judge* (escala Likert 1–5); varios
  métodos caros quedaron sin resultado por *timeout*, lo que sesga la comparación.
- **Relación con CODEFEST:** **admisible**. Refuerza priorizar métodos baratos primero.
  DenseX y LumberChunker son además **no admisibles** por usar decoders.

### 1.6 Rethinking Chunk Size For Long-Document Retrieval: A Multi-Dataset Analysis

- **Autores:** Bhat, Rudat et al. · **Año:** 2025 (mayo) · **URL:** <https://arxiv.org/abs/2505.21700>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** fixed-size con varios tamaños, sobre datasets de respuesta corta y
  de comprensión amplia, con varios encoders.
- **Hallazgo relevante:** **no existe un tamaño óptimo universal**. Los chunks pequeños
  (64–128 tokens) ganan cuando la respuesta es un hecho concreto; los grandes (512–1024)
  ganan cuando la consulta requiere contexto amplio. Además los encoders tienen
  sensibilidades distintas: Stella se beneficia de chunks grandes, Snowflake de pequeños.
- **Limitaciones:** solo fixed-size; datasets en inglés.
- **Relación con CODEFEST:** **admisible**, y es la justificación técnica de por qué **no se
  puede fijar el tamaño antes de elegir el encoder** ni copiarlo de un blog.

---

## 2. Contexto añadido al chunk

### 2.1 Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models

- **Autores:** Michael Günther, Isabelle Mohr, Daniel J. Williams, Bo Wang, Han Xiao (Jina AI)
- **Año:** 2024, v3 de julio 2025 · **URL:** <https://arxiv.org/abs/2409.04701>
  (divulgación: <https://jina.ai/news/late-chunking-in-long-context-embedding-models/>)
- **Tipo:** paper + documentación de proyecto
- **Estrategia estudiada:** *late chunking*: tokenizar el documento completo, pasarlo entero
  por el transformer, y **solo después** aplicar las fronteras de chunk sobre la secuencia de
  embeddings de token, haciendo *mean pooling* por segmento. Ocurre en **indexación**, no en
  consulta; la consulta se embebe igual que siempre.
- **Hallazgo relevante:** sobre BeIR (SciFact, NFCorpus, FiQA, TRECCOVID) con
  jina-embeddings-v2-small/v3 y nomic-embed-text-v1: nDCG@10 medio 52,2 → 54,0 con fronteras
  fijas de 256 tokens (+1,8), 52,4 → 54,3 con fronteras de 5 oraciones (+1,9), 52,4 → 53,8
  con fronteras semánticas (+1,5). La variante *long late chunking* trocea en "macro chunks"
  solapados para documentos que exceden el contexto del modelo.
- **Limitaciones:** **exige un encoder de contexto largo con mean pooling** (no sirve con
  pooling CLS); la memoria crece muy rápido con la longitud, lo que hace inviable el paso
  único en documentos muy largos; **no ayuda** cuando el contexto añadido es irrelevante
  (datasets sintéticos Needle-8192 y Passkey-8192) y el naive gana a veces en tareas de
  comprensión lectora con chunks de 512+ tokens.
- **Relación con CODEFEST:** **admisible con reservas**. No usa decoder (es un encoder), así
  que es legal. Pero cambia el *encoder* y el *pooling*, no la política de fronteras: mezclarlo
  con la primera ablación de chunking confundiría dos variables.

### 2.2 Contextual Retrieval (Anthropic)

- **Proyecto:** Anthropic · **Año:** 2024 (sept.)
- **URL:** <https://www.anthropic.com/engineering/contextual-retrieval>
- **Tipo:** documentación técnica de proveedor con números propios
- **Estrategia estudiada:** anteponer a cada chunk 50–100 tokens de contexto **generados por
  un LLM** a partir del documento completo, antes de embeber (*contextual embeddings*) y antes
  de indexar en BM25 (*contextual BM25*).
- **Hallazgo relevante:** reducción de fallos de recuperación del 35 % solo con contextual
  embeddings (5,7 % → 3,7 %), 49 % combinado con contextual BM25 (5,7 % → 2,9 %), y 67 %
  añadiendo reranking.
- **Limitaciones:** medición interna del propio proveedor; el coste de generación escala con
  el número de chunks; se han descrito fallos cuando el prefijo generado es una paráfrasis
  casi literal del chunk y no aporta contexto nuevo.
- **Relación con CODEFEST:** **NO ADMISIBLE.** Requiere un decoder generativo dentro de la
  indexación. Es exactamente el caso que §8.3 prohíbe. Se registra para poder decir en el
  informe técnico por qué se descartó una técnica conocidamente efectiva, y para separarla del
  **contexto determinístico** (título, encabezado, hoja), que sí es admisible.

### 2.3 Reconstructing Context: Evaluating Advanced Chunking Strategies for RAG

- **Autores:** Carlo Merola, Jaspinder Singh · **Año:** 2025 (abril)
- **URL:** <https://arxiv.org/abs/2504.19754>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** comparación directa de *late chunking* y *contextual retrieval*
  frente a *early chunking* + recuperación tradicional, sobre NFCorpus y MSMarco, con Jina V3,
  Jina ColBERT V2, Stella V5 y BGE-M3.
- **Hallazgo relevante:** **ninguna de las dos técnicas avanzadas es una solución
  definitiva**. Late chunking a veces supera y a veces queda por debajo del early chunking
  según el encoder; contextual retrieval preserva mejor la coherencia semántica pero exige
  ~20 GB de VRAM para contextualizar los chunks. Las ganancias en NDCG@5 sobre NFCorpus con
  Jina-V3 son marginales (0,303–0,312 tradicional vs. 0,317 con fusión + reranking).
- **Limitaciones:** por restricciones de GPU trabajaron con ~5.000 documentos y 1.000
  consultas (RQ#1) y con el 20 % de NFCorpus, ~300 documentos y 50 consultas (RQ#2). Muestras
  muy pequeñas: los propios autores lo señalan.
- **Relación con CODEFEST:** **admisible** como evidencia. Es la fuente que mejor sostiene
  "no adoptar late chunking por reputación": es el único trabajo que lo mide contra el
  baseline con varios encoders y encuentra resultados mixtos.

---

## 3. Granularidad y unidad de recuperación

### 3.1 Dense X Retrieval: What Retrieval Granularity Should We Use?

- **Autores:** Tong Chen et al. · **Año:** 2023 (arXiv), EMNLP 2024
- **URL:** <https://arxiv.org/abs/2312.06648> · <https://aclanthology.org/2024.emnlp-main.845/>
- **Tipo:** paper (conferencia, EMNLP main)
- **Estrategia estudiada:** compara tres unidades de indexación sobre la misma Wikipedia
  (FactoidWiki): pasajes de 100 palabras, oraciones, y *proposiciones* (expresiones atómicas
  autocontenidas).
- **Hallazgo relevante:** **la unidad de indexación es una decisión de primer orden**.
  Indexar por proposiciones supera claramente a indexar por pasajes: +50–55 % de EM@100 con
  retrievers no supervisados y +19–26 % con supervisados.
- **Limitaciones:** las proposiciones se generan con un **LLM propositionizer**; solo inglés;
  Wikipedia.
- **Relación con CODEFEST:** el **hallazgo** (granularidad fina ayuda a la recuperación) es
  admisible y transferible; el **método** (generar proposiciones con un decoder) es **NO
  ADMISIBLE**. La lectura útil para nosotros es que las unidades pequeñas se recuperan mejor,
  lo que empuja hacia arquitecturas *retrieval unit ≠ evidence unit*.

### 3.2 Sentence Window Retrieval / Auto-Merging (Parent-Child) Retriever — LlamaIndex

- **Proyecto:** LlamaIndex · **Año:** 2023–2026 (documentación viva)
- **URL:** <https://docs.llamaindex.ai/en/latest/examples/retrievers/auto_merging_retriever/>
- **Tipo:** documentación técnica oficial de proyecto
- **Estrategia estudiada:** (a) *sentence window*: se embebe una oración y se devuelve una
  ventana de N oraciones a su alrededor (por defecto 5 a cada lado); (b) *auto-merging /
  small-to-big*: `HierarchicalNodeParser` produce una jerarquía padre-hijo; se recuperan los
  hijos y, si suficientes hijos del mismo padre entran en el top-k, se devuelve el padre.
- **Hallazgo relevante:** patrón consolidado de que la unidad **embebida** puede ser más
  pequeña que la unidad **devuelta**. La documentación describe el mecanismo; no aporta
  benchmark propio.
- **Limitaciones:** es documentación de producto, sin evaluación controlada. Cualquier
  ganancia hay que medirla en el propio corpus.
- **Relación con CODEFEST:** **admisible** y directamente aplicable: la §9.2.1 de la
  especificación autoriza exactamente esta separación (concatenar el vecino inmediato del
  mismo documento hasta 250 palabras).

### 3.3 Lost in a Single Vector: Improving Long-Document Retrieval with Chunk Evidence Aggregation

- **Autores:** Shanshan Lyu, Yiwei Wang, Yujun Cai, Jiafeng Guo, Shenghua Liu
- **Año:** 2026 (junio) · **URL:** <https://arxiv.org/abs/2606.18781>
- **Tipo:** paper (preprint arXiv), código publicado
- **Estrategia estudiada:** DICE — trocear el documento, embeber cada chunk por separado con
  el modelo congelado, y **agregar** los vectores de chunk en un único vector de documento.
  Introducen el *Evidence Dilution Index* (EDI): cuánto cae la representación a nivel
  documento por debajo de su mejor chunk.
- **Hallazgo relevante:** en LongEmbed, mejoras grandes por encima de 4k tokens (Passkey >4k:
  30,0 → 90,0; Needle >4k: 23,3 → 74,0), y EDI menor que el baseline de vector único en el
  92,8 % de 12.779 muestras. La evidencia decisiva de un documento largo **se diluye** cuando
  se comprime en una sola representación.
- **Limitaciones:** parte de la evaluación es sobre tareas sintéticas tipo *needle in a
  haystack*; el objetivo es el ranking de documentos, no de fragmentos.
- **Relación con CODEFEST:** **admisible**, y directamente relevante para **F1@3**: sostiene
  que la agregación a documento debe partir de puntuaciones de chunk (max-pooling o similar,
  que es lo que la §8.6 ya autoriza) y no de un vector único por documento.

---

## 4. Estructura y tablas

### 4.1 Structure-Aware Chunking for Tabular Data in RAG (STC)

- **Año:** 2026 · **URL:** <https://arxiv.org/abs/2605.00318>
- **Tipo:** paper (preprint arXiv)
- **Estrategia estudiada:** unidades a **nivel de fila**, con un *Row Tree* donde cada fila se
  codifica como bloque clave-valor; división con restricción de tokens alineada a fronteras
  estructurales, y fusión *greedy* sin solapamiento.
- **Hallazgo relevante:** reduce el número de chunks hasta un 40 % frente a recursive y un
  56 % frente a baselines clave-valor; mejora MRR de 0,3576 a 0,5945 en modo híbrido y
  Recall@1 de 0,366 a 0,754 en BM25.
- **Limitaciones:** dominio tabular específico; la ganancia mayor se mide sobre BM25, no sobre
  recuperación densa pura.
- **Relación con CODEFEST:** **admisible**. Valida el diseño que nuestros parsers de CSV/XLSX
  ya implementan (`columna: valor | columna: valor` por fila) y sugiere que empaquetar filas
  consecutivas hasta un presupuesto de tokens es una variante a medir.

### 4.2 Evaluation of Table Representations to Answer Questions from Tables in Documents

- **Año:** 2024 · **URL:** <https://arxiv.org/abs/2408.17008>
- **Tipo:** paper (preprint arXiv, caso de estudio sobre especificaciones 3GPP)
- **Estrategia estudiada:** representaciones de tabla y granularidad (tabla completa vs. fila),
  con y sin repetición de la cabecera en cada unidad.
- **Hallazgo relevante:** el chunking **a nivel de fila supera al de tabla completa**, y
  repetir la cabecera en cada fila mejora la exactitud de recuperación frente a no repetirla.
- **Limitaciones:** un solo dominio técnico (telecomunicaciones), en inglés.
- **Relación con CODEFEST:** **admisible**. Es el respaldo directo de la decisión ya tomada en
  los parsers: la fila es atómica y lleva su cabecera dentro.

### 4.3 Chunking por títulos / secciones — Unstructured

- **Proyecto:** Unstructured.io · **Año:** 2024–2026 (documentación viva)
- **URL:** <https://docs.unstructured.io/ui/chunking>
- **Tipo:** documentación técnica oficial de proyecto
- **Estrategia estudiada:** `chunk_by_title`: respeta fronteras de sección (y opcionalmente de
  página) de modo que un chunk nunca mezcla dos secciones; `combine_text_under_n_chars`
  (default 500) fusiona elementos cortos consecutivos hasta alcanzar un mínimo;
  `multipage_sections` permite respetar además el salto de página.
- **Hallazgo relevante:** el problema práctico que documentan —los elementos cortos mal
  clasificados como título producen chunks muchísimo más pequeños de lo deseado— es
  exactamente el riesgo de una política puramente estructural, y su mitigación es un mínimo de
  tamaño por chunk.
- **Limitaciones:** documentación de producto sin benchmark propio.
- **Relación con CODEFEST:** **admisible**. Aporta dos parámetros de diseño que conviene tener
  en la ablación estructural: un **mínimo** además de un máximo, y la pregunta explícita de si
  el salto de página debe ser una frontera dura (relevante: nuestros bloques de PDF **son**
  páginas).

### 4.4 Segment Any Text (SaT) / wtpsplit

- **Autores:** Markus Frohmann, Igor Sterner, Benjamin Minixhofer, Ivan Vulić, Markus Schedl
- **Año:** 2024 (EMNLP main) · **URL:** <https://arxiv.org/abs/2406.16678> ·
  <https://github.com/segment-any-text/wtpsplit>
- **Tipo:** paper (conferencia) + proyecto
- **Estrategia estudiada:** segmentación de oraciones universal, robusta a puntuación
  ausente, sobre 85 idiomas; arquitectura de tres capas con preentrenamiento con ruido.
- **Hallazgo relevante:** estado del arte entre modelos de pesos abiertos en 85 idiomas y 8
  corpus, con ~3× de mejora de velocidad frente a métodos previos; explícitamente diseñado
  para no depender de la puntuación.
- **Limitaciones:** es un modelo, no una regla: añade una dependencia y un coste de inferencia
  que `pysbd` (puro Python) no tiene.
- **Relación con CODEFEST:** **admisible** — es un modelo de segmentación, no un decoder
  generativo. Relevante porque **`pysbd` no tiene reglas de portugués** (soporta 23 idiomas,
  sin `pt`) y el corpus tiene ~55 documentos de INPE en portugués, además de texto de OCR
  donde la puntuación llega dañada.

---

## 5. Diversificación y agregación (frontera con recuperación, no con chunking)

### 5.1 Maximal Marginal Relevance (MMR)

- **Autores:** Jaime Carbonell, Jade Goldstein · **Año:** 1998 (SIGIR), vigente
- **URL:** <https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results>
  (descripción de implementación); formulación original en SIGIR'98
- **Tipo:** trabajo fundacional + documentación de proveedor
- **Estrategia estudiada:** reordenar el top-k combinando linealmente similitud a la consulta y
  disimilitud con lo ya seleccionado, controlado por un parámetro λ.
- **Hallazgo relevante:** las funciones de recuperación clásicas ignoran las relaciones entre
  documentos devueltos, así que el top-k puede ser relevante y **redundante** a la vez. Las
  métricas conscientes de diversidad (α-nDCG, *intent-aware*) existen precisamente porque
  nDCG estándar no penaliza la redundancia.
- **Limitaciones:** MMR optimiza diversidad, que **no** es lo que mide el nDCG@10 estándar del
  reto; puede bajar nDCG si se aplica sin medir.
- **Relación con CODEFEST:** **admisible** (es aritmética sobre vectores y puntuaciones). Pero
  pertenece a **post-procesamiento de resultados**, no al chunker: es la herramienta natural
  para el riesgo de que los 10 fragmentos vengan de 1–2 documentos y se degrade F1@3.

---

## 6. Fuentes internas del repositorio consultadas

No son literatura externa, pero condicionan qué de lo anterior aplica:

| Documento | Qué aporta a esta investigación |
|---|---|
| `CLAUDE.md` §2.1, §2.2, §5 | Prohibición de decoders; completitud lingüística; dónde atacar la métrica |
| `.claude/reference/spec-etapa1.md` §2, §3.3, §9.2.1, Tabla 1 | Contrato de metadata, regla de 250 palabras, permiso explícito de dividir/concatenar |
| `docs/sondeo-corpus.md` | Inventario 1.826/1.826; 50 consultas en español; distribución por formato |
| `docs/decisions/001…006` | Contrato real de `RawDoc.blocks` por formato y qué se decidió no hacer en la extracción |
| `data/interim/benchmarks/microvalidation/` | Recall 5-gram de PyMuPDF sobre los gold (0,9977) y auditoría de identidad PBF |
| `data/interim/benchmarks/prechunk/` | Devset interno (9 consultas, 15 gold), TOCs candidatos, columnas PBF |

---

## 7. Fuentes descartadas y por qué

- Listados tipo *"las 10 mejores estrategias de chunking"* en blogs de SEO y agregadores
  (firecrawl, agenta, digitalapplied, webscraft, premai y similares aparecieron repetidamente
  en las búsquedas): repiten recomendaciones de tamaño sin experimento propio ni corpus
  reproducible. No se citan.
- Artículos de Medium sin experimento (varios sobre contextual chunking, sentence window y
  small-to-big): se usó en su lugar la documentación oficial del proyecto correspondiente.
- Trabajos sobre chunking **multimodal / documento-como-imagen** (*Visual Late Chunking*,
  *Document-as-Image*, ColPali y derivados): fuera de alcance — nuestro pipeline entrega texto
  y el reto evalúa el campo `text`.
- Trabajos de chunking asistido o evaluado enteramente por decoders (LumberChunker, AutoChunker,
  *Adaptive Chunking*, *chunk filtering* con LLM): se registran solo como contexto negativo,
  porque son inaplicables bajo §8.3.
