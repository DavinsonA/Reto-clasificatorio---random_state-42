# CLAUDE.md — Reto Clasificatorio CODEFEST Ad Astra 2026

Equipo `random_state = 42`. Repo: `Reto-clasificatorio---random_state-42`.
Este archivo es el contrato entre el equipo y los asistentes de IA. Leer **§2 Reglas duras** y **§3 Contradicciones** antes de escribir cualquier línea de código.

Referencia extendida de la especificación: `.claude/reference/spec-etapa1.md`.

---

## 1. Qué se entrega

Sistema de **recuperación densa** sobre un corpus multilingüe y multiformato (~1.826 archivos, ES/EN/PT), organizado en 3 fenómenos. **No hay generación**: la Etapa 1 evalúa exclusivamente calidad de recuperación.

Entrada: 50 consultas en lenguaje natural (`q001`–`q050`).
Salida por consulta: 3 `doc_id` + 10 fragmentos de ≤ 250 palabras.

Estructura obligatoria del artefacto (`entrega/`):

```
entrega/
  resultados.jsonl          # exactamente 50 líneas, orden q001..q050
  generador.py              # reproduce resultados.jsonl desde el índice
  informe_tecnico.pdf       # máx. 8 páginas
  README.md                 # instrucciones de ejecución para el evaluador
  requirements.txt          # GENERADO con uv export, no editar a mano
  base_vectorial/
    encoder_<nombre>/
      index.faiss           # faiss.write_index()
      metadata.jsonl        # 1 objeto por línea, orden == ids internos FAISS
    encoder_<nombre2>/      # si aplica
    grafo/                  # bonus, si aplica
      grafo.graphml
```

No respetar la estructura → penalización severa o exclusión. Si `generador.py` no reproduce `resultados.jsonl`, la entrega se **excluye de la evaluación**.

**Deadline: 8 agosto 2026, 16:00 hora Bogotá. Congelamiento interno: 7 agosto 12:00.**

---

## 2. Reglas duras — violarlas invalida la entrega

### 2.1 Prohibición de decoders (§8.3 de la especificación)

Ningún modelo generativo (GPT, Llama, Gemini, Claude, Mistral…) puede intervenir en **ninguna** etapa de indexación o recuperación. Prohibido explícitamente:

- reranking con LLM
- filtrado o selección de fragmentos por generación de texto
- reformulación o expansión de consulta con un decoder
- síntesis o resumen de fragmentos para armar el `text` de salida

Permitido: vectores, puntuaciones de similitud, metadata, y **cross-encoders** (familia BERT, no generativos — confirmado en la sesión 5).

Claude Code puede escribir el código; Claude **no puede estar dentro del pipeline en runtime**. Si un asistente propone `anthropic`, `openai`, `ollama` o similar como dependencia de `generador.py`, es un error crítico: rechazar.

### 2.2 Completitud lingüística (§3.3)

Ningún fragmento —ni en el índice ni en la salida— puede contener oraciones cortadas. Una oración que empieza en un chunk termina en ese chunk. Todo corte por tamaño debe **retroceder** al final de la última oración completa que quepa. Usar un segmentador de oraciones real (`pysbd`), no `str.split(".")`.

### 2.3 Esquema de salida (§9.3)

Exactamente 50 líneas. Por línea: `documents` con **exactamente 3** objetos, `fragments` con **exactamente 10** objetos. `text` ≤ 250 palabras. Campos faltantes, cardinalidad distinta o exceso de palabras → penalizado o descartado por el validador automático del comité.

Contar palabras como `len(texto.split())`. Diseñar con margen: objetivo ≤ 240.

### 2.4 Metadata obligatoria por chunk (Tabla 1)

`doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno` (int 1|2|3), `posicion` (int, empieza en 0), `num_tokens` (int), `texto`. Se pueden añadir campos extra (idioma, título, fecha), nunca omitir los obligatorios.

**La línea `i` de `metadata.jsonl` debe corresponder al id interno `i` de FAISS.** Esto implica: escribir metadata en el mismo orden y en el mismo paso en que se hace `index.add()`. No reordenar, no deduplicar después.

### 2.5 Licencias

Código propio bajo Apache 2.0 (ya está el `LICENSE`). Dependencias y modelos de HuggingFace: solo permisivas (Apache 2.0, MIT, BSD, CC BY). **GPL/AGPL/LGPL copyleft fuerte descalifica.** Verificar la licencia de cada encoder antes de usarlo — varios modelos multilingües populares tienen licencias no comerciales o custom.

---

## 3. Contradicciones conocidas en las fuentes — decisiones tomadas

**Regla del equipo: la sesión 5 de Q&A prevalece sobre el documento técnico cuando se contradicen.** Aun así, donde el costo de cubrir ambas lecturas es bajo, se cubren ambas.

| # | Conflicto | Decisión |
|---|---|---|
| C1 | §10.2.1 dice que el emparejamiento a nivel documento con el ground truth se hace por el campo **`fuente`**; la sesión 5 (§8.3) dice que se usa **`doc_id`**, suministrado con el corpus. | **Cubrir ambas.** `doc_id` = el identificador literal del índice de ADL, sin modificar. `fuente` = la **ruta relativa exacta** del archivo tal como aparece en el índice de ADL. Así ninguna de las dos lecturas nos rompe. Nunca inventar `doc_id` propios. |
| C2 | Tabla 1 restringe `formato` a `pdf`, `html`, `md`; la sesión 5 aclara que es la **extensión real en minúsculas** y que esa lista eran ejemplos. | Extensión real en minúsculas (`json`, `xlsx`, `csv`, `pbf`, `png`, …). |
| C3 | La sesión 5 dice que la evaluación corre en **Python 3.9.5 o superior**; `pyproject.toml` declara `requires-python = ">=3.11"`. | **Riesgo abierto.** `generador.py` debe ser **sintácticamente válido y ejecutable en 3.9**: sin `match`, sin `X \| Y` en anotaciones evaluadas en runtime, sin `tomllib`, sin genéricos `list[str]` en firmas sin `from __future__ import annotations`. El resto del repo puede seguir en 3.11. Verificar con `python3.9 -m py_compile entrega/generador.py` o `vermin`. |
| C4 | El documento técnico habla de PDF/HTML/MD; el corpus real trae además JSON (~52%), PDF (~42%), XLSX, CSV, imágenes y PBF. | El router de formatos se dimensiona por la distribución **real** del corpus. JSON primero: es la mayoría. |
| C5 | Inventario: la especificación indica ~1.826 archivos; conteos previos del equipo solo reconciliaban ~1.386. | Tarea bloqueante del día 1: reconciliar el conteo real contra el archivo índice de ADL antes de indexar nada. Documentar faltantes/corruptos/duplicados con política explícita. |

Cualquier nueva contradicción se añade a esta tabla con su decisión. No resolverla en silencio dentro del código.

---

## 4. Cómo se decide (no qué se decide)

El `.claude` **no fija** encoder, chunking ni tipo de índice. Los fija el equipo con evidencia. Lo que sí es obligatorio es el **protocolo**:

### 4.1 Selección de encoder — criterios (§4.3)

Evaluar candidatos contra, en este orden de peso:

1. **Multilingüe nativo ES/EN/PT.** El corpus y las consultas mezclan los tres idiomas; la recuperación cross-lingual es el caso central, no un extra. Un encoder solo-inglés está descartado por diseño.
2. **Desempeño en recuperación densa** en MTEB/BEIR (tarea *retrieval*, no STS ni clasificación).
3. **Licencia permisiva** (§2.5). Bloqueante.
4. **Longitud máxima de entrada** (típico 512 tokens) — condiciona el chunking, no al revés.
5. **Dimensionalidad** — afecta almacenamiento y velocidad. Más dimensiones ≠ mejor.
6. **Costo de inferencia** en el hardware disponible (RTX 5070 12 GB + RTX 5060 8 GB, Blackwell sm_120, ruedas `cu128`).

Registrar cada candidato descartado y por qué, en `docs/decisions/`. El informe técnico exige justificar la elección; sin bitácora hay que reconstruirla a última hora.

### 4.2 Chunking — criterios (§3.2)

Estrategias válidas: tamaño fijo, por oración, por párrafo, jerárquica/estructural, semántica con solapamiento, o híbridas. La única restricción dura es §2.2 (completitud lingüística) y el límite de tokens del encoder.

Decidir por evidencia, no por gusto. Variables a barrer: tamaño objetivo, solapamiento, si se respeta o no la estructura del documento, y granularidad por formato (una fila de CSV no se chunkea igual que un PDF de 80 páginas).

### 4.3 Índice FAISS (§5.2)

`IndexFlatIP` con vectores L2-normalizados = coseno exacto. Para el volumen del reto la especificación lo declara suficiente. Cualquier índice aproximado (IVF, HNSW) exige justificar en el informe qué se ganó y cuánta exactitud se perdió — y medirlo, no suponerlo.

### 4.4 Fusión multi-encoder (§8.4)

Si hay más de un encoder: un índice FAISS **independiente por encoder**, y fusión por CombSUM, CombMNZ o RRF (`k0=60`). RRF es robusto a diferencias de escala entre encoders; CombSUM/CombMNZ exigen puntuaciones comparables. Elegir midiendo.

### 4.5 Agregación a documento (§8.6)

Agrupar los top-`k_chunk` por `doc_id`, puntuar el documento (max-pooling / suma / media ponderada), ordenar, tomar 3. Solo aritmética sobre puntuaciones de FAISS.

### 4.6 Protocolo de ablación

No se acepta "mejoró" sin número. Toda variante se compara sobre el **mismo** set de consultas de desarrollo y se registra en `docs/ablaciones.md` con: configuración, métrica proxy, delta, y decisión. Ver `/ablacion`.

---

## 5. Dónde atacar la métrica

Entender qué se mide evita optimizar lo que no puntúa.

- **NDCG@10 (fragmentos)** se juzga sobre el **contenido textual** del campo `text`, no sobre `chunk_id`. El `chunk_id` es solo trazabilidad. Consecuencia práctica: §9.2.1 permite concatenar un chunk con su vecino inmediato del mismo documento hasta 250 palabras. Un chunk de 60 palabras entregado tal cual desperdicia ~190 palabras de cobertura potencial. **Rellenar hasta el límite es una palanca directa de NDCG, no un adorno.**
- **NDCG penaliza más los errores arriba.** La posición 1 pesa `1/log2(2)=1.0`; la 10 pesa `1/log2(11)≈0.29`. Ordenar bien el top-3 de fragmentos vale más que afinar la cola.
- **F1@3 (documentos)** es métrica de conjunto: el orden entre los 3 no importa, pero la precisión sí. Con `R@3 = |D̂∩D*| / min(|D*|,3)`, si el ground truth tiene ≥3 documentos relevantes, cada documento errado cuesta a la vez precisión y recall.
- **Riesgo de colapso por documento**: si los 10 fragmentos vienen de 1–2 documentos, la señal para elegir 3 documentos distintos se degrada. Considerar diversificación explícita a nivel documento en la agregación, y medir si ayuda o estorba.
- **Borda count**: el leaderboard suma posiciones en las dos tablas. Un equipo mediocre-consistente vence a uno excelente-desbalanceado (ejemplo de la §11.2.2 de la especificación). **No sacrificar F1@3 por NDCG@10 ni al revés.**

---

## 6. Fases

Orden no negociable. Nada de la fase 2 empieza hasta que la fase 1 esté completa, validada y commiteada.

**Fase 0 — Reconciliación del corpus (bloqueante).**
Inventario reproducible contra el índice de ADL. Conteo real por formato, por fenómeno, por observatorio. Política explícita para faltantes, corruptos y duplicados. Sin esto, cualquier métrica posterior es ruido.

**Fase 1 — Baseline entregable de punta a punta.**
Extracción → limpieza → chunking → 1 encoder → `IndexFlatIP` → recuperación → `resultados.jsonl` válido → `generador.py` que reproduce → `entrega/` completa y validada. **Objetivo: tener una entrega válida lo antes posible**, aunque la métrica sea mediocre. Una entrega mediocre puntúa; una entrega inválida no puntúa.

**Fase 2 — Mejora medida.**
Segundo encoder + fusión, reranking con cross-encoder, ajuste de chunking, post-filtros por metadata. Cada cambio pasa por el protocolo de ablación (§4.6).

**Fase 3 — Bonus: grafo de conocimiento.**
NER multilingüe → extracción de relaciones → `grafo.graphml` (NetworkX), con cada tripleta referenciando `doc_id` y `chunk_id`. Integración por RRF tratando el grafo como un índice adicional (§8.5). **Solo si la fase 2 está cerrada y hay margen de tiempo.** El bonus no compensa una entrega frágil.

**Fase 4 — Informe y empaquetado.**
Informe técnico (≤ 8 páginas) desde la bitácora de decisiones, README para el evaluador, `requirements.txt` regenerado, prueba de reproducción en entorno limpio.

---

## 7. Convenciones

- **Código, funciones y variables**: inglés. **Docstrings y comentarios**: español.
- Módulos `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE`.
- Type hints obligatorios en funciones públicas. Docstrings Google-style.
- `ruff` para lint y formato.
- Nunca `except:` desnudo. Errores de extracción por archivo se **loggean y se saltan**, no tumban el pipeline: 1.826 archivos heterogéneos garantizan que algunos fallen.
- Nada de rutas absolutas ni de `print()` en el pipeline. Logging estructurado.
- Commits: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`), subject sin tildes.
- **No commitear** el corpus, los `.faiss`, los embeddings ni archivos > 10 MB. `data/` y `entrega/base_vectorial/` van en `.gitignore`; la entrega se sube a la carpeta compartida en la nube.
- `entrega/requirements.txt` es **generado**. Regenerar, nunca editar:
  ```
  uv export --extra cpu --no-dev --no-hashes --no-emit-project --emit-index-url -o entrega/requirements.txt
  ```

### Frontera de dependencias

| Dónde | Qué va ahí | ¿Lo instala el evaluador? |
|---|---|---|
| `[project.dependencies]` | solo lo que importa `generador.py` | **sí** |
| `[dependency-groups] dev` | extracción, chunking, notebooks | no |
| `[project.optional-dependencies]` | variante de torch (`cpu`/`gpu`) | no |

Antes de `uv add <paquete>`, preguntar: ¿lo importa `generador.py`? Si no → `uv add --group dev <paquete>`. Cada dependencia de runtime innecesaria es superficie de fallo en la máquina del evaluador.

---

## 8. Comandos

```bash
uv sync --extra gpu                      # entorno con CUDA 12.8 (Blackwell)
uv sync --extra cpu                      # entorno CPU
uv run --extra gpu python -c "import torch; print(torch.cuda.is_available())"

uv run --extra gpu python src/indexar.py           # construye entrega/base_vectorial/
uv run --extra gpu python entrega/generador.py     # produce entrega/resultados.jsonl

ruff check src/ entrega/ && ruff format src/ entrega/
```

**Siempre `--extra gpu` también en `uv run`**: `uv run` sincroniza antes de ejecutar y sin el flag revierte torch a CPU.

Slash commands del repo: `/indexar`, `/generar`, `/validar`, `/ablacion`, `/empaquetar`.
Subagentes: `extractor`, `chunker`, `indexador`, `evaluador`, `revisor-entrega`.

---

## 9. Roles

| Persona | Rol | Frontera |
|---|---|---|
| Davinson Arteaga | Líder / arquitecto | Arquitectura, decisiones de scope, interlocución con el comité, informe técnico |
| Juan Villegas | Data & Geo Engineer | Extracción por formato, PBF/geo, inventario y reconciliación |
| Daniela Castaño | RAG & LLM Engineer | Chunking, encoders, índice, recuperación, métricas |
| Gian Mendoza | Visualización | Diagnóstico del corpus, análisis de ablaciones, figuras del informe |

Contratos entre roles: el esquema `RawDoc` (salida de extracción) y el esquema de metadata de la Tabla 1 son los **contratos estrictos**. Se acuerdan una vez y no se cambian sin avisar a todos.

---

## 10. Anti-patrones

- Meter un LLM en cualquier parte del pipeline de recuperación. Descalifica (§2.1).
- Optimizar la métrica antes de tener una entrega válida de punta a punta.
- Reordenar o deduplicar `metadata.jsonl` después de construir el índice: rompe la correspondencia con los ids de FAISS y con ella toda la trazabilidad.
- Cortar chunks con `str.split(".")` — falla en abreviaturas, decimales, siglas y URLs, y viola §2.2.
- Devolver menos de 10 fragmentos o menos de 3 documentos "porque no había buenos candidatos". El esquema es de cardinalidad fija; rellenar siempre.
- Añadir dependencias pesadas a `[project.dependencies]` que `generador.py` no importa.
- Empezar el informe técnico el 7 de agosto. La documentación es criterio de evaluación, no epílogo.
- Perseguir el grafo bonus con la fase 1 sin cerrar.
- Aceptar "mejoró bastante" sin número y sin línea en `docs/ablaciones.md`.
