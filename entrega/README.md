# CODEFEST Ad Astra 2026 — Etapa 1

> Equipo `random_state = 42`

Sistema de **recuperación densa** sobre el corpus multilingüe de ADL (1.826 documentos, ES/EN/PT,
3 fenómenos). Dadas 50 consultas en lenguaje natural, devuelve por cada una **3 documentos** y
**10 fragmentos** de ≤ 250 palabras.

## Contenido

| Ruta | Descripción |
| --- | --- |
| `generador.py` | punto de entrada único: reproduce `resultados.jsonl` desde la base vectorial |
| `codefest_runtime/` | runtime de recuperación (autocontenido, sin dependencias del repo) |
| `base_vectorial/encoder_bge_m3/` | `index.faiss`, `metadata.jsonl` y `manifest.json` |
| `requirements.txt` | dependencias de ejecución, fijadas con `==` |
| `consultas.jsonl` | las 50 consultas de entrada (lo aporta el comité) |
| `resultados.jsonl` | salida generada |

## Requisitos

- **Python 3.9.5 o superior.** Verificado ejecutando el pipeline completo en un entorno limpio de
  **Python 3.9.25**.
- **CPU.** No se necesita GPU ni CUDA: no se regeneran los embeddings del corpus, solo se carga el
  índice ya construido, se codifican las 50 consultas y se busca.
- ~2 GB de RAM para el índice y la metadata, y ~1,7 GB de disco para `base_vectorial/`.

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generador.py
```

Esto escribe `resultados.jsonl` en esta misma carpeta. Los tres argumentos son **opcionales**;
la invocación anterior equivale a:

```bash
python generador.py \
  --consultas consultas.jsonl \
  --base-vectorial ./base_vectorial \
  --salida ./resultados.jsonl
```

| Argumento | Por defecto | Para qué sirve |
| --- | --- | --- |
| `--consultas` | `consultas.jsonl` | JSONL de entrada, un objeto por línea |
| `--base-vectorial` | `./base_vectorial` | carpeta con `encoder_bge_m3/` |
| `--salida` | `./resultados.jsonl` | dónde escribir la salida |
| `--device` | `cpu` | dispositivo de PyTorch |
| `--verbose` | — | log de nivel DEBUG |

Los valores por defecto se resuelven respecto a la carpeta donde vive `generador.py`, así que
también funciona invocándolo por ruta absoluta desde cualquier directorio.

## Encoder y red

El índice se construyó con **`BAAI/bge-m3`**, revisión
`5617a9f61b028005a4858fdac845db406aefb181` (1024 dimensiones, vectores L2-normalizados). La misma
revisión se fija en el runtime: consultar con otros pesos cambiaría el ranking.

El encoder se resuelve **local primero**:

1. si existe `modelos/bge_m3/` dentro de esta carpeta, se carga desde ahí y **no hace falta red**;
2. si no, se descarga de HuggingFace fijando esa revisión exacta.

> **Esta copia de la entrega no incluye el checkpoint empaquetado**, así que **la primera
> ejecución necesita conexión a internet** para descargar ~2,2 GB (las siguientes usan la caché de
> HuggingFace). Si el entorno de evaluación no tiene red, copie el checkpoint en `modelos/bge_m3/`
> y la ejecución será completamente offline.

## Formato de entrada

```json
{"query_id": "q001", "query": "¿Cómo ...?"}
```

Se exigen exactamente 50 líneas con `query_id` de `q001` a `q050`, en ese orden, sin repetidos y
con texto no vacío. El texto de la consulta se usa **literal**: no se traduce, normaliza ni
expande.

## Formato de salida

Una línea JSON por consulta, UTF-8 sin escapar:

```json
{
  "query_id": "q001",
  "documents": [{"rank": 1, "doc_id": "..."}, {"rank": 2, "doc_id": "..."}, {"rank": 3, "doc_id": "..."}],
  "fragments": [{"rank": 1, "chunk_id": "...", "doc_id": "...", "text": "..."}]
}
```

Siempre 3 documentos (`rank` 1–3, distintos) y 10 fragmentos (`rank` 1–10, `text` ≤ 250 palabras).

## Cómo funciona

```
consultas.jsonl → BGE-M3 → FAISS IndexFlatIP → top-100 por consulta
                → M4: concatena el mejor vecino inmediato del mismo documento si cabe en 250 palabras
                → normalización: divide lo que exceda 250 palabras sin cortar ninguna oración
                → 10 fragmentos
                → max-pooling documental sobre todo el soporte legalmente entregable
                → 3 documentos
```

`IndexFlatIP` sobre vectores normalizados es coseno exacto: no hay aproximación ni parámetros de
búsqueda que ajustar. La selección de vecino de M4 usa la similitud BGE entre la consulta y el
vector del vecino, reconstruido del propio índice.

Un fragmento cuya unidad lingüística es indivisible y supera las 250 palabras (una "oración" de un
PDF mal extraído, una fila de CSV enorme) **no se trunca**: se salta y se registra en el log, y se
continúa con los siguientes candidatos del mismo top-100.

## Rendimiento medido

En un portátil con CPU, Python 3.9.25, sobre el índice completo (326.866 vectores):

| Etapa | Tiempo |
| --- | --- |
| Carga de `index.faiss` + `metadata.jsonl` | ~3 s |
| Carga de BGE-M3 (ya en caché) | ~12 s |
| Recuperación de las 50 consultas | ~3 s (0,06 s/consulta) |
| **Total** | **~18 s** |

La primera ejecución añade la descarga del encoder.

## Errores comunes

| Síntoma | Causa y solución |
| --- | --- |
| `no existe el archivo de consultas` | falta `consultas.jsonl` o la ruta de `--consultas` es incorrecta |
| `se esperaban 50 consultas y hay N` | el JSONL no trae las 50 consultas oficiales |
| `no existe encoder_bge_m3 dentro de la base vectorial` | `base_vectorial/` incompleta; no se hace fallback a otra carpeta |
| `desalineacion FAISS <-> metadata` | `index.faiss` y `metadata.jsonl` no son del mismo build |
| `el indice no es IndexFlatIP` | se sustituyó el índice por uno aproximado |
| error al cargar el encoder | sin red y sin `modelos/bge_m3/`: empaquete el checkpoint |

Cualquier fallo termina con código de salida distinto de 0 y **sin** escribir un
`resultados.jsonl` a medias: la salida se escribe de forma atómica solo cuando las 50 consultas se
han procesado.

## Reproducibilidad

Mismas consultas, mismo índice y misma revisión del encoder producen un `resultados.jsonl`
byte a byte idéntico (verificado comparando SHA256 de dos ejecuciones). No se usa aleatoriedad;
`langdetect` corre con semilla fija.

`base_vectorial/encoder_bge_m3/manifest.json` documenta la procedencia: modelo, revisión,
dimensión, tipo de índice, número de filas y documentos, y los SHA256 del índice y la metadata.
