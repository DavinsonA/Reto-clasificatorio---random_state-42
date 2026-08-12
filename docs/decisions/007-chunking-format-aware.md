# ADR-007: Baseline de chunking format-aware

- **Fecha**: 2026-08-11
- **Estado**: propuesta — se adopta como **baseline de implementación**; no está demostrado que
  sea la política óptima y queda sujeto a ablación (E1–E6 de `docs/research/chunking-best-practices-codefest.md` §19)
- **Responsable**: Daniela Castaño (RAG & LLM Engineer)

## Contexto

Los parsers de `src/extract/` están congelados y entregan `RawDoc(doc_id, fuente, formato,
fenomeno, title, blocks, extra)`. Los `blocks` son unidades naturales de extracción en orden de
lectura, **no** chunks de recuperación. El perfil medido sobre los 318.314 bloques reales
(`docs/research/chunking-best-practices-codefest.md` §3) hace inviable una política única:

- **PDF**: bloque = página. Mediana 353 palabras, **66,5 % de las páginas superan las 250
  palabras** del límite de salida y 22,5 % superan 512.
- **JSON**: bloque = párrafo. Mediana 37 palabras; **41,5 % ≤ 25 palabras**.
- **XLSX**: bloque = fila. Mediana 21 palabras; 80 % ≤ 25.
- **CSV**: bloque = fila. Mediana 60 palabras, pero **80 % de todos los bloques del corpus**.
  `F1-AIINDEX-056` aporta él solo 111.775 filas (35,1 % del índice con bloque = chunk).
- **PBF**: bloque = feature. Mediana 86 palabras, distribución muy uniforme.

Restricciones que acotan el diseño:

- **§3.3 de la especificación**: ningún fragmento con oraciones incompletas; los cortes solo en
  fronteras oracionales. Segmentador real (`pysbd`), nunca `str.split(".")`.
- **§9.2.1**: el fragmento entregado va ≤ 250 palabras, y un chunk se puede dividir o concatenar
  con su vecino inmediato del mismo documento. **La unidad indexada y la entregada no tienen que
  coincidir.**
- **§8.3 / `CLAUDE.md` §2.1**: ningún decoder en ninguna etapa. El chunker es aritmética sobre
  palabras y fronteras lingüísticas.
- **`num_tokens` (Tabla 1) exige el tokenizador del encoder, que todavía no está elegido.**
- 1.105 bloques narrativos contienen una "oración" de más de 250 palabras (§3.8): para ellos es
  **imposible** emitir un fragmento ≤ 250 palabras sin romper una frontera oracional.

## Opciones consideradas

| Opción | A favor | En contra |
|---|---|---|
| **P0 — bloque = chunk** (E0) | Cero código; el contrato de los parsers ya particiona | 318.314 vectores; el 67 % de los fragmentos de PDF hay que dividirlos igualmente en la salida; los tabulares desperdician el 66–92 % del presupuesto |
| **Presupuesto único para todo el corpus** | Una sola rama de código | Optimiza de hecho para filas de CSV: el 85,2 % de los bloques son tabulares. Ninguna cifra agregada del corpus describe a la vez una página de 353 palabras y una fila de 21 |
| **Semántico / propositional / late chunking** | Literatura favorable en dominios homogéneos | Coste de embeddings sobre 30,5 M de palabras antes de tener encoder; sensibilidad a umbrales no medibles sin ground truth; §22 del research los descarta para esta fase |
| **Format-aware + packing cruzando bloques (adoptada)** | Es la única hipótesis que el perfil del corpus exige por sí solo; reduce la explosión de vectores tabulares sin borrar una sola fila; divide la prosa larga en fronteras lingüísticas | Más superficie de código; atribuir la mejora a la rama correcta obliga a medir por formato |

## Decisión

**Sin medición de recuperación.** Existe un development set interno
(`data/interim/benchmarks/prechunk/devset.jsonl`: 9 consultas, 8 con gold, 15 fragmentos gold, 11
documentos, **todos PDF**, fenómenos 1 y 3, sin cobertura de F2), pero **todavía no existe el
pipeline encoder → embeddings → índice FAISS → recuperación** que haría falta para comparar esta
política con métricas de recuperación sobre él. Lo que sustenta esta decisión es el perfil
estructural del corpus y las reglas duras de la especificación, no una fila de
`docs/ablaciones.md`. **La deuda está marcada**: el objetivo de 200 palabras es un punto de partida
razonado, no un óptimo demostrado.

Cuando ese devset se use, hay que leer antes sus limitaciones (research §4.1): es diminuto, solo
PDF, no cubre F2 y su uniformidad (138–180 palabras) sugiere la convención de recorte del anotador
más que la longitud natural de la evidencia.

### Política narrativa (`json`, `pdf`, `txt`, `jpg`, `jpeg`, `png`, `avif`)

- Un bloque que cabe en `max_words` se conserva **indivisible**: no se convierte preventivamente
  en oraciones. Los párrafos de JSON ya son la frontera que la literatura identifica como la mejor
  familia, y llegan gratis.
- Un bloque que supera `max_words` se segmenta con `pysbd`, y cada oración pasa a ser una unidad.
- La **frontera de página de PDF es blanda**: la cola de una página se puede empaquetar con el
  inicio de la siguiente. `posicion` es ordinal dentro del documento, no dentro de la página, así
  que no hay conflicto con la Tabla 1.

### Política tabular (`csv`, `xlsx`, `pbf`)

- **La fila y la feature son unidades atómicas**: no pasan por segmentación de oraciones, no se
  parten por `|` ni por número de columnas, no se resumen, no se reordenan, no se deduplican.
- El chunker **sí agrupa** filas o features consecutivas hasta el presupuesto. Los comentarios de
  `csv_docs.py` describen la fila como "un chunk ya delimitado"; la lectura vigente es más
  precisa: **fila = unidad atómica ≠ retrieval chunk**. No se tocó el parser para corregir esa
  terminología.
- **No se mezclan hojas de XLSX ni capas de PBF**: un cambio de `group_key` (derivado del prefijo
  `[Hoja]` / `[Capa]` que ya añaden los parsers, y solo para esos dos formatos) cierra el chunk.

### Parámetros

```
narrativa y tabular:
    target_words       = 200     # objetivo blando
    soft_min_words     = 120     # evita colas inútilmente pequeñas
    max_words          = 250     # techo de un chunk empaquetado
    overlap_units      = 0
    cross_block        = true    # dentro del mismo documento y del mismo grupo
    page_boundary_pdf  = blanda

salida (evidencia):
    output_target_words = 240
    output_max_words    = 250
    neighbor expansion  = infraestructura disponible, NO conectada a recuperación
```

`overlap_units != 0` **falla explícitamente** (`NotImplementedError`): la interfaz queda, el
comportamiento se implementa cuando E3 se mida. Dos estudios independientes no encuentran mejora
con solapamiento (§7 del research).

### Unidades atómicas > 250 palabras

Una unidad indivisible que supera el techo **se emite entera, sola, sin vecinos**, marcada con
`oversized_atomic = True`. Un `ChunkDraft` puede por tanto superar las 250 palabras: eso es
deliberado. El límite de 250 rige la **evidencia entregada**, no es excusa para destruir una
oración o una fila durante la indexación.

Al construir la evidencia, `split_for_output` lanza `UnreturnableAtomicUnitError` en estos casos
en vez de truncar. **La política final para convertirlos en fragmentos entregables sigue abierta**
(§ Riesgos, deuda 1).

### Interacción medida entre el packing (E2) y la expansión por vecinos (E6)

**E2 y E6 son sustitutos, no complementos, con `target_words = 200`.** Medido sobre los 171.780
chunks de la corrida completa, contando cuántas ventanas de evidencia legales (≤ 250 palabras)
admite cada chunk:

| Ventanas legales | Chunks | % |
|---|---:|---:|
| 0 (el propio chunk ya pasa de 250 → `split_for_output`) | 1.547 | 0,90 % |
| 1 (solo el propio chunk) | 169.191 | **98,49 %** |
| 2 (un vecino) | 1.040 | 0,61 % |
| 3 | 2 | 0,00 % |

Mediana del presupuesto de salida sin usar: **69 palabras** (p90 = 109).

El research (§10.3, §19-E6) esperaba que E6 fuera "el mayor beneficio por unidad de esfuerzo"
porque bajo bloque = chunk los fragmentos usaban entre el 8 % y el 34 % del presupuesto. Este
baseline **consume ese margen en la indexación**: al empaquetar hasta ~200 palabras ya no queda
sitio para concatenar un vecino. La infraestructura de `evidence_candidates` sigue siendo correcta
y necesaria, pero **E6 solo será medible de verdad con un `target_words` bastante menor** (la banda
120–160 del research). No es un defecto de la implementación: es la consecuencia aritmética de esta
elección de tamaño, y hay que tenerla presente al diseñar el orden de las ablaciones.

### Lo que este ADR NO decide

Encoder, tokenizador, tipo de índice, fusión, agregación documental, reranking, prefijo de
contexto determinístico (E4) y selección entre ventanas de evidencia (E6). El chunker deja los
tres últimos preparados sin activarlos.

## Consecuencias

**Qué se gana**

- Reducción fuerte del número de vectores tabulares **sin eliminar una sola fila**.
- La prosa larga se divide en fronteras lingüísticas reales (§3.3 cumplido por construcción).
- Los casos imposibles quedan **contados y trazables** (`oversized_atomic`), no escondidos.
- Arquitectura preparada para barrer target narrativo, presupuesto tabular, overlap, tokenizer,
  contexto determinístico y expansión por vecinos **sin reescribir el chunker**.
- Invariante de conservación verificable por documento: `src/chunking/audit.py`.

**Qué se pierde o queda pendiente**

- Un chunk tabular deja de ser "una fila" y pasa a ser "un rango de filas contiguas": si la
  evidencia relevante es una fila concreta, empaquetarla con otras la diluye. No medido.
- Un chunk de PDF puede empezar en una página y acabar en la siguiente, uniendo dos contenidos sin
  relación temática. No medido.
- El presupuesto está en **palabras, no en tokens**: falta validar `palabras × ratio_subpalabra(ES/PT)`
  contra `model_max_seq_length` cuando el encoder esté elegido.

**Qué habría que revisar si esta decisión resulta equivocada**

Si E2/E5 muestran que el packing daña NDCG@10 más de lo que ayuda a F1@3, la vuelta atrás es
barata: las políticas de referencia son configuración explícita y están cubiertas por tests
(`tests/test_chunking_policies.py`).

| Política | Cómo se reproduce |
|---|---|
| **E0** — bloque = chunk | `block_as_chunk_config()` (sin packing entre bloques **y** sin segmentación por oraciones) |
| **E1** — packing solo dentro del bloque | `ChunkingConfig(cross_block_packing=False)` |
| **E2/E5** — baseline vigente | `ChunkingConfig()` |

> Corregido tras la auditoría del 2026-08-11. Una versión anterior de este ADR afirmaba que
> `ChunkingConfig(target_words=1)` reproducía bloque = chunk. Es **falso** por partida doble: esa
> llamada lanza `ValueError` (`soft_min_words` no puede superar `target_words`), y aun corrigiéndola
> a `ChunkingConfig(target_words=1, soft_min_words=1)` solo coincide con E0 en los formatos
> tabulares — en narrativa un bloque por encima de `max_words` se sigue segmentando por oraciones
> (2 bloques → 12 chunks en el caso probado). E0 exige además desactivar esa segmentación, que es
> justo lo que hace `block_as_chunk_config()`.

## Riesgos y deudas abiertas

1. **Unidades atómicas > 250 palabras.** Existen y no tienen política de salida. Son tablas
   volcadas como texto corrido (`F2-CSIS-171`) y PDF con glifos cifrados por falta de `ToUnicode`
   CMap (`F3-CEOBS-030`; RESDAL). Hoy `split_for_output` falla explícitamente.
   **Medido en la corrida completa**: 1.547 chunks `oversized_atomic` (0,90 % del índice) en 269
   documentos — 1.202 `pdf`, 314 `csv`, 21 `json`, 6 `pbf`, 4 `xlsx`. El más largo es una fila de
   9.235 palabras (`F1-AIINDEX-041`).
   El total narrativo (1.223) **no coincide** con los 1.105 bloques que el research contó con una
   "oración" > 250, y la diferencia está explicada, no es una regresión: 241 de los 243 documentos
   históricos coinciden y los `json` cuadran exacto (21 = 21). El exceso viene de (a) los 72 bloques
   cuya segmentación se rechazó por no ser lossless y se conservaron enteros, y (b) bloques
   multiescritura donde la puntuación no lleva espacio adyacente —p. ej. una sección en árabe de
   `F1-DAIO-031`, donde `pysbd` propone 13 trozos y la regla de fronteras los vuelve a unir en 1
   para no partir palabras—. Ambas son consecuencias deliberadas de preservar el contenido.
2. **Portugués.** `pysbd` no tiene reglas de `pt` (108 documentos del corpus). Se usa el ruleset
   español como el romance más cercano, marcado con `fallback=True` y contabilizado en la
   auditoría. Es una aproximación declarada, no una verdad lingüística.
   **Cómo leer la métrica**: la detección de idioma es *perezosa* — solo corre en documentos con
   algún bloque por encima de `max_words`. En la corrida completa hay 11 fallbacks `pt`, que **no**
   son "11 documentos en portugués": son los documentos en portugués que además necesitaron
   segmentación. El perfil reporta `documents_with_language_detection` como denominador.
   Caso límite detectado: `F3-CEOBS-030` (glifos cifrados) se detecta como **alemán** con
   `fallback=False`; el idioma es basura porque el texto lo es, y ninguna señal lo delata.
3. **Presupuesto de tokens.** `max_tokens` y `TokenCounter` existen en la interfaz pero el baseline
   corre solo con palabras. Bloqueado por la selección de encoder.
4. **Frontera de página blanda en PDF.** Pendiente de ablación (Q2 del research).
5. **Packing de filas.** Pendiente de métricas de recuperación (E2/E5).
6. **Expansión por vecinos (E6).** `evidence_candidates` genera las ventanas legales pero **no
   elige entre ellas**: elegir sin medir sería enterrar la pregunta bajo una heurística. Además,
   con `target_words = 200` el 98,49 % de los chunks no admite ningún vecino (ver *Interacción
   medida entre E2 y E6*): medir E6 exige bajar el target.
7. **Contexto determinístico (E4).** No activado. `texto` no lleva prefijos; cuando se pruebe, irá
   en un `embedding_text` aparte, sin mutar `texto` (Tabla 1).
8. **`target_words = 200` no es un óptimo medido.** Los candidatos del research son 160 / 240 / 350
   para narrativa y 240 / 500 para tabular.

## Nota de implementación: fronteras de oración dentro de una palabra

`pysbd` corta `CT.;` en `CT.` y `;` (`F1-AIINDEX-002`), y `advantage.”` en `advantage.` y `”`
(`F2-SWF-121`). No pierde caracteres, pero concatenar esas unidades con un separador convierte una
palabra en dos y rompe el conteo. El chunker **rehace** toda frontera que no tenga un espacio
pegado a uno de sus dos lados, y descarta la segmentación entera si no reproduce el bloque de
entrada.

La frontera se juzga con los propios trozos devueltos por `pysbd`, **no** con desplazamientos
acumulados sobre el bloque original: `pysbd` normaliza algún espacio de vez en cuando y a partir de
ahí cualquier índice calculado se desalinea, que fue exactamente el fallo detectado en
`F2-SWF-121`. Con la regla local, la conservación de contenido se cumple palabra a palabra sobre
los 1.826 documentos.
