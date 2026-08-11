# Chunking research for CODEFEST Ad Astra 2026

**Fecha:** 2026-08-10 · **Branch:** `dev` · **HEAD:** `b75993b` · **Estado:** investigación, sin código de producción
**Bitácora de fuentes externas:** [`chunking-sources.md`](chunking-sources.md)
**Perfil cuantitativo:** `data/interim/research/chunking/` (regenerable, ver §25)

Este documento **no elige** la estrategia de chunking. Reúne la evidencia —del corpus real, de
la especificación y de la literatura— necesaria para que el equipo diseñe una batería pequeña
de ablaciones y la decida con números. Todas las cifras del corpus salen de medir los 1.826
`RawDoc` extraídos, no de estimaciones.

---

## 1. Scope and constraints

### 1.1 Qué se decide después de este documento

- Qué unidades del corpus son **atómicas** y cuáles se pueden partir o unir.
- Si hay **una política única** o **políticas por formato**.
- Tamaño objetivo, solapamiento y si la unidad embebida coincide con el fragmento entregado.
- Qué 5–7 experimentos se implementan y en qué orden.

### 1.2 Qué NO se decide aquí, explícitamente

Encoder, modelo de embeddings, tipo de índice FAISS, BM25, reranker, fusión multi-encoder y
agregación documental final. Esas decisiones **condicionan** el chunking (§10) pero pertenecen
a otras fases del experimento; mezclarlas invalidaría la lectura de cualquier ablación.

### 1.3 Reglas oficiales que acotan el espacio de diseño

| # | Regla | Fuente | Consecuencia para el chunker |
|---|---|---|---|
| R1 | Ninguna oración puede quedar cortada; el corte retrocede al final de la última oración completa | spec §3.3 | El segmentador de oraciones es **infraestructura obligatoria**, no un adorno |
| R2 | `posicion` es ordinal dentro del documento, empieza en 0 | Tabla 1 | El chunker define el orden; no se puede reordenar después |
| R3 | Metadata obligatoria por chunk: `doc_id`, `chunk_id`, `fuente`, `formato`, `fenomeno`, `posicion`, `num_tokens`, `texto` | Tabla 1 | Cualquier política debe poder emitir los 8 campos |
| R4 | La línea `i` de `metadata.jsonl` = id interno `i` de FAISS | `CLAUDE.md` §2.4 | Metadata y `index.add()` en el mismo paso; nada de deduplicar después |
| R5 | La relevancia se juzga sobre el campo `text`, no sobre `chunk_id` | spec §10.2 | El `chunk_id` es trazabilidad; el texto entregado es lo que puntúa |
| R6 | `text` ≤ 250 palabras | spec §9.3 | Límite de **salida**, no de indexación |
| R7 | Chunk > 250 → dividir respetando oraciones; chunk corto → concatenar con el vecino inmediato del mismo documento | spec §9.2.1 | **La unidad recuperada y la unidad entregada pueden diferir** |
| R8 | 10 fragmentos + 3 documentos por consulta, cardinalidad fija | spec §9.3 | Siempre hay que rellenar; no se devuelve menos |
| R9 | Ningún decoder generativo en indexación ni recuperación | spec §8.3 | Excluye contextual retrieval generado, propositionizers y chunking asistido por LLM |

**R7 es la regla más infrautilizada de la especificación** y la que más condiciona este
análisis: autoriza explícitamente una arquitectura *retrieval unit ≠ evidence unit* (§14).

---

## 2. Current RawDoc contract

El chunker recibe `RawDoc(doc_id, fuente, formato, fenomeno, title, blocks, extra)`. Los
parsers de `src/extract/` están **congelados**: `blocks` son unidades naturales de extracción
en orden de lectura, **no chunks finales**. Nada de lo que sigue propone tocarlos.

| Formato | Bloque natural | Docs | Bloques | Palabras | Bloques/doc (mediana global 9) | Riesgo principal para el chunking |
|---|---|---:|---:|---:|---:|---|
| `json` | párrafo / sección / campo estructurado | 954 | 10.475 | 676.552 | 11,0 | Bloques **muy cortos**: mediana 37 palabras, 85 % ≤ 100 |
| `pdf` | página | 759 | 36.570 | 13.302.268 | 48,2 | Bloques **muy largos**: mediana 353 palabras, 67 % > 250 |
| `csv` | fila | 26 | 255.793 | 15.862.177 | 9.838,2 | **Domina el índice**: 80 % de todos los bloques del corpus |
| `xlsx` | fila | 4 | 8.901 | 208.564 | 2.225,2 | Filas minúsculas: mediana 21 palabras, 80 % ≤ 25 |
| `pbf` | feature geográfica | 73 | 6.523 | 507.421 | 89,4 | Repetición cross-zoom entre documentos (59,5 % de huellas duplicadas) |
| `jpg`/`avif` | línea de transcripción u OCR | 9 | 38 | 866 | 4,2 | Volumen irrelevante (0,01 % de los bloques) |
| `txt` | unidad textual reconstruida | 1 | 14 | 598 | 14,0 | Caso único, ya limpio |
| **Total** | | **1.826** | **318.314** | **30.558.446** | | |

`extra` trae además señal aprovechable y determinística: `observatorio` (siempre),
`num_paginas`, `paginas_ocr`, `paginas_baja_densidad` (PDF), `columnas`, `tabular`,
`serie_temporal`, `orden_temporal`, `num_filas` (CSV/XLSX), `capas`, `num_entidades` (PBF),
`contenido_minimo` (27 documentos), `source_url`, `scraped_at` (TXT). Todo esto es contexto
**ya existente**: usarlo no viola R9 (§12).

### 2.1 La asimetría que define el problema

**103 documentos tabulares (5,6 % del corpus) producen 271.217 bloques: el 85,2 % del índice.**
Los 1.723 documentos narrativos (94,4 %) producen 47.097 bloques (14,8 %). En palabras el
reparto es casi 50-50 (16,58 M tabulares vs. 13,98 M narrativas), pero en **número de vectores**
—que es lo que compite en el top-k— la desproporción es de 5,8 a 1.

Cualquier política única aplicada a todo el corpus está, de hecho, optimizando para filas de
tabla.

---

## 3. Corpus block profile

Medido sobre los 318.314 bloques reales. Fuente:
`data/interim/research/chunking/block_profile_by_*.csv`.

### 3.1 Percentiles de palabras por bloque, por formato

| Formato | min | p10 | p25 | **p50** | p75 | p90 | p95 | p99 | max | media |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `csv` | 7 | 46 | 54 | **60** | 67 | 75 | 82 | 153 | 9.235 | 62,0 |
| `pdf` | 1 | 90 | 196 | **353** | 492 | 647 | 740 | 990 | 4.630 | 363,7 |
| `json` | 1 | 6 | 12 | **37** | 74 | 118 | 148 | 274 | 6.561 | 64,6 |
| `xlsx` | 2 | 15 | 18 | **21** | 25 | 28 | 31 | 37 | 6.050 | 23,4 |
| `pbf` | 3 | 24 | 28 | **86** | 101 | 116 | 128 | 170 | 260 | 77,8 |
| `jpg` | 4 | 6 | 15 | **29** | 32 | 32 | 32 | 33 | 33 | 23,2 |
| `txt` | 12 | 13 | 21 | **29** | 39 | 108 | 108 | 108 | 108 | 42,7 |
| **Global** | 1 | 38 | 53 | **61** | 72 | 164 | 390 | 671 | 9.235 | 96,0 |

La mediana global de 61 palabras **es un artefacto**: refleja la fila mediana de CSV, no el
corpus de lectura. Ese es exactamente el tipo de número que no debe usarse para fijar un tamaño.

### 3.2 Umbrales acumulados (% de bloques por debajo de N palabras)

| Formato | ≤25 | ≤50 | ≤100 | ≤150 | ≤200 | ≤250 | >250 | >400 | >512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `csv` | 2,83 | 16,88 | 97,92 | 98,96 | 99,70 | 99,88 | **0,12** | 0,03 | 0,02 |
| `pdf` | 3,70 | 6,04 | 11,50 | 17,78 | 25,80 | 33,47 | **66,53** | 40,97 | 22,47 |
| `json` | 41,45 | 59,66 | 85,42 | 95,23 | 98,07 | 98,83 | **1,17** | 0,76 | 0,74 |
| `xlsx` | 80,23 | 99,90 | 99,94 | 99,96 | 99,96 | 99,96 | **0,04** | 0,04 | 0,04 |
| `pbf` | 18,34 | 28,87 | 74,51 | 96,72 | 99,71 | 99,91 | **0,09** | 0,00 | 0,00 |
| `txt` | 42,86 | 78,57 | 85,71 | 100,0 | 100,0 | 100,0 | **0,00** | 0,00 | 0,00 |
| **Global** | 6,69 | 19,62 | 87,16 | 89,49 | 91,16 | 92,22 | **7,78** | 4,75 | 2,62 |

Lectura directa: **el problema de "chunk demasiado grande" es exclusivamente de PDF** (66,5 %
de las páginas superan el límite de salida; 22,5 % superan incluso 512 palabras). El problema
de "chunk demasiado pequeño" es de JSON (41 % ≤ 25 palabras) y XLSX (80 % ≤ 25).

### 3.3 Por fenómeno

| Fenómeno | Docs | Bloques | Palabras | Bloques/doc | p25 | p50 | p75 | p95 | >250 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F1 — IA y capacidades | 459 | 275.598 | 19.542.541 | 600,4 | 53 | 60 | 68 | 122 | 2,56 % |
| F2 — Seguridad espacial | 479 | 19.561 | 5.875.653 | 40,8 | 103 | 265 | 449 | 704 | 51,81 % |
| F3 — Dinámicas territoriales | 888 | 23.155 | 5.140.252 | 26,1 | 27 | 95 | 394 | 706 | 32,81 % |

F1 está masivamente sesgado por los CSV del AI Index. **F2 y F3 son fenómenos de prosa larga**;
F1, tal como está extraído, es mayoritariamente tabular. Esto importa porque las 50 consultas se
reparten de forma equilibrada entre los tres fenómenos (spec §10.1).

### 3.4 Por observatorio (extracto; tabla completa en el CSV)

| Observatorio | Docs | Bloques | Bloques/doc | p50 palabras | >250 |
|---|---:|---:|---:|---:|---:|
| AI_Index_Stanford | 65 | 264.208 | 4.064,7 | 60 | 0,78 % |
| Amazon_Underworld | 75 | 10.893 | 145,2 | 77 | 0,06 % |
| CSIS_Aerospace | 214 | 9.125 | 42,6 | 317 | 56,88 % |
| SWF_Counterspace | 135 | 4.991 | 37,0 | 291 | 53,86 % |
| Atlantic_Council | 186 | 4.221 | 22,7 | 40 | 0,43 % |
| RESDAL | 107 | 3.549 | 33,2 | 525 | 85,26 % |
| Alertas_Tempranas | 425 | 2.669 | 6,3 | 52 | 30,39 % |
| ILIA_Latam | 10 | 1.721 | 172,1 | 414 | 81,35 % |
| CEEEP | 80 | 160 | 2,0 | 88 | 3,75 % |
| INPE | 59 | 558 | 9,5 | 54 | 5,20 % |

**Un solo observatorio (AI_Index_Stanford) aporta el 83 % de los bloques del corpus.**

### 3.5 Distribución a nivel documento

| Métrica | p10 | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bloques por documento | 2 | 5 | **9** | 27 | 69 | 137 | 457 | **111.775** |
| Palabras por documento | 125 | 188 | **1.040** | 6.156 | 19.723 | 36.058 | 160.529 | **6.769.205** |

- **73 documentos tienen un único bloque**; 281 tienen ≤ 3.
- **580 documentos (31,8 %) tienen menos de 250 palabras en total**: el documento completo cabe
  en un solo fragmento entregable. 102 tienen menos de 100.
- 27 documentos tienen `contenido_minimo` (bloque sintético de título+observatorio).

### 3.6 Concentración: el hallazgo más consecuente del perfil

Con la política trivial "un bloque = un chunk":

| | Bloques | % del índice |
|---|---:|---:|
| `F1-AIINDEX-056` (CSV, 111.775 filas, 6,77 M palabras) | 111.775 | **35,11 %** |
| Top-5 documentos | 240.696 | **75,62 %** |
| Top-10 documentos | 263.426 | **82,76 %** |

Un solo `doc_id` de 1.826 posee más de un tercio del índice. Cinco documentos poseen tres
cuartas partes. Empaquetar filas hasta 250 palabras reduce `F1-AIINDEX-056` de 111.775 a 31.039
chunks —pero eso sigue siendo el **22 %** del índice resultante. **El chunking mitiga la
concentración; no la resuelve** (§16, §24).

> Nota: el brief menciona `F1-AIINDEX-042` (~8.865 filas) como el caso extremo. Medido, ese
> documento es el **sexto**: 8.866 filas. Los cuatro CSV de AI_Index por encima de él
> (`056`, `063`, `059`, `057`) suman 231.830 filas.

### 3.7 Oraciones

Segmentado con `pysbd` sobre los 47.097 bloques narrativos (JSON, PDF, TXT, imágenes):

| Formato | Oraciones | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pdf` | 543.032 | 8 | **19** | 32 | 49 | 65 | 125 | 1.391 |
| `json` | 29.860 | 11 | **21** | 30 | 41 | 48 | 73 | 582 |
| `txt` | 24 | 16 | **23** | 30 | 43 | 47 | 52 | 52 |

Una oración mediana pesa ~20 palabras. **Un presupuesto de 250 palabras ≈ 12 oraciones; uno de
150 ≈ 7.** Ese es el grano real con el que trabajará cualquier packer.

### 3.8 Dos hallazgos que condicionan el requisito R1 (completitud lingüística)

**(a) 1.105 bloques (2,35 % de los narrativos) contienen una "oración" de más de 250 palabras**,
con un máximo de 1.391. Para esos bloques **es imposible** emitir un fragmento ≤ 250 palabras sin
romper una frontera oracional. Inspeccionados, no son prosa:

- `F2-CSIS-171`: una tabla presupuestaria volcada como texto corrido, sin puntuación terminal
  ("*Department of Homeland Security U.S. Coast Guard … PPA Discretionary - Appropriation…*").
- `F3-CEOBS-030` (120 de los 138 casos de CEOBS): **texto ilegible por fuente sin `ToUnicode` CMap** —
  "*/LVW RI 7DEOHV 8EFPI %REP]WMW SJ XLI PIKEP…*" es un índice de tablas cifrado por el
  desplazamiento de glifos. Es el problema que el ADR-003 ya anotó para RESDAL (`�`), aquí en
  forma más severa. RESDAL, con el mismo problema en forma mas leve, aporta 276 casos repartidos en 54 documentos.

Consecuencia: el chunker necesita una **política de escape declarada** para el caso "no hay
frontera oracional dentro del presupuesto". No es un caso teórico: son 1.105 bloques reales.

**(b) `pysbd` no tiene reglas de portugués.** Soporta 23 idiomas y `pt` no está entre ellos. El
perfil detecta **108 documentos en portugués** (INPE sobre todo, F2 y F3). En este perfil se
segmentaron con el *ruleset* español, el romance más cercano disponible. Es una aproximación
aceptable para medir, pero es una decisión que hay que tomar explícitamente antes de indexar
(§23, Q6).

### 3.9 Idiomas detectados (por documento)

| Idioma | Docs | | Idioma | Docs |
|---|---:|---|---|---:|
| inglés | 1.022 | | desconocido | 39 |
| español | 616 | | francés | 11 |
| portugués | 108 | | otros (ca, ar, ru, zh, de, ko, tr) | 30 |

Cruce con fenómeno: F3 es mayoritariamente español (574), F1 y F2 mayoritariamente inglés
(415 y 374). Los "otros" son casi todos falsos positivos del detector sobre bloques tabulares
cortos o sobre OCR ruidoso. Recordatorio del sondeo: **las 50 consultas están todas en
español**, así que la recuperación cross-lingual es el caso central, no el excepcional.

---

## 4. Gold evidence profile

Development set interno (`data/interim/benchmarks/prechunk/devset.jsonl`), construido a partir
de `FASE ORDENADA CODEFEST.xlsx`: **9 consultas, 8 con gold, 15 fragmentos gold, 11 documentos,
todos PDF, fenómenos 1 y 3.** No es el ground truth oficial.

| Métrica | min | p25 | mediana | media | p75 | max |
|---|---:|---:|---:|---:|---:|---:|
| Palabras | 138 | 151 | **156** | 159,3 | 165 | 180 |
| Caracteres | 906 | — | ~1.067 | — | — | 1.168 |
| Oraciones | 2 | 4 | **5** | 5,3 | 7 | 9 |
| Palabras/oración | 18,9 | — | 32,2 | 34,6 | 41,0 | 78,0 |

Cuántos gold caben completos dentro de una unidad de tamaño N (**evidencia adicional, no
criterio de selección**):

| N (palabras) | 100 | 150 | 200 | 250 | 350 | 500 |
|---|---:|---:|---:|---:|---:|---:|
| Gold que caben enteros | **0 / 15** | 4 / 15 | **15 / 15** | 15 / 15 | 15 / 15 | 15 / 15 |

### 4.1 Limitaciones de este devset — leer antes de usar cualquiera de esos números

1. **Es diminuto**: 8 consultas con gold. Cualquier diferencia menor a varios puntos es ruido.
2. **No cubre F2** en absoluto, que es justo el fenómeno con la prosa más larga (mediana 265
   palabras/bloque, 51,8 % > 250).
3. **Solo tiene PDF.** Cero evidencia sobre JSON (52 % del corpus), tabulares o PBF.
4. **La uniformidad es sospechosa.** Los 15 gold caben en una banda de 138–180 palabras y
   906–1.168 caracteres. Un rango tan estrecho sugiere que refleja la **convención de recorte de
   quien anotó** —probablemente un extracto de ~1.000 caracteres— y no la longitud natural de la
   evidencia. Usarlo para fijar el tamaño de chunk sería optimizar contra el anotador, no contra
   la tarea.

Lo único que este devset sostiene con solidez: **un presupuesto de 250 palabras es holgado para
la evidencia observada**, y **100 palabras es demasiado corto** (0/15). Nada más.

---

## 5. What the literature says

Organizado por familia. Detalle, URLs y limitaciones de cada fuente en
[`chunking-sources.md`](chunking-sources.md).

### 5.1 Fixed-size / token-based

- **Qué resuelve:** trocear a coste cero, sin depender de la estructura del documento.
- **Cómo funciona:** ventana de N tokens/caracteres, avance N−overlap.
- **Necesita:** un tokenizador. Nada más.
- **Coste:** despreciable (<1 s por corpus en 2606.00881).
- **Evidencia positiva:** en Qu et al. 2024 el fixed-size gana en 3 de 5 datasets de
  recuperación de evidencia y empata en el resto; en 2606.00881 logra 87,71 de Accuracy@5 en
  <1 s frente a 89,36 del recursive semantic en ~5 min.
- **Evidencia negativa:** en la evaluación de 36 métodos (2603.06976) el fixed-size **por
  carácter** es el peor de todos (nDCG@5 < 0,244; P@1 2–3 %) frente a 0,459 del agrupamiento por
  párrafos. La contradicción se resuelve al mirar la variante: fixed-size **por oración** es
  competitivo; fixed-size **por carácter** no lo es.
- **Compatibilidad CODEFEST:** cortar por carácter/token sin retroceder a la frontera oracional
  **viola R1**. Solo la variante alineada a oraciones es admisible.

### 5.2 Sentence-aware packing

- **Qué resuelve:** llenar un presupuesto sin romper unidades lingüísticas.
- **Cómo funciona:** segmentar en oraciones, acumular hasta el presupuesto, cerrar en la última
  oración que quepa.
- **Necesita:** un segmentador real (`pysbd`, SaT). Coste bajo (medido aquí: **4,8 ms por página
  de PDF**, 461 s para el corpus completo incluyendo simulaciones).
- **Impacto en vectores:** ver §17. Con presupuesto 250, PDF pasa de 36.570 a 73.687 chunks (×2)
  si se empaqueta dentro de la página, o a 58.758 (×1,6) si se empaqueta cruzando páginas.
- **Evidencia:** en 2601.14123 el chunking por oración y el semántico rinden parecido y por
  encima del chunking por token puro. En 2409.04701 las fronteras por oración dan el mejor
  baseline naive (52,4 nDCG@10) de las tres probadas.
- **Riesgos:** depende de la calidad del segmentador; el corpus tiene 1.105 bloques donde falla
  (§3.8a) y 108 documentos en un idioma que `pysbd` no cubre (§3.8b).
- **Compatibilidad CODEFEST:** es la familia que **satisface R1 por construcción**.

### 5.3 Paragraph-aware packing

- **Qué resuelve:** respetar la unidad de argumentación del autor, no solo la gramatical.
- **Evidencia positiva:** es la estrategia ganadora del estudio más amplio disponible
  (2603.06976, 36 métodos × 6 dominios × 5 encoders): **Paragraph Group Chunking, nDCG@5 ≈ 0,459
  y P@1 ≈ 24 %**, frente a 0,244 y 2–3 % del fixed-size por carácter.
- **Evidencia neutra:** ese estudio usa juicios de relevancia generados por LLM, no humanos.
- **Aplicabilidad aquí:** **el corpus ya trae los párrafos**. Los `blocks` de JSON son
  `body_paragraphs` tal cual los entregó ADL: una frontera de párrafo gratuita y de buena
  calidad. En PDF, en cambio, el bloque es la **página**, no el párrafo — no tenemos fronteras de
  párrafo fiables ahí (§13.2).

### 5.4 Recursive chunking

- **Qué resuelve:** degradar ordenadamente por una jerarquía de separadores
  (`\n\n` → `\n` → `. ` → ` `) hasta caber en el presupuesto.
- **Evidencia:** en el reporte de Chroma el `RecursiveCharacterTextSplitter` es el baseline y
  se comporta razonablemente a 200 tokens (precisión 7,0 ± 5,6) y mal a 800 (1,5 ± 1,3).
- **Riesgo bajo CODEFEST:** el último nivel de la cascada (` `, corte por espacio) **viola R1**.
  Solo es admisible una variante cuyo nivel más profundo sea la oración, y con la política de
  escape de §3.8a para lo que no quepa.
- **Aplicabilidad aquí:** limitada. La cascada de separadores es valiosa cuando se recibe texto
  crudo; nosotros recibimos bloques ya segmentados por los parsers. Recursive sobre `RawDoc` es
  casi equivalente a sentence packing con una capa extra de complejidad.

### 5.5 Structural / hierarchical chunking

- **Qué resuelve:** que un chunk nunca mezcle dos secciones distintas.
- **Cómo funciona:** `chunk_by_title` de Unstructured respeta fronteras de sección (y
  opcionalmente de página); `combine_text_under_n_chars` fusiona elementos cortos para evitar
  chunks minúsculos.
- **Evidencia:** el propio proyecto documenta el fallo típico —listas o párrafos cortos
  clasificados erróneamente como título producen chunks mucho menores de lo deseado— y su
  mitigación (un **mínimo** de tamaño, además de un máximo).
- **Aplicabilidad aquí:** **parcial y desigual**. Tenemos estructura explícita en JSON
  (`sections.heading` en CENIA) y en TXT, pero **no en PDF**: la microvalidación mostró que
  PyMuPDF detecta **0 headings** en los 14 documentos probados, mientras `pymupdf4llm` detectó
  19 y Docling 31 sobre los rangos gold. Cambiar de extractor está fuera de alcance (parsers
  congelados) y además ambos alternativos tienen **peor recall de texto** sobre los gold
  (0,9421 y 0,9340 frente a **0,9977** de PyMuPDF).
- **Señal estructural que sí tenemos y no cuesta nada:** 346 de los 759 PDF tienen al menos una
  página candidata a índice/TOC (`toc_summary.json`, 530 páginas). Son páginas de puro texto
  ornamental con *dot leaders*; competirían en el índice sin aportar evidencia.

### 5.6 Semantic chunking

- **Qué resuelve:** poner la frontera donde cambia el tema, no donde se acaba el presupuesto.
- **Variantes:** *breakpoint* por caída de similitud entre oraciones consecutivas (con umbral
  absoluto o percentil), y *clustering* de oraciones (aglomerativo, DBSCAN).
- **Coste:** hay que **embeber cada oración** del corpus antes de decidir las fronteras. Para
  nosotros: 543.032 oraciones de PDF + 29.860 de JSON ≈ **573.000 embeddings adicionales**, más
  el pase final de embeddings de los chunks resultantes.
- **Evidencia positiva:** el `ClusterSemanticChunker` de Chroma a 200 tokens gana en precisión
  (8,0 ± 6,0) e IoU; a 400 tokens queda segundo en recall (91,3 %).
- **Evidencia negativa:** Qu et al. 2024 concluye que **el coste no se justifica**: en
  recuperación de evidencia el fixed-size gana en 3 de 5 datasets con diferencias de centésimas
  (ExpertQA 47,11 vs. 47,08 vs. 46,87). Las ganancias grandes aparecen solo en documentos
  sintéticos *stitched*, con diversidad temática artificialmente alta. 2606.00881 llega a la
  misma conclusión desde el ángulo del coste.
- **Dependencia crítica:** las fronteras dependen del **encoder que se use para decidirlas**.
  Hacer semantic chunking antes de elegir encoder es medir con una regla que después se cambia.
- **Compatibilidad CODEFEST:** admisible (usa un encoder, no un decoder), **siempre que la
  variante no sea `LLMSemanticChunker` ni LumberChunker**, que sí usan decoders.

### 5.7 Sliding-window / overlap

Tratado por separado en §7 por su impacto directo en el tamaño del índice.

### 5.8 Sentence-window retrieval y 5.9 Small-to-big / parent-child

Tratados en §10 y §14, porque en nuestro caso no son técnicas de chunking sino de
**post-procesamiento del resultado**, expresamente autorizadas por R7.

### 5.10 Context-enriched chunks y 5.11 Late chunking

Tratados en §12 y §11 respectivamente.

### 5.12 Format-aware / heterogeneous chunking

- **Qué resuelve:** que una fila de tabla y una página de informe no se traten igual.
- **Evidencia:** 2603.06976 muestra que la estrategia ganadora **cambia por dominio** (dynamic
  token size en biología/física/salud; paragraph grouping en legal/matemáticas). 2505.21700
  muestra que el tamaño óptimo cambia por dataset **y por encoder**. La literatura sobre tablas
  (§5.13) recomienda una política de fila que sería absurda en prosa.
- **Aplicabilidad aquí:** el perfil de §3.1–3.2 hace el argumento solo. Un mismo presupuesto de
  250 palabras es un **límite** para el 66,5 % de las páginas de PDF y un **objetivo inalcanzable**
  para el 80 % de las filas de XLSX. No hay un tamaño que sirva a los dos.

### 5.13 Chunking de tablas

- **Evidencia:** el chunking **a nivel de fila supera al de tabla completa**, y **repetir la
  cabecera en cada fila** mejora la exactitud de recuperación (2408.17008, especificaciones
  3GPP). El framework STC (2605.00318) formaliza la fila como unidad clave-valor, con fusión
  greedy sin solapamiento: −40 % de chunks frente a recursive, MRR 0,3576 → 0,5945 en híbrido,
  Recall@1 0,366 → 0,754 en BM25.
- **Aplicabilidad aquí:** **nuestros parsers ya implementan exactamente eso** —
  `columna: valor | columna: valor` por fila, cabecera repetida, celdas vacías omitidas. Lo que
  la literatura añade es la parte que **no** tenemos: empaquetar filas consecutivas hasta un
  presupuesto de tokens en vez de emitir una fila por vector.

### 5.14 Chunking de documentos largos

- **Evidencia:** DICE (2606.18781) muestra que la evidencia decisiva de un documento largo **se
  diluye** al comprimirse en un vector único, y que agregar puntuaciones de chunk lo corrige
  (Passkey >4k: 30,0 → 90,0). 2505.21700 muestra que los documentos largos con consultas de
  contexto amplio prefieren chunks grandes (512–1024 tokens) y los de respuesta puntual, chunks
  pequeños (64–128).
- **Aplicabilidad aquí:** tenemos PDF de hasta 1.330 páginas (`F2-CSIS-156`) y 502.000 palabras
  (`F2-CSIS-201`). La §8.6 de la especificación ya autoriza max-pooling sobre chunks para
  puntuar el documento: es, funcionalmente, lo que DICE propone.

### 5.15 Retrieval unit vs. returned evidence unit

- **Evidencia:** Dense X Retrieval (EMNLP 2024) demuestra que la unidad de indexación es una
  decisión de primer orden: indexar por proposiciones supera a indexar por pasajes en +50–55 %
  de EM@100 (retrievers no supervisados). El **método** de generar proposiciones usa un LLM y es
  **no admisible**; el **hallazgo** —granularidad fina recupera mejor— sí lo es.
- Desarrollado en §14.

---

## 6. Fixed-size vs linguistic boundaries

### 6.1 Lo que dice la evidencia

La comparación honesta no es "fijo vs. lingüístico" sino **a qué se alinea el corte**:

| Alineación | nDCG@5 (2603.06976) | ¿Cumple R1? |
|---|---:|---|
| Carácter, tamaño fijo | 0,244 | **No** |
| Token, tamaño fijo | intermedio | **No** |
| Oración | competitivo con semántico (2601.14123) | **Sí** |
| Grupo de párrafos | **0,459** (mejor global) | **Sí** |

Es decir: **la evidencia externa y la regla obligatoria de CODEFEST apuntan en la misma
dirección.** No hay tensión que resolver. El coste de cumplir R1 no es un impuesto: es la opción
que la literatura ya prefería.

### 6.2 Palabras, tokens y por qué no se puede fijar el tamaño todavía

Este análisis está en **palabras** (`len(texto.split())`) porque:

1. Es la unidad en la que la especificación define el límite duro (250 palabras, R6).
2. No hay encoder elegido, y elegir un tokenizador solo para este perfil sesgaría la conclusión
   hacia ese modelo (2505.21700 muestra que el tamaño óptimo cambia **por encoder**).

Para convertir el diseño a `model_max_seq_length` cuando se elija encoder:

```
tokens ≈ palabras × ratio_subpalabra(idioma, tokenizador)
```

Con tokenizadores multilingües tipo SentencePiece/XLM-R el ratio típico es ~1,3–1,5 para inglés
y **~1,5–2,0 para español y portugués** (más morfología, peor cobertura de vocabulario). Con eso:

| Presupuesto en palabras | Tokens estimados (EN) | Tokens estimados (ES/PT) | ¿Cabe en 512? |
|---:|---:|---:|---|
| 150 | 195–225 | 225–300 | Sí |
| 250 (= límite de salida) | 325–375 | **375–500** | Justo, al filo |
| 400 | 520–600 | **600–800** | **No: truncaría** |

**Consecuencia práctica y no obvia:** con un encoder de 512 tokens, un chunk de 250 palabras en
español ya está al límite, y uno de 400 palabras se truncaría silenciosamente —perdiendo
precisamente la cola del texto que sí se entregaría al evaluador. Esto **debe verificarse con el
tokenizador real** del encoder elegido antes de fijar ningún presupuesto, y es una razón fuerte
para no elegir el tamaño en esta fase.

---

## 7. Overlap

### 7.1 Evidencia externa: dos resultados negativos independientes

- **2601.14123 (2026):** añadir 10–20 % de solapamiento **no mejora** BERTScore ni Exact Match
  (diferencias dentro del margen estadístico) mientras infla el número de chunks por
  `1/(1−r)`. Los autores recomiendan explícitamente **0 %**.
- **Chroma (2024):** el solapamiento **castiga el IoU** por tokens redundantes recuperados; el
  default de 800 tokens con 400 de solapamiento produce "las peores puntuaciones en el resto de
  métricas".

No se encontró un trabajo con experimento reproducible que muestre una ganancia clara y
consistente del solapamiento en recuperación densa. **`overlap = 0` es el baseline obligatorio,
no una carencia a corregir.**

### 7.2 Overlap medido en unidades distintas

El brief pide no tratar el solapamiento como "20 %" a secas. Las variantes tienen propiedades
distintas:

| Unidad de solapamiento | Compatible con R1 | Duplicación inducida | Comentario |
|---|---|---|---|
| Tokens / caracteres crudos | **No** — parte oraciones | proporcional a `r` | Descartado por la regla obligatoria |
| Palabras | **No** por sí solo | proporcional a `r` | Necesitaría retroceso a frontera oracional, con lo que se convierte en el siguiente caso |
| **Oraciones** (1–2) | **Sí** | variable, depende del tamaño de la oración | La única variante limpia bajo R1 |
| **Bloques** (1 página / 1 fila) | Sí (el bloque es atómico) | alta en PDF (una página ≈ 353 palabras) | Sensato solo con bloques pequeños |

### 7.3 Coste medido en nuestro corpus

Solapamiento de **1 oración** sobre empaquetado cruzando bloques (medido, no estimado):

| Presupuesto | Sin overlap | Con 1 oración | Inflación |
|---:|---:|---:|---:|
| 120 palabras | 326.114 | 432.438 | **+32,6 %** |
| 180 | 200.533 | 300.649 | **+49,9 %** |
| 250 | 138.353 | 177.336 | **+28,2 %** |
| 350 | 95.746 | 113.470 | +18,5 % |
| 500 | 65.809 | 74.076 | +12,6 % |

Con presupuestos pequeños —los que la evidencia de precisión favorece— el solapamiento de una
sola oración añade entre un cuarto y la mitad de vectores.

### 7.4 El argumento específico de CODEFEST contra el overlap

R7 ya permite **reconstruir contexto en el momento de la entrega**: un fragmento corto puede
concatenarse con su vecino inmediato del mismo documento hasta 250 palabras. Eso cubre el
problema que el solapamiento intenta resolver (evidencia partida entre dos chunks) **sin pagar
duplicación en el índice** y sin diluir la señal. Además, el solapamiento aumenta el riesgo de
que el top-10 se llene de fragmentos casi idénticos del mismo documento, que es exactamente el
mecanismo que degrada F1@3 (§16).

---

## 8. Structural chunking

### 8.1 Qué estructura tenemos realmente

| Fuente de estructura | Disponible | Calidad | Coste |
|---|---|---|---|
| `RawDoc.title` | Siempre | Buena en JSON/TXT; en PDF cae al nombre de archivo si no hay metadata incrustada | 0 |
| `extra["observatorio"]` | Siempre | Exacta (viene del índice de ADL) | 0 |
| Párrafo | JSON (`body_paragraphs`), TXT | Alta — es la segmentación original de la fuente | 0 |
| `sections.heading` | JSON de CENIA (15 docs) | Alta | 0 |
| Nombre de hoja | XLSX multi-hoja | Exacta, ya prefijada por el parser (`[hoja] fila`) | 0 |
| Cabeceras de columna | CSV/XLSX/PBF | Exacta, ya dentro de cada fila | 0 |
| Página | PDF | Exacta (`posicion` del bloque) | 0 |
| **Heading de sección** | **PDF: no** | PyMuPDF detecta 0 headings | Requeriría cambiar de extractor |
| Página de TOC | PDF: detectable | 346 docs / 530 páginas identificadas heurísticamente | Ya medido |

### 8.2 La lectura honesta

La estrategia estructural que la literatura premia (paragraph grouping, chunk-by-title)
**podemos aplicarla al 52 % del corpus (JSON) casi gratis, y no podemos aplicarla al PDF** sin
romper la restricción de parsers congelados. Y no conviene romperla: los extractores que sí
detectan headings tienen peor recall del texto gold (0,9421 y 0,9340 vs. 0,9977 de PyMuPDF), y
el texto es lo que se evalúa.

La pregunta útil, tal como la plantea el brief, es: **¿podemos obtener buen chunking con los
bloques que ya tenemos?** Para PDF la única frontera estructural disponible es la **página**, y
la pregunta abierta es si debe ser una frontera **dura** (no empaquetar cruzando páginas) o
**blanda** (§13.2, §23-Q2). Unstructured expone exactamente esta dicotomía en su parámetro
`multipage_sections`, señal de que no tiene respuesta universal.

### 8.3 Páginas de índice/TOC

530 páginas de 346 documentos son candidatas a TOC. Sus características —líneas con *dot
leaders*, sin oraciones, alta densidad de números— las hacen a la vez malas como evidencia y
peligrosas como chunk (son parte de los bloques sin frontera oracional de §3.8a). Filtrarlas o
no es una hipótesis medible, pero **es filtrado, no chunking**: pertenece a una ablación propia
y afecta al recall documental si se equivoca.

---

## 9. Semantic chunking

### 9.1 Coste concreto para este corpus

Aplicar semantic chunking (cualquier variante basada en embeddings) exige embeber **~573.000
oraciones** antes de decidir una sola frontera, más el pase de embeddings de los chunks
resultantes. Sobre una RTX 5070 de 12 GB con un encoder de ~110 M parámetros eso es del orden de
decenas de minutos por pasada — repetible por cada variante de umbral que se quiera probar, y
**repetible otra vez si se cambia de encoder**.

### 9.2 Sensibilidad y dependencia

- **Al umbral:** las variantes por percentil (p90, p95) y por umbral absoluto producen números
  de chunks muy distintos sobre el mismo texto; Qu et al. probaron 4 umbrales relativos y 2
  absolutos precisamente porque no hay un valor canónico.
- **Al encoder:** las fronteras las decide el modelo con el que se calcula la similitud. Elegir
  fronteras con el encoder A y luego indexar con el encoder B es un experimento mal controlado.
- **Al corpus:** las ganancias reportadas se concentran en documentos con **alta diversidad
  temática interna**. Nuestros documentos son informes institucionales monotemáticos y filas de
  tabla: el escenario donde Qu et al. encuentran que no compensa.

### 9.3 Evidencia enfrentada, y cómo leerla

| Fuente | Veredicto | Métrica | Matiz |
|---|---|---|---|
| Qu et al. 2024 | No compensa | F1@5 evidencia | Gana solo en documentos sintéticos *stitched* |
| Chroma 2024 | Sí compensa | Precisión/IoU a nivel token | El ganador es `ClusterSemanticChunker`, no el breakpoint clásico |
| 2603.06976 (2026) | Parcial | nDCG@5 | Lo semántico queda **por debajo** del agrupamiento por párrafos |
| 2606.00881 (2026) | No compensa | Accuracy@5 vs. tiempo | Recursive semantic +1,65 pts por ~300× más tiempo |

Los cuatro son consistentes con una misma lectura: **el semantic chunking rara vez es el
ganador; cuando lo es, gana por poco y a un coste alto, y nunca supera a una política que respete
las fronteras naturales del documento cuando esas fronteras existen.** En nuestro caso existen
(párrafos en JSON, filas en tabulares, páginas en PDF).

**No es candidato para la primera ronda.** Merece ablación *baja*, y solo después de agotar las
políticas baratas.

---

## 10. Sentence-window and small-to-big retrieval

### 10.1 El patrón

- **Sentence window:** se embebe una oración; se devuelve una ventana de N oraciones a su
  alrededor.
- **Small-to-big / parent-child:** se embebe el hijo (chunk pequeño); se devuelve el padre
  (sección, página, documento) o se fusionan hijos cuando bastantes del mismo padre entran en el
  top-k (`AutoMergingRetriever`).

La idea común: **la unidad que mejor discrimina en el espacio vectorial no es la que mejor
informa al lector.** Dense X Retrieval lo cuantifica: granularidad fina mejora la recuperación
en +19–55 % de EM@100 según el retriever.

### 10.2 Por qué esto encaja con CODEFEST mejor que en un RAG normal

En un RAG convencional el "padre" se devuelve a un LLM que lo resume. Aquí no hay LLM (R9), pero
**tampoco hace falta**: R7 autoriza literalmente el patrón sin generación.

> *"Chunk < 250 palabras → se puede concatenar con el fragmento inmediatamente anterior o
> posterior del mismo documento para enriquecer contexto, sin superar el límite. […] el
> `chunk_id` reportado es el del fragmento original del índice."*

Es un *sentence-window retrieval* con ventana asimétrica y presupuesto de 250 palabras, escrito
en la propia especificación. Y R7 cubre también la dirección contraria: un chunk grande se
divide en sub-fragmentos que respetan el límite, **cada uno ocupando su propio rank**.

### 10.3 La palanca numérica

`CLAUDE.md` §5 ya lo señala: un fragmento de 60 palabras entregado tal cual **desperdicia 190
palabras de cobertura**. Con el perfil real:

- **JSON: mediana de bloque 37 palabras.** Entregado sin rellenar, un fragmento típico de JSON
  usa el **15 %** del presupuesto disponible.
- **XLSX: mediana 21 palabras** → 8 % del presupuesto.
- **CSV: mediana 60 palabras** → 24 %.
- **PBF: mediana 86 palabras** → 34 %.
- **PDF: mediana 353 palabras** → hay que dividir, no rellenar.

Sobre el 85 % del índice (los tabulares), entregar el chunk crudo desperdicia entre dos tercios
y el 92 % del texto que el evaluador está dispuesto a leer. **Rellenar hasta 250 no es un
adorno: es la palanca de NDCG con mejor relación beneficio/coste identificada en todo este
análisis**, y no requiere ni un vector adicional en el índice.

Riesgo simétrico: rellenar con vecinos irrelevantes **diluye** la evidencia y puede bajar la
relevancia juzgada del fragmento. Hay que medirlo, no asumirlo.

---

## 11. Late Chunking

### 11.1 Qué es exactamente

Tokenizar el documento **entero**, pasarlo por el transformer, y aplicar las fronteras de chunk
**después**, sobre la secuencia de embeddings de token, haciendo *mean pooling* por segmento.
Ocurre en **indexación**; la consulta se embebe igual que siempre. El resultado sigue siendo un
vector por chunk, con sus fronteras y su texto: **la trazabilidad y la metadata de la Tabla 1 se
mantienen intactas** (R3, R4 no se ven afectadas).

### 11.2 Requisitos

1. Encoder de **contexto largo** (8k tokens en los experimentos: jina-v2-small, jina-v3,
   nomic-embed-text-v1). Un encoder de 512 tokens no permite late chunking en ningún sentido
   útil.
2. **Mean pooling** obligatorio. No funciona con pooling `CLS`.
3. Memoria: el paso único se vuelve inviable en documentos muy largos; la variante *long late
   chunking* trocea en macro-chunks solapados.

### 11.3 Evidencia

| Fronteras | nDCG@10 naive | nDCG@10 late | Δ |
|---|---:|---:|---:|
| Fijas, 256 tokens | 52,2 | 54,0 | **+1,8** |
| 5 oraciones | 52,4 | 54,3 | **+1,9** |
| Semánticas | 52,4 | 53,8 | **+1,5** |

Medias sobre SciFact, NFCorpus, FiQA y TRECCOVID con tres encoders. Ganancia real pero modesta.
Casos donde **no** ayuda: Needle-8192 y Passkey-8192 (el contexto añadido es irrelevante) y
comprensión lectora con chunks de 512+ tokens, donde el naive gana a veces. Y el estudio
independiente que lo compara cabeza a cabeza (2504.19754) encuentra que late chunking *"a veces
supera y a veces queda por debajo"* del early chunking **según el encoder**, sobre muestras
pequeñas (~300–5.000 documentos).

### 11.4 ¿Es una estrategia de chunking? No

Late chunking **no cambia dónde se ponen las fronteras**: cambia **cómo se calcula el vector**
dado un conjunto de fronteras. En el propio paper es una variable ortogonal — lo miden con
fronteras fijas, por oración y semánticas, y mejora las tres por igual.

Clasificarlo correctamente:

- **No es** una política de chunking → no pertenece a la primera ablación.
- **Sí es** una estrategia conjunta de **encoder + pooling** → pertenece a la fase de selección
  de encoder, y **restringe** esa elección a modelos de contexto largo con mean pooling.
- Si el encoder elegido por otros criterios (multilingüe ES/EN/PT, licencia permisiva, MTEB
  retrieval) no cumple ambos requisitos, late chunking **queda fuera automáticamente**, sin
  necesidad de medirlo.

### 11.5 Interacción con el límite de 250 palabras

Ninguna directa. Las fronteras siguen existiendo y el texto asociado a cada chunk no cambia; lo
único que cambia es el vector. R6 y R7 se aplican igual. **Compatible, pero fuera de fase.**

---

## 12. Context enrichment

### 12.1 La distinción que decide la admisibilidad

| | Contexto **determinístico** | Contexto **generado** |
|---|---|---|
| Origen | Campos que ya existen: `title`, heading, nombre de hoja, cabeceras, `observatorio`, jerarquía geográfica | Un LLM redacta 50–100 tokens describiendo el chunk |
| Coste | ~0 (concatenación de strings) | Una llamada a decoder por chunk |
| Reproducible | Sí, byte a byte | No sin fijar semilla y modelo |
| Evidencia | 2408.17008: repetir cabecera mejora recuperación en tablas. 2603.06976: agrupar por párrafo/estructura es la mejor familia | Anthropic: −35 % de fallos con contextual embeddings, −49 % con contextual BM25 |
| **CODEFEST** | **Admisible** | **GENERALMENTE ÚTIL, PERO NO ADMISIBLE** (R9 / spec §8.3) |

La técnica de Anthropic funciona y está bien medida. **Es irrelevante para nosotros**: mete un
decoder en la indexación y descalifica la entrega. Se registra para poder justificarlo en el
informe técnico, no para usarlo. Ninguna propuesta de este documento genera prefijos con un
modelo generativo.

Lo mismo aplica a Dense X Retrieval: el hallazgo (granularidad fina) es transferible; el
propositionizer basado en LLM, no.

### 12.2 `embedding_text` ≠ `returned_text`

La forma admisible de enriquecer:

```
embedding_text = "Documento: {title}. Observatorio: {obs}. Sección: {heading}. {texto}"
returned_text  = "{texto}"                       # sin modificaciones (Tabla 1: "texto original")
```

Esto es **legal y coherente con la Tabla 1**: el campo `texto` de la metadata es "el texto
original del fragmento, sin modificaciones", y el vector se calcula sobre otra cadena. La
correspondencia FAISS↔metadata (R4) no se ve afectada: sigue habiendo un vector por línea.

### 12.3 Beneficios plausibles, riesgos reales

**A favor:**
- Desambigua chunks huérfanos: una fila `Year: 2019 | Count: 4821` no dice de qué serie es;
  `Documento: PubMed AI publications timeline. Year: 2019 | Count: 4821` sí.
- Ayuda al caso cross-lingual: las 50 consultas están en español y el 56 % de los documentos en
  inglés. Un título traducible o un nombre de observatorio reconocible da un anclaje adicional.
- Sería la señal principal para los 27 documentos con `contenido_minimo`.

**En contra:**
- **Homogeneización.** Si los 111.775 chunks de `F1-AIINDEX-056` comparten el mismo prefijo, sus
  vectores se acercan entre sí y se alejan menos de la consulta: puede subir el recall del
  documento y hundir la discriminación **dentro** del documento — justo lo contrario de lo que
  necesita el NDCG@10 sobre fragmentos.
- **Dilución.** En un chunk de 21 palabras (XLSX mediano), un prefijo de 15 palabras es el 42 %
  del texto embebido. El prefijo puede dominar el vector.
- **Sesgo de agregación.** Un prefijo compartido puede hacer que documentos enteros suban o bajen
  en bloque, afectando F1@3 de forma difícil de atribuir.
- Para recuperación léxica futura (BM25), repetir el título en 111.775 chunks destroza el IDF de
  esos términos.

Es un candidato claro a ablación, con la advertencia de que **el riesgo escala con el número de
chunks por documento**, es decir, exactamente donde el corpus es más extremo.

---

## 13. Format-specific analysis

### 13.1 JSON — 954 docs, 10.475 bloques, 676.552 palabras

**Perfil:** mediana 37 palabras/bloque; p25 = 12; **41,5 % de los bloques ≤ 25 palabras**;
85,4 % ≤ 100. Solo 1,17 % supera 250. 11 bloques por documento de media, pero la distribución es
bimodal: Atlantic_Council mediana 40 palabras/bloque y CEEEP solo **2 bloques por documento**
(título + abstract).

**Lo que ya tenemos:** los `blocks` **son los párrafos originales de la fuente**. Es la frontera
que 2603.06976 identifica como la mejor familia, y llega gratis.

**Preguntas del brief, respondidas con el perfil:**

- *¿Paragraph packing o un bloque por chunk?* El bloque mediano usa el 15 % del presupuesto de
  250 palabras. Empaquetar párrafos consecutivos hasta 250 reduce JSON de 10.475 a 3.313 chunks
  (−68 %) y triplica el texto por vector. Hipótesis con base doble (literatura + perfil).
- *¿Sentence-aware packing dentro del bloque?* Irrelevante para el 98,8 % de JSON: casi ningún
  bloque supera 250. Solo hace falta para los 18 bloques > 3.000 palabras (CSIS).
- *¿Mantener section boundaries?* Solo existen en los 15 documentos de CENIA. Coste bajo,
  impacto acotado.
- *¿Añadir `title` a cada chunk?* Ver §12. En JSON el riesgo de homogeneización es **bajo**
  (mediana 11 chunks/documento) y el beneficio de desambiguación **alto** en los documentos de
  2 bloques (CEEEP, 80 docs) y en los 27 con `contenido_minimo`. Es el subcorpus donde el
  prefijo determinístico tiene mejor relación beneficio/riesgo.
- *¿Cuándo NO unir párrafos?* Cuando cruzan la frontera entre `alerta_meta`/`fields`
  (estructurado) y el cuerpo narrativo: son registros distintos. El parser los emite como
  bloques separados y esa frontera es información, no ruido.
- *¿Documentos muy cortos?* 580 documentos del corpus tienen < 250 palabras totales; la mayoría
  son JSON. Para ellos **el documento entero es un fragmento entregable**, y la pregunta de
  chunking desaparece.

### 13.2 PDF — 759 docs, 36.570 bloques (páginas), 13,3 M palabras

**Perfil:** mediana 353 palabras/página; p75 = 492; p95 = 740; máx. 4.630. **66,5 % de las
páginas superan las 250 palabras de salida y 22,5 % superan 512.** Es el único formato con un
problema real de exceso.

**Consecuencia inmediata:** *una página ≠ un chunk* no es una preferencia estética. Con
"bloque = chunk", dos de cada tres fragmentos de PDF **tendrían que dividirse en el momento de la
entrega** (R7) — es decir, la división ocurre igualmente; la pregunta es si ocurre en el índice
(donde afecta al vector y a la recuperación) o en la salida (donde solo afecta al texto
mostrado).

**Sentence packing cruzando páginas.** Medido:

| Política | Chunks de PDF | vs. página |
|---|---:|---:|
| Página = chunk | 36.570 | 1,00× |
| Packing dentro de la página, 250 palabras | 73.687 | 2,01× |
| Packing **cruzando** páginas, 250 palabras | 58.758 | 1,61× |
| Packing cruzando páginas, 350 palabras | 41.314 | 1,13× |

Cruzar páginas produce ~20 % menos vectores que respetarlas, porque recupera el desperdicio de
los finales de página. El coste es que un chunk puede empezar en la página 7 y acabar en la 8,
lo que complica `posicion` y la trazabilidad — pero `posicion` es ordinal dentro del **documento**
(R2), no dentro de la página, así que no hay conflicto formal.

**Contra la frontera dura de página:** un párrafo partido por un salto de página se recuperaría
peor. **A favor:** en un PDF a dos columnas o con notas al pie, el final de página suele ser
basura de layout, y unirlo con el inicio de la siguiente crea una unión semánticamente falsa.
Ninguno de los dos argumentos está medido en nuestro corpus. Es la pregunta abierta Q2.

**Headings, TOCs y orden.** Sin headings disponibles (§8.2). Título del documento: disponible,
pero cae al nombre de archivo cuando el PDF no trae metadata — variable en calidad. 530 páginas
candidatas a TOC identificadas. El orden se preserva por construcción (los parsers emiten
páginas en orden y R2 lo exige).

**No se cambia PyMuPDF.** La microvalidación sobre los 15 gold da recall 5-gram de **0,9977**
(mediana 1,0; mínimo 0,9658) y tasa de subcadena exacta del 93,3 %, frente a 0,9421
(pymupdf4llm) y 0,9340 (Docling). Docling detecta más headings (31 vs. 0) pero pierde texto y
cuesta 33,5 s frente a 1,0 s. La pregunta no es cambiar de extractor.

### 13.3 CSV / XLSX — 30 docs, 264.694 bloques, 16,07 M palabras

**Perfil:** CSV mediana 60 palabras/fila (97,9 % ≤ 100); XLSX mediana 21 (80,2 % ≤ 25). En
conjunto, **83 % de todos los bloques del corpus** salen de 30 archivos.

**La fila es atómica.** Lo dice la especificación (§2.1: "cada fila puede ser una unidad de
fragmentación independiente"), lo dicen los ADR-004/005, y lo respalda la literatura
(2408.17008: fila > tabla completa; repetir cabecera ayuda). **No se hace sentence split dentro
de una fila, no se resume, no se convierte la tabla en narrativa.** El ADR-004 ya revirtió
exactamente ese error.

**Lo que sí queda abierto: empaquetar filas consecutivas.** Medido:

| Política | Chunks CSV+XLSX | vs. fila |
|---|---:|---:|
| Fila = chunk | 264.694 | 1,00× |
| Packing de filas, 250 palabras | 73.687 | **0,28×** |
| Packing de filas, 500 palabras | 34.168 | **0,13×** |

Empaquetar filas hasta 250 palabras **elimina el 72 % de los vectores del corpus**. Coste
conceptual: un chunk deja de ser "una fila" y pasa a ser "un rango de filas contiguas". Riesgo:
si la evidencia relevante es una fila concreta, empaquetarla con otras 3 la diluye. Beneficio:
cada vector lleva 4× más información y el fragmento entregado usa el presupuesto completo.
STC (2605.00318) reporta exactamente este trade-off resuelto a favor del packing (−40 % chunks,
MRR 0,3576 → 0,5945), pero midiendo sobre BM25 híbrido.

**El caso `F1-AIINDEX-056`** (111.775 filas, no `042` como decía el brief; ver §3.6). Ni siquiera
con packing a 250 baja de 31.039 chunks (22 % del índice). **Distinguir con precisión:**

| Problema | Dónde se resuelve | Herramienta |
|---|---|---|
| Demasiados vectores por documento | Chunking | Packing de filas |
| Un documento gigante domina el top-10 de fragmentos | **Recuperación / post-proceso** | Cupo por documento, MMR |
| Un documento gigante domina el ranking documental | **Agregación (§8.6)** | Max-pooling en vez de suma; media de top-N |
| Filas casi idénticas compitiendo entre sí | **Post-proceso** | Diversificación |

**Ninguno de los tres últimos se arregla en el chunker**, y ninguno justifica borrar filas: cada
fila eliminada es información que no se puede recuperar (§22).

### 13.4 Timelines — 5 CSV

`F1-AIINDEX-055/058/060/062/064`, 44–67 filas de `Year: N | Count: M`, orden descendente
verificado, `extra["serie_temporal"] = True`.

Opciones de agrupación:

| Opción | Palabras/chunk | Chunks por archivo | Comentario |
|---|---:|---:|---|
| 1 fila | ~6 | 44–67 | Desperdicia el 97,6 % del presupuesto de salida |
| N filas consecutivas | ~6N | 44/N | Con N=40 la serie entera cabe en un fragmento |
| Serie completa | ~300 | 1 | Excede 250 palabras en los archivos de 67 filas |

**Una consulta sobre "la evolución de las publicaciones de IA" no la responde bien un chunk que
dice `Year: 2019 | Count: 4821`.** Agrupar años consecutivos preserva el orden, no inventa
agregaciones, es lossless y cabe en el presupuesto. Es el subcaso donde el packing de filas tiene
el argumento más fuerte — y la literatura no hace falta para verlo: 6 palabras contra un
presupuesto de 250.

Prohibido en cualquier caso (ADR-004): generar tendencias, calcular totales, traducir, reordenar.

### 13.5 PBF — 73 docs, 6.523 bloques, 507.421 palabras

**Perfil:** mediana 86 palabras/feature; 74,5 % ≤ 100; **0,09 % > 250**; máx. 260. Distribución
extraordinariamente uniforme — son registros de atributos, no prosa. `F3-AMAZONUW-067` tiene
1.162 features.

**Hechos ya verificados** (`pbf_summary.json`): 73/73 archivos MVT, 6.509 features en la
auditoría, **0 duplicados exactos intra-documento**, 3.871 ocurrencias duplicadas cross-file
(59,5 % de tasa de duplicación), 2.098 grupos duplicados cross-file. Los identificadores `fid` y
los concatenados aparecen hasta en 14 archivos distintos.

**No deduplicar globalmente.** Cada tile es un `doc_id` propio del índice de ADL; borrar features
de un documento porque aparecen en otro destruye la recuperabilidad de ese `doc_id` y con ella su
F1@3, sin ningún respaldo experimental.

Opciones de chunking:

- **Una feature = un chunk.** Ya es el contrato. Usa el 34 % del presupuesto de salida.
- **Agrupar features.** Packing a 250 palabras: 6.523 → 2.580 chunks (−60 %). Como las features
  de un tile son geográficamente vecinas, agruparlas es semánticamente defendible.
- **Prefijo determinístico de jerarquía geográfica** (país / región / municipio, cuando la
  columna exista): §12. Sería el complemento natural de los atributos booleanos sueltos, que sin
  contexto no dicen nada. Riesgo de homogeneización moderado (89 chunks/documento de media).

Los cuatro problemas del brief **son cuatro problemas distintos** y solo el primero es del
chunker:

1. **Chunking** — cuántas features por vector. Aquí.
2. **Supresión de duplicados** — 59,5 % de huellas repetidas entre archivos. Índice/post-proceso.
3. **Diversificación de resultados** — evitar 10 fragmentos cross-zoom equivalentes en el top-10.
   Post-proceso (MMR o cupo por documento).
4. **Agregación documental** — cómo puntuar 73 tiles casi idénticos para elegir 3 documentos.
   §8.6.

Resolver (2) dentro del chunker sería exactamente el error que §24 del brief advierte.

### 13.6 Imágenes — 9 docs, 38 bloques, 866 palabras

4 con texto real (transcripción manual: tablas ASAT, matriz de capacidades contraespaciales,
gráfico de barras, portada), 5 sin texto útil. Mediana 29 palabras/bloque, máximo 33.

**Política:** tratarlas como cualquier otra fuente textual corta. Con 38 bloques de 318.314
(0,012 %), **cualquier arquitectura específica para imágenes es coste sin retorno medible**.
Son el caso ideal para el relleno de §10: 29 palabras contra 250 de presupuesto, y el documento
completo cabe holgadamente en un fragmento.

### 13.7 TXT — 1 doc, 14 bloques, 598 palabras

`F2-SWF-113`. Bloques de mediana 29 palabras, máximo 108, ninguno > 250, ninguna oración
problemática. El ADR-006 ya reconstruyó las oraciones partidas por el scraper.

**Política:** la misma que JSON — es prosa ya segmentada en unidades naturales. El ADR-006 anota
la pregunta pendiente: si los bullets de "Major Updates" necesitan el encabezado de sección como
contexto, es un prefijo determinístico (§12), no un cambio de parser. Con 14 bloques, el impacto
es nulo en agregado; solo importa si ese documento aparece en el ground truth.

---

## 14. Retrieval chunk vs returned fragment

R7 desacopla las dos unidades. Cinco arquitecturas posibles:

| | A: chunk = fragmento | B: chunk pequeño → fragmento + vecinos | C: chunk grande → sub-fragmento ≤250 | D: hijo recuperado → padre expandido | E: sentence/window |
|---|---|---|---|---|---|
| **Recall** | medio | **alto** (unidad fina discrimina) | medio-alto | alto | **el más alto** |
| **Especificidad** | media | **alta** | baja (el vector mezcla temas) | alta | **la más alta** |
| **Completitud contextual del entregado** | baja si el chunk es corto | **alta** (rellena hasta 250) | alta | **la más alta** | alta |
| **Tamaño del índice** | 1× | **1×** (misma unidad embebida) | **0,3–0,5×** | 1,2–2× (hijos + padres) | 3–10× (una oración por vector) |
| **Vectores duplicados** | no | no | no | sí (padre e hijo) | no |
| **Cumplimiento de R6** | requiere división en salida (67 % de PDF) | por construcción | por construcción | requiere recorte | por construcción |
| **Complejidad** | mínima | **baja** | baja | media | media-alta |
| **Autorizado por §9.2.1** | sí | **explícitamente** | **explícitamente** | sí (el "padre" = vecinos del mismo doc) | sí |

**Lecturas:**

- **A no es realmente una opción neutra.** Con nuestro perfil, A obliga igualmente a dividir el
  67 % de los fragmentos de PDF en la salida y desperdicia el 66–92 % del presupuesto en los
  tabulares. "No hacer nada" también es una política, y es la peor medida en las dos direcciones.
- **B es la que mejor encaja con el corpus:** el 85 % del índice son bloques pequeños donde
  rellenar es gratis y no añade un solo vector.
- **C es obligatoria para PDF** en algún punto del pipeline; la pregunta es si el sub-fragmento se
  produce en indexación o en salida.
- **E (una oración por vector)** multiplicaría el índice por 3–10 (543.032 oraciones solo en PDF).
  Dense X Retrieval sugiere que rendiría bien, pero el coste de almacenamiento y la interacción
  con la concentración documental (§16) lo hacen poco atractivo aquí sin una medición previa.
- **B y C son combinables**: presupuesto de indexación intermedio, relleno hacia arriba para los
  cortos y división hacia abajo para los largos. Ese es el diseño que R7 parece anticipar.

---

## 15. Impact on NDCG@10

**Qué se juzga:** el contenido textual del campo `text` de cada uno de los 10 fragmentos, con
descuento logarítmico: la posición 1 pesa 1,0; la 10 pesa ≈0,29.

**Cómo influye la granularidad:**

1. **Chunks demasiado grandes → precisión baja.** Un vector que mezcla tres temas se parece
   medianamente a muchas consultas y mucho a ninguna. Chroma lo mide: precisión 7,0 con 200
   tokens contra 1,5 con 800. En nuestro PDF, una página mediana de 353 palabras ya está en la
   zona ancha.
2. **Chunks demasiado pequeños → contexto insuficiente en el texto entregado.** Los 15 gold
   observados tienen 138–180 palabras y **ninguno cabe en 100**. Un fragmento de 21 palabras
   (XLSX mediano) difícilmente será juzgado relevante aunque el vector haya acertado.
3. **Rellenar hasta 250 sube el techo de relevancia sin tocar el índice** (§10.3). Es la
   intervención con mejor relación beneficio/coste del análisis.
4. **El descuento premia ordenar bien el top-3.** Cualquier política que mejore el recall bruto
   pero desordene las tres primeras posiciones puede perder frente a una peor en recall.
5. **La redundancia no se penaliza explícitamente** en el NDCG estándar (por eso existen α-nDCG
   y las métricas *intent-aware*). Diez fragmentos casi idénticos del mismo tile PBF pueden
   puntuar bien en NDCG **y** hundir F1@3 — que es precisamente el desequilibrio que el conteo de
   Borda castiga.

**Dónde una estrategia puede mejorar NDCG y empeorar F1:** cualquiera que concentre los 10
fragmentos en 1–2 documentos. Si esos documentos son los correctos, NDCG sube; pero la señal para
elegir **tres** documentos distintos se degrada, y F1@3 cae. El solapamiento y los prefijos
determinísticos compartidos empujan en esa dirección.

---

## 16. Impact on F1@3

**Qué se juzga:** conjunto (el orden entre los 3 no puntúa). `P@3 = |D̂∩D*|/3`,
`R@3 = |D̂∩D*|/min(|D*|,3)`. Con ≥3 relevantes, cada documento errado cuesta precisión **y**
recall a la vez.

**Los tres mecanismos por los que el chunking afecta a F1@3:**

**(a) Concentración del índice.** Con `bloque = chunk`, 5 documentos poseen el 75,6 % de los
vectores (§3.6). Aunque cada chunk individual sea mediocre, la probabilidad de que **alguno** de
los 111.775 chunks de `F1-AIINDEX-056` entre en el top-k es alta por puro volumen. Con
max-pooling ingenuo, ese documento aparecerá en el top-3 de muchas consultas de F1. El packing
de filas mitiga (111.775 → 31.039) pero no elimina el sesgo.

**(b) Documentos con pocos chunks quedan estructuralmente en desventaja.** 73 documentos tienen
un solo bloque; 281 tienen ≤3; 580 tienen menos de 250 palabras. Frente a un documento con
decenas de miles de oportunidades, tienen una. Cualquier agregación que sume o promedie sobre
chunks los castiga; el max-pooling los trata con más justicia. **Es una decisión de agregación
(§8.6), no de chunking**, pero el chunking determina cuán extremo es el desequilibrio de partida.

**(c) Redundancia cross-documento.** Los PBF tienen 59,5 % de huellas duplicadas entre archivos
(2.098 grupos cross-file). Si una consulta territorial acierta en una feature, es probable que
los 14 tiles que contienen esa misma feature entren juntos, ocupando dos o tres de los tres
huecos documentales con contenido equivalente. Es el riesgo más concreto de perder F1@3 por
redundancia estructural del corpus.

**Herramienta correcta para cada uno:**

| Mecanismo | Chunking puede | Chunking NO puede | Quién resuelve |
|---|---|---|---|
| (a) Concentración | Reducir vectores por documento con packing | Igualar la representación | Agregación (max-pooling / media de top-N) |
| (b) Documentos pequeños | Evitar fragmentarlos innecesariamente | Compensar la asimetría | Agregación + normalización |
| (c) Redundancia | Nada útil sin destruir `doc_id` | — | Diversificación (MMR, cupo por documento) |

**Por eso la batería de ablaciones debe medir recuperación de fragmentos y recuperación de
documentos por separado (§20)**: una política puede mover las dos en direcciones opuestas, y el
Borda del leaderboard suma ambas tablas.

---

## 17. Index-size and runtime implications

Simulado sobre los 1.826 documentos reales (`packing_simulation.csv`). Las políticas: **P0** un
bloque = un chunk; **P1** empaquetado por oraciones **dentro** del bloque; **P2** empaquetado
cruzando bloques del mismo documento; **P3** = P2 con solapamiento de 1 unidad.

| Política | json | pdf | csv | xlsx | pbf | **Total** | **×P0** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **P0** bloque = chunk | 10.475 | 36.570 | 255.793 | 8.901 | 6.523 | **318.314** | 1,00 |
| P1 @120 | 12.944 | 139.812 | 255.793 | 8.901 | 6.523 | **424.025** | 1,33 |
| P1 @180 | 11.599 | 97.179 | 255.793 | 8.901 | 6.523 | **380.047** | 1,19 |
| P1 @250 | 11.151 | 73.687 | 255.793 | 8.901 | 6.523 | **356.107** | 1,12 |
| P1 @350 | 10.919 | 58.343 | 255.793 | 8.901 | 6.523 | **340.531** | 1,07 |
| P2 @120 | 6.830 | 128.198 | 184.150 | 1.751 | 5.164 | **326.114** | 1,02 |
| P2 @180 | 4.487 | 83.214 | 107.570 | 1.133 | 4.113 | **200.533** | 0,63 |
| **P2 @250** | 3.313 | 58.758 | 72.882 | 805 | 2.580 | **138.353** | **0,43** |
| P2 @350 | 2.504 | 41.314 | 49.640 | 571 | 1.705 | **95.746** | 0,30 |
| P2 @500 | 1.950 | 28.516 | 33.770 | 398 | 1.163 | **65.809** | 0,21 |
| P3 @250 (ov 1) | 3.621 | 67.495 | 100.999 | 889 | 4.317 | **177.336** | 0,56 |

P1 no cambia nada en tabulares por construcción (la fila es atómica y casi nunca supera el
presupuesto): su único efecto es dividir páginas de PDF.

### 17.1 Almacenamiento

`storage_vectores ≈ n_chunks × embedding_dim × 4 bytes` (float32, `IndexFlatIP`):

| n_chunks | dim 384 | dim 768 | dim 1024 |
|---:|---:|---:|---:|
| 318.314 (P0) | 466 MiB | **933 MiB** | 1,21 GiB |
| 356.107 (P1@250) | 522 MiB | 1,02 GiB | 1,36 GiB |
| 177.336 (P3@250) | 260 MiB | 520 MiB | 693 MiB |
| **138.353 (P2@250)** | 203 MiB | **405 MiB** | 540 MiB |
| 65.809 (P2@500) | 96 MiB | 193 MiB | 257 MiB |

**Esto multiplica por el número de encoders** (§4.4 de `CLAUDE.md`: un índice independiente por
encoder). Dos encoders de 768 dimensiones con P0 son ~1,9 GiB de vectores; con P2@250, ~810 MiB.
La entrega va a una carpeta compartida en la nube, así que no es bloqueante, pero sí condiciona
el tiempo de subida y el de carga en la máquina del evaluador (que debe poder correr en CPU).

### 17.2 Metadata

`metadata.jsonl` guarda el `texto` completo de cada chunk. Sin solapamiento el texto total es
constante (~30,5 M palabras ≈ **200 MB** de UTF-8) más ~150 bytes de campos fijos por línea:

- P0: 200 MB + 318.314 × 150 B ≈ **248 MB**
- P2@250: 200 MB + 138.353 × 150 B ≈ **221 MB**
- P3@250 (overlap): el texto **se duplica parcialmente** → ~250 MB + overhead

### 17.3 Tiempo

- Segmentación con `pysbd`: **461 s** para el corpus completo incluyendo todas las simulaciones
  (medido). Despreciable frente a la codificación.
- Codificación: proporcional al número de chunks × longitud. P2@250 tiene 2,3× menos vectores
  que P0 pero cada uno es más largo; el coste total en tokens es similar salvo por el
  solapamiento, que es coste puro añadido.
- **El semantic chunking añadiría ~573.000 embeddings de oración por cada variante probada**
  (§9.1), y habría que repetirlos si se cambia de encoder.

---

## 18. Evidence matrix

`Merece ablación`: **alta** / **media** / **baja** / **no admisible**. No hay ganador declarado.

| Estrategia | Evidencia externa | Compatible con el corpus | Legal en CODEFEST | Coste | Riesgo | Merece ablación |
|---|---|---|---|---|---|---|
| **Fixed-size (carácter/token)** | Negativa: peor de 36 métodos, nDCG@5 0,244 (2603.06976) | Indiferente | **No** — viola R1 salvo con retroceso oracional | nulo | Descalificación | **no admisible** en su forma cruda |
| **Sentence packing** | Positiva/neutra: a la par del semántico, mejor que token (2601.14123); mejor baseline naive en 2409.04701 | Sí, todos los formatos narrativos | Sí — satisface R1 por construcción | bajo (4,8 ms/página) | Falla en 1.105 bloques sin frontera oracional; sin reglas `pt` en pysbd | **alta** |
| **Paragraph packing** | **La más positiva**: mejor familia global, nDCG@5 0,459 (2603.06976) | **Sí en JSON** (los párrafos ya vienen); **no en PDF** (bloque = página) | Sí | nulo — la frontera ya existe | Solo cubre el 52 % de los documentos | **alta** |
| **Recursive** | Mixta: competitivo con coste bajo (2606.00881); dependiente del tamaño (Chroma) | Redundante: los parsers ya entregan bloques segmentados | Solo si el nivel más profundo es la oración | bajo | Complejidad sin ganancia clara sobre sentence packing | **baja** |
| **Overlap** | **Negativa × 2**: sin mejora medible (2601.14123); castiga IoU (Chroma) | +28–50 % de vectores medidos | Sí (con overlap por oración) | medio-alto | Duplicación, concentración documental, peor F1@3 | **media** — solo como contraste contra `overlap = 0` |
| **Structural (headings/secciones)** | Positiva (2603.06976; Unstructured) | **Parcial**: sí en JSON/TXT, **no en PDF** (0 headings con PyMuPDF) | Sí | bajo donde existe | Exigiría cambiar de extractor para PDF, con peor recall de texto | **media** (limitada a JSON) |
| **Semantic** | **Contradictoria**: no compensa (2410.13070, 2606.00881) vs. mejor precisión/IoU (Chroma); por debajo de párrafo (2603.06976) | Corpus monotemático → el escenario donde menos rinde | Sí, salvo variantes con LLM | **alto**: ~573 K embeddings de oración por variante | Depende del encoder que aún no existe | **baja** |
| **Sentence window** | Positiva por analogía (Dense X: +19–55 % EM@100) | Índice ×3–10 (543 K oraciones solo en PDF) | Sí | alto en almacenamiento | Agrava la concentración documental | **baja** |
| **Parent-child / small-to-big** | Documentado como patrón (LlamaIndex); sin benchmark controlado | Sí | Sí — **§9.2.1 lo autoriza literalmente** | bajo (sin vectores extra si el padre no se indexa) | Diluir la evidencia al expandir | **alta** |
| **Prefijo de contexto determinístico** | Positiva en tablas (2408.17008: repetir cabecera mejora) | Sí; toda la señal ya existe en `RawDoc`/`extra` | **Sí** — no interviene ningún decoder | nulo | Homogeneización en documentos de decenas de miles de chunks; dilución en chunks de 21 palabras | **media-alta** |
| **Contextual retrieval generado** | Positiva y bien medida (−35 % / −49 % de fallos) | Sí técnicamente | **NO** — decoder en indexación (R9) | alto | **Descalificación de la entrega** | **no admisible** |
| **Late chunking** | Positiva pero modesta (+1,5/+1,9 nDCG@10); mixta en réplica independiente (2504.19754) | Exige encoder de contexto largo + mean pooling, que aún no está elegido | Sí (encoder, no decoder) | medio-alto | **No es chunking**: es encoder+pooling; mezclarlo confunde variables | **baja** — y en la fase de encoder, no en esta |
| **Format-aware** | Positiva: el ganador cambia por dominio (2603.06976) y por dataset/encoder (2505.21700); fila > tabla en tabulares (2408.17008, 2605.00318) | **Sí — el perfil lo exige**: 250 palabras es techo para el 67 % de páginas y objetivo inalcanzable para el 80 % de filas XLSX | Sí | bajo (una rama por formato) | Más código que mantener; más difícil atribuir causas | **alta** |

---

## 19. Candidate hypotheses for ablation

Siete experimentos. Cada uno cambia **una** hipótesis. Ninguno se ejecuta todavía.

### E0 — Baseline: bloque = chunk

- **Hipótesis:** el contrato de los parsers ya produce una partición utilizable; las políticas
  posteriores deben demostrar que mejoran algo.
- **Qué cambia:** nada. Un bloque, un chunk. División en salida solo cuando el bloque supera 250
  palabras (obligado por R6), respetando oraciones.
- **Qué se fija:** encoder, índice, `top_k`, agregación, conjunto de consultas.
- **Beneficio esperado:** ninguno; es la referencia. Y es el camino más rápido a una **entrega
  válida de punta a punta**, que según `CLAUDE.md` §6 es la prioridad de la fase 1.
- **Riesgo principal:** 318.314 vectores (933 MiB a dim 768); 5 documentos con el 75,6 % del
  índice; el 66 % de los fragmentos de PDF hay que dividirlos igualmente.
- **Qué medir:** todo el bloque A, B, C y D de §20. Es la línea contra la que se compara el resto.

### E1 — Sentence-aware packing dentro del bloque

- **Hipótesis:** dividir las páginas largas de PDF en unidades de un presupuesto fijo, respetando
  oraciones, mejora la precisión del fragmento sin perder recall.
- **Qué cambia:** solo los bloques que superan el presupuesto se dividen. Nada se une.
- **Qué se fija:** todo lo demás, incluida la política de tabulares (fila = chunk).
- **Beneficio esperado:** más precisión en PDF, que es donde está el 43 % de las palabras del
  corpus y el 100 % de los gold del devset. Chroma predice una mejora fuerte de precisión al
  bajar de ~800 a ~200 tokens.
- **Riesgo:** +12 % de vectores a 250 palabras (356.107); si la evidencia gold abarca 156
  palabras de media, un presupuesto pequeño podría partirla en dos chunks.
- **Qué medir:** además de A–D, la fracción de gold del devset que queda **contenida en un solo
  chunk** frente a partida en dos.

### E2 — Sentence/row packing cruzando bloques del mismo documento

- **Hipótesis:** unir bloques consecutivos hasta el presupuesto elimina el desperdicio de las
  unidades cortas (el 85 % del índice) y reduce drásticamente el número de vectores sin perder
  información.
- **Qué cambia:** el chunk pasa de ser "un bloque" a "un rango contiguo de bloques del mismo
  documento". La fila y la feature siguen siendo **atómicas** (no se parten), solo se agrupan.
- **Qué se fija:** presupuesto, encoder, agregación.
- **Beneficio esperado:** 318.314 → 138.353 vectores (−57 %); cada fragmento entregado usa el
  presupuesto completo; `F1-AIINDEX-056` baja de 111.775 a 31.039 chunks.
- **Riesgo principal:** diluir una fila concreta relevante entre otras 3; unir dos páginas de PDF
  sin relación temática; degradar la especificidad que Dense X sugiere que importa.
- **Qué medir:** A–D, y explícitamente la comparación **fragmento vs. documento** (una política
  puede mejorar F1@3 por reducción de ruido y empeorar NDCG@10 por dilución).

### E3 — Overlap de 1 oración/fila sobre la mejor de E1/E2

- **Hipótesis (a refutar):** el solapamiento recupera la evidencia que queda partida entre dos
  chunks.
- **Qué cambia:** solo el solapamiento. Todo lo demás igual al ganador de E1/E2.
- **Qué se fija:** presupuesto, política de fronteras, encoder.
- **Beneficio esperado:** **bajo.** Dos estudios independientes no encuentran mejora medible.
  Se corre para tener el número propio, no porque se espere ganar.
- **Riesgo:** +28 % de vectores a 250 palabras; texto duplicado en `metadata.jsonl`; más
  concentración documental en el top-10 → peor F1@3.
- **Qué medir:** A–D **y** la diversidad documental del top-10, que es donde se espera el daño.
  Si no gana claramente, se descarta y queda documentado en `docs/ablaciones.md`.

### E4 — Prefijo de contexto determinístico (`embedding_text` ≠ `returned_text`)

- **Hipótesis:** anteponer al texto embebido información que ya existe (`title`, observatorio,
  hoja, sección, jerarquía geográfica) mejora la recuperación cross-lingual y desambigua los
  chunks huérfanos, sin tocar el texto entregado.
- **Qué cambia:** únicamente la cadena que se embebe. `texto` en la metadata sigue siendo el
  original (Tabla 1), y `metadata.jsonl` mantiene una línea por vector (R4).
- **Qué se fija:** fronteras, presupuesto, encoder.
- **Beneficio esperado:** mayor en JSON (11 chunks/documento de media, 41 % de bloques ≤ 25
  palabras), en los 27 documentos con `contenido_minimo` y en los timelines.
- **Riesgo principal:** **homogeneización** en los documentos gigantes — 31.039 chunks con el
  mismo prefijo se acercan entre sí; y **dilución** en chunks de 21 palabras donde el prefijo es
  el 42 % del texto embebido.
- **Qué medir:** A–D, desglosado **por formato**, más la dispersión intra-documento de los
  vectores (si cae mucho, la homogeneización está ocurriendo). **Ningún prefijo generado por un
  modelo.**

### E5 — Política por formato (format-aware)

- **Hipótesis:** una política única no puede servir a la vez a páginas de 353 palabras y a filas
  de 21; separar la política narrativa de la tabular gana en ambas.
- **Qué cambia:** dos ramas — narrativa (JSON/PDF/TXT/imágenes) con packing por oraciones, y
  tabular (CSV/XLSX/PBF) con packing de filas atómicas —, cada una con su propio presupuesto.
- **Qué se fija:** encoder, agregación, conjunto de consultas.
- **Beneficio esperado:** es la única hipótesis que el perfil del corpus **exige** por sí solo
  (§3.2), y la literatura respalda que el ganador cambia por tipo de contenido.
- **Riesgo:** más superficie de código; atribuir la mejora a la rama correcta requiere medir por
  formato, no solo en agregado.
- **Qué medir:** A–D **desglosado por formato y por fenómeno** (F1 es tabular; F2/F3 son prosa).

### E6 — Retrieval unit ≠ evidence unit (relleno y división en la salida)

- **Hipótesis:** con las fronteras del índice fijas, expandir los fragmentos cortos con sus
  vecinos hasta ~240 palabras y dividir los largos en sub-fragmentos sube NDCG@10 **sin añadir un
  solo vector**.
- **Qué cambia:** solo el post-proceso de la salida. El índice no se toca.
- **Qué se fija:** absolutamente todo lo demás.
- **Beneficio esperado:** el mayor por unidad de esfuerzo según §10.3 — sobre el 85 % del índice
  los fragmentos crudos usan entre el 8 % y el 34 % del presupuesto disponible.
- **Riesgo:** rellenar con vecinos irrelevantes diluye la evidencia; los sub-fragmentos de un
  chunk largo compiten entre sí por posiciones del top-10 y pueden desplazar a otros documentos,
  dañando F1@3.
- **Qué medir:** NDCG@10 proxy y F1@3 **en paralelo**, más la distribución de palabras por
  fragmento entregado y el número de documentos distintos en el top-10.
- **Nota:** E6 es ortogonal a E0–E5 y puede aplicarse encima de cualquiera de ellas. Por eso va
  al final del orden (§21).

---

## 20. Candidate parameter ranges

Rangos, no valores. Derivados del perfil y de la literatura; se cierran midiendo.

### 20.1 Presupuesto para contenido narrativo (JSON, PDF, TXT, imágenes)

| Candidato | De dónde sale | Advertencia |
|---:|---|---|
| **~160 palabras** | Mediana de los 15 gold (156); p50 de la ventana observada 138–180 | El devset es diminuto y sospechosamente uniforme (§4.1). **No usar como objetivo, solo como punto de anclaje** |
| **~240 palabras** | Justo bajo el límite duro de 250 (R6), con margen como pide `CLAUDE.md` §2.3 | Hace coincidir la unidad indexada con la entregable: simplifica, pero renuncia a la flexibilidad de R7 |
| **~350 palabras** | p50 real de una página de PDF (353); permite que la unidad indexada sea mayor que la entregada | En ES/PT ≈ 525–700 tokens: **truncaría con un encoder de 512** (§6.2) |

Tres valores, no cinco. Y la elección final **está condicionada al tokenizador del encoder**: la
regla operativa es que `presupuesto_palabras × ratio_subpalabra(ES/PT)` debe caber en
`model_max_seq_length` con margen.

### 20.2 Presupuesto para contenido tabular (CSV, XLSX, PBF)

| Candidato | Efecto medido | Comentario |
|---|---|---|
| **1 fila** (baseline) | 271.217 chunks | El contrato actual. Desperdicia 66–92 % del presupuesto de salida |
| **~240 palabras** | 76.267 chunks (−72 %) | Alinea el chunk con el fragmento entregable |
| **~500 palabras** | 35.331 chunks (−87 %) | Solo tiene sentido con E6: el chunk excede el límite de salida a propósito |

Para timelines, expresar el presupuesto además en **filas** (~40 filas ≈ una serie completa) y
verificar que no se rompe el orden.

### 20.3 Solapamiento

**`0` y `1 oración` (o `1 fila`). Nada más.** Dos estudios independientes no encuentran mejora
con solapamiento; probar 0 %/10 %/20 %/30 % sería gastar cómputo en refutar algo ya refutado dos
veces. `0` es el baseline obligatorio.

### 20.4 Matriz resultante

Con 3 presupuestos narrativos × 3 tabulares × 2 solapamientos saldrían 18 configuraciones antes
de tocar E4/E6: explosión combinatoria sobre 8 consultas de devset. **Restricción propuesta:**
fijar primero el presupuesto narrativo con solapamiento 0 y política tabular = baseline (3
corridas), después el tabular con el narrativo ya fijado (2 corridas más), y solo entonces probar
solapamiento (1 corrida). **6 corridas en vez de 18.**

---

## 21. Recommended experimental order

Criterio: primero lo barato y lo que reduce riesgo de entrega; después lo que cambia una sola
variable; al final lo que depende de decisiones aún no tomadas.

| # | Experimento | Por qué en esta posición | Prerrequisito |
|---|---|---|---|
| 1 | **E0 — baseline** | Da una entrega válida de punta a punta y la referencia numérica. `CLAUDE.md` §6: entrega válida antes que métrica | Encoder elegido (cualquiera, congelado para toda la batería) |
| 2 | **E5 — format-aware** | Es la única hipótesis que el perfil exige por sí solo, y define la estructura sobre la que se afinan E1/E2. Además es la que más reduce el índice | E0 |
| 3 | **E1 vs. E2 — dónde poner las fronteras** | Con la separación por formato ya hecha, comparar "dividir" contra "unir" es una sola variable | E5 |
| 4 | **E6 — relleno y división en salida** | No toca el índice: se puede aplicar sobre el ganador de (3) sin reindexar. Máximo beneficio por coste | Ganador de (3) |
| 5 | **E4 — prefijo determinístico** | Requiere reindexar y su riesgo depende del número de chunks por documento, que solo se conoce tras (3) | Ganador de (3) |
| 6 | **E3 — overlap** | Se espera que no gane; se corre para tener el número y cerrar la pregunta con evidencia propia | Ganador de (3)/(5) |
| — | *Semantic chunking* | Solo si (3) muestra que las fronteras importan mucho **y** sobra tiempo. Coste ~573 K embeddings por variante, y hay que rehacerlo si cambia el encoder | Fase 2 avanzada |
| — | *Late chunking* | **No pertenece a esta batería.** Es encoder + pooling: evaluar en la fase de selección de encoder, y solo si el elegido tiene contexto largo y mean pooling | Encoder de contexto largo |

**Regla transversal:** el encoder se congela para toda la batería. Comparar chunking con dos
encoders distintos no mide chunking.

---

## 22. Practices explicitly rejected

Rechazadas por evidencia, no por gusto. Cada una con lo que la sostiene.

1. **Cortar oraciones a mitad (fixed-size por carácter o token sin retroceso).**
   Viola R1 (spec §3.3) — motivo suficiente. Y además es la peor familia medida: nDCG@5 0,244 y
   P@1 2–3 % frente a 0,459 y 24 % del agrupamiento por párrafos (2603.06976). La regla del reto
   y la evidencia coinciden.

2. **Prefijos de contexto generados por un LLM (contextual retrieval de Anthropic, Dense X
   propositionizer, LumberChunker, chunking asistido por decoder).**
   Funcionan (−35 % / −49 % de fallos de recuperación) y son **inadmisibles**: meten un decoder en
   la indexación (spec §8.3, `CLAUDE.md` §2.1). Descalifican la entrega. El sustituto legal es el
   contexto determinístico de §12.

3. **Resumir, sintetizar o narrar filas de tabla.**
   Ya se probó y se revirtió en el ADR-004: la narrativa sintetizada hacía irrecuperables los años
   intermedios de los timelines y afirmaba agregaciones (total, pico) que el archivo no contiene.
   Es lossy y, si se hiciera con un modelo, además ilegal.

4. **Tratar las filas como prosa y aplicarles segmentación de oraciones.**
   Una fila es `columna: valor | columna: valor`: no tiene estructura oracional. La literatura
   específica de tablas apunta al contrario — fila atómica con cabecera repetida es lo que mejora
   la recuperación (2408.17008; 2605.00318).

5. **Deduplicar PBF globalmente destruyendo `doc_id`.**
   Hay 59,5 % de huellas duplicadas cross-file, pero **0 duplicados exactos intra-documento**.
   Cada tile es un `doc_id` del índice de ADL; vaciar un documento porque su contenido aparece en
   otro lo hace irrecuperable y cuesta F1@3 sin ninguna medición que lo respalde. La redundancia
   se trata en diversificación de resultados, no en el chunker.

6. **Borrar filas de los CSV/XLSX gigantes para "arreglar" la concentración.**
   `F1-AIINDEX-056` tiene el 35 % del índice; la respuesta no es tirar 111.775 filas. El packing
   lo reduce a 31.039, y el resto es un problema de agregación y diversificación (§16, §24). Cada
   fila borrada es evidencia que jamás podrá recuperarse.

7. **Fijar un tamaño de chunk universal copiado de una recomendación externa.**
   2505.21700 muestra que el óptimo cambia por dataset **y por encoder** (Stella prefiere grande,
   Snowflake pequeño); 2603.06976, que cambia por dominio. Y nuestro perfil muestra que 250
   palabras es un techo para el 67 % de las páginas de PDF y un objetivo inalcanzable para el
   80 % de las filas de XLSX. No existe ese número.

8. **Solapamiento como default sin medirlo.**
   Dos estudios independientes no encuentran mejora (2601.14123: "no mejora BERTScore ni EM";
   Chroma: castiga el IoU), y cuesta +28–50 % de vectores en nuestro corpus. Puede probarse (E3),
   pero **no** asumirse.

9. **Adoptar late chunking por reputación, o mezclarlo con la primera ablación de chunking.**
   Mejora +1,5/+1,9 nDCG@10 en su propio paper, pero la réplica independiente encuentra
   resultados mixtos según el encoder (2504.19754), exige contexto largo y mean pooling, y **no
   es una decisión de fronteras**: es encoder + pooling. Mezclarlo aquí confundiría dos variables.

10. **Cambiar de extractor de PDF para conseguir headings.**
    Docling detecta 31 headings frente a 0 de PyMuPDF, pero con recall 5-gram 0,9340 frente a
    **0,9977** sobre los gold, y 33,5 s frente a 1,0 s. Se evalúa el texto, no los headings.
    Además los parsers están congelados.

---

## 23. Open questions

Solo lo que de verdad necesita un experimento o una decisión de equipo.

| # | Pregunta | Por qué no se puede responder ahora | Cómo se resuelve |
|---|---|---|---|
| **Q1** | ¿Una política única o políticas por formato? | El perfil hace el argumento a favor de separar, pero no está medido contra la métrica | E5 |
| **Q2** | En PDF, ¿la página es frontera **dura** o **blanda**? | Cruzar páginas da −20 % de vectores; partir un párrafo por un salto de página puede costar recall. Ninguno medido aquí | E1 vs. E2 restringido a PDF |
| **Q3** | ¿Empaquetar filas de tabla o dejar fila = chunk? | −72 % de vectores contra riesgo de dilución. La literatura de tablas apoya el packing, pero midiendo sobre BM25 | E5/E2 sobre el subcorpus tabular |
| **Q4** | ¿El prefijo determinístico ayuda o homogeneiza? | Depende del número de chunks por documento, que cambia con la política elegida | E4, medido por formato |
| **Q5** | ¿Qué hacer con los 1.105 bloques sin frontera oracional dentro de 250 palabras? | Es una violación **inevitable** de R1 en el 2,35 % de los bloques narrativos. Hay que elegir política de escape: cortar por palabra y anotarlo, truncar, o excluir el bloque | Decisión de equipo + verificación de que el validador del comité no penalice |
| **Q6** | ¿Qué segmentador para los 108 documentos en portugués? | `pysbd` no tiene reglas `pt`. Opciones: `es` como aproximación, SaT/wtpsplit (mejor pero añade un modelo), o regla propia | Prueba sobre una muestra de INPE |
| **Q7** | ¿Se filtran las 530 páginas de TOC? | Malas como evidencia, pero filtrarlas mal cuesta recall documental | Ablación propia, **no** dentro del chunker |
| **Q8** | ¿Qué presupuesto en **tokens** corresponde al presupuesto en palabras? | No hay encoder elegido; el ratio ES/PT es ~1,5–2,0 y 250 palabras ya rozan los 512 tokens | Medir con el tokenizador real, antes de fijar nada |
| **Q9** | ¿Cómo se evita que los documentos gigantes dominen? | Es un problema de agregación y diversificación, no de chunking | Fuera de esta batería: pertenece al diseño de recuperación (§8.6) |

---

## 24. Sources

17 fuentes externas revisadas y fichadas en
[`chunking-sources.md`](chunking-sources.md), cubriendo **1998 y 2023–2026**. Resumen:

| Bloque | Fuentes | Qué aportan |
|---|---|---|
| Evidencia comparativa | Qu et al. 2024 (2410.13070); Chroma 2024; 2603.06976; 2601.14123; 2606.00881; 2505.21700 | Los números que sostienen "empezar barato": semantic no compensa, overlap no mejora, párrafo > carácter, no hay tamaño universal |
| Contexto añadido | Late Chunking (2409.04701); Contextual Retrieval (Anthropic); Reconstructing Context (2504.19754) | Distinguen contexto determinístico (legal) de generado (ilegal aquí), y sitúan late chunking como encoder+pooling |
| Granularidad | Dense X Retrieval (EMNLP 2024); LlamaIndex (sentence window / auto-merging); DICE (2606.18781) | Sostienen la arquitectura *retrieval unit ≠ evidence unit* y la agregación por chunk para el ranking documental |
| Estructura y tablas | STC (2605.00318); 2408.17008; Unstructured; SaT (2406.16678) | Validan fila atómica + cabecera repetida, y aportan el mínimo-además-de-máximo y el problema del portugués |
| Diversificación | MMR (Carbonell & Goldstein, SIGIR 1998) | La herramienta para el riesgo de redundancia, en post-proceso y no en el chunker |

**Balance deliberado:** cuatro de las seis fuentes del bloque comparativo reportan resultados
**negativos o neutros** para las técnicas complejas. No se hizo *cherry-picking* de los trabajos
donde una estrategia gana; cuando dos fuentes se contradicen (semantic chunking, §9.3) se
presentan ambas y se explica por qué difieren.

**Fuentes descartadas:** blogs SEO y listados de "mejores estrategias" sin experimento, artículos
de Medium sin benchmark, y trabajos de chunking multimodal (documento-como-imagen), fuera del
alcance de una entrega que se evalúa sobre el campo `text`. Detalle en §7 de la bitácora.

---

## 25. Reproducibility

### 25.1 Estado del repositorio al generar este documento

```
branch:        dev  (== origin/dev)
HEAD:          b75993b6bbf8f8b7987f71a9378edb4e7abc08b0
dev vs main:   dev +3 commits, 0 detrás
working tree:  limpio al iniciar
```

### 25.2 Volcados de `RawDoc` usados

| Archivo | Docs | Origen |
|---|---:|---|
| `data/interim/final_json.jsonl` | 954 | preexistente, ADR-001 |
| `data/interim/raw_pdf_ocr.jsonl` | 759 | preexistente, **volcado de referencia con OCR** (ADR-003) |
| `data/interim/final_csv.jsonl` | 26 | preexistente, ADR-004 |
| `data/interim/final_xlsx.jsonl` | 4 | preexistente, ADR-005 |
| `data/interim/final_images.jsonl` | 9 | preexistente, ADR-002 |
| `data/interim/final_pbf_txt.jsonl` | 74 | **generado en esta sesión** (no existía volcado de PBF ni de TXT) |

`final_pdf.jsonl` (sin OCR, 66 documentos con `contenido_minimo`) **no se usó**: el ADR-003 lo
declara reemplazado por `raw_pdf_ocr.jsonl` (0 con `contenido_minimo`).

### 25.3 Comandos exactos

```bash
# 1. Volcado de los 73 PBF + 1 TXT (unicos formatos sin volcado previo).
#    No modifica src/extract/: solo ejecuta los parsers ya congelados.
uv run --extra cpu python -m src.extract --formato pbf txt \
    -o data/interim/final_pbf_txt.jsonl
# -> INFO src.extract: extraidos 74 documentos, 0 fallidos / 508.019 palabras

# 2. Perfil completo del corpus + simulacion de politicas de empaquetado.
uv run --extra cpu python scripts/research/profile_rawdocs_for_chunking.py
# -> data/interim/research/chunking/  (461 s en la maquina del equipo)

# Variantes utiles:
#   --sin-oraciones   omite pysbd y la simulacion (perfil de tamanos en ~20 s)
#   --limite N        corta tras N documentos (pruebas)
#   --salida RUTA     escribe en otro directorio

# 3. Lint del script de investigacion (no toca codigo productivo).
uv run --with ruff ruff check  --line-length 100 scripts/research/
uv run --with ruff ruff format --check --line-length 100 scripts/research/
```

### 25.4 Salidas generadas

Todas bajo `data/interim/research/chunking/` (dentro de `data/`, ignorado por git — se regeneran,
no se commitean):

| Archivo | Contenido |
|---|---|
| `summary.json` | Totales, percentiles globales, umbrales, oraciones por formato, perfil gold, top-20 documentos por bloques |
| `doc_profile.csv` | Una fila por documento (1.826): formato, fenómeno, observatorio, idioma, bloques, palabras, min/mediana/max de bloque |
| `block_profile_by_formato.csv` | Percentiles y umbrales por formato |
| `block_profile_by_fenomeno.csv` | Ídem por fenómeno |
| `block_profile_by_observatorio.csv` | Ídem por observatorio (20 filas) |
| `packing_simulation.csv` | Número de chunks por política (P0–P3) × presupuesto (120/180/250/350/500) × formato |
| `gold_fragment_profile.csv` | Los 15 gold: palabras, caracteres, oraciones, oración más larga |
| `long_sentences.csv` | Los 1.105 bloques con una "oración" > 250 palabras (doc_id, formato, idioma, tamaños) |

### 25.5 Determinismo y advertencias

- La detección de idioma fija `DetectorFactory.seed = 42`; sin eso `langdetect` no es
  reproducible.
- La segmentación en portugués usa el *ruleset* español (`pysbd` no soporta `pt`): las cifras de
  oraciones de los 108 documentos en portugués son **aproximadas**.
- Las políticas simuladas (P0–P3) son **conteos**, no una implementación propuesta: cuentan
  cuántos vectores produciría cada política para poder comparar coste antes de elegir ninguna.
  El chunker real no existe todavía y este documento no lo especifica.
- El script **no escribe en `src/`** ni modifica ningún parser.
