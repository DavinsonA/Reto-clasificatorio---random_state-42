# Sondeo del corpus y estrategias de extracción

**Fecha:** 2026-08-03 · **Estado:** hallazgos, sin código todavía
**Fuente:** `data/Indice_Datos_Codefest.xlsx` + inspección directa de los 1.837 archivos en disco.

Este documento cierra la **Fase 0** (reconciliación del corpus, bloqueante según `CLAUDE.md` §6) y fija el terreno para escribir los parsers. Ninguna cifra de aquí es estimada: todas salen de recorrer el corpus completo, salvo donde se indica "muestra".

---

## 1. Los cuatro hallazgos que cambian el plan

### 1.1 Las 50 consultas están **todas en español**; el corpus no

Se extrajeron las 50 consultas de `Extracto_Preguntas_50_v2.pdf` y se detectó el idioma de cada una: **50 de 50 en español.** La §10.1 de la especificación decía que se distribuirían "en los tres idiomas del corpus"; no es lo que llegó.

En cambio el corpus sí es trilingüe. Sobre una muestra de 156 JSON con cuerpo:

| Idioma | Documentos | Observatorios |
|---|---:|---|
| Español | 74 | Alertas_Tempranas, CEEEP |
| Inglés | 71 | Atlantic_Council, CSIS, SIPRI, SWF, CEOBS, ESA |
| Portugués | 11 | INPE |

**Consecuencia:** la recuperación *cross-lingual* no es un caso extra, es **el caso central**. Toda consulta en español debe alcanzar documentos en inglés y portugués. Un encoder que no alinee bien los tres idiomas en el mismo espacio pierde de entrada el grueso del corpus de los fenómenos 1 y 2, que son mayoritariamente en inglés. Esto sube el peso del criterio "multilingüe nativo" (§4.3) por encima de todos los demás.

Vale la pena verificar antes de indexar si el encoder candidato mantiene la alineación es→en en textos largos, no solo en frases cortas tipo STS.

### 1.2 Los nombres de archivo **no son únicos**; `fuente` tiene que ser la ruta

| Clave | Valores únicos | ¿Sirve como identificador? |
|---|---:|---|
| `DOC_ID` | 1.826 / 1.826 | ✅ |
| Ruta relativa (`Carpeta` + nombre) | 1.826 / 1.826 | ✅ |
| `Nombre estandarizado` | **1.699** / 1.826 | ❌ |

**186 filas comparten nombre de archivo** con otra: CSET_Georgetown (112), Amazon_Underworld (72), ESA_Space_Debris (2). Los 73 tiles PBF, por ejemplo, se reducen a solo 13 nombres distintos repartidos en `tiles/z/x/y/`.

**Consecuencia:** si `fuente` se construye con el nombre del archivo, 186 documentos colapsan entre sí. Y como §10.2.1 empareja el ground truth **por el campo `fuente`**, cada colisión es una consulta potencialmente mal puntuada en F1@3. `fuente` debe ser la ruta relativa completa tal como aparece en el índice.

### 1.3 El índice de ADL ya trae `DOC_ID` — la contradicción C1 queda resuelta

La hoja `Inventario de Archivos` (1.826 filas) trae las columnas `Fenómeno`, `Observatorio`, `Código Observatorio`, **`DOC_ID`**, `Nombre estandarizado`, `Carpeta` y `Tipo`. Los `DOC_ID` tienen la forma `F1-AIINDEX-001`, `F3-RESDAL-042`.

Esto confirma la lectura de la sesión 5 sobre la del documento técnico: **los `doc_id` vienen dados, no se inventan.** El índice es además la fuente de `fenomeno` (1/2/3) para la metadata de la Tabla 1: no hay que inferirlo de la ruta.

### 1.4 No existe un solo archivo HTML

La especificación (§2.1) dedica un apartado a limpiar marcado HTML. **El corpus no trae HTML.** Lo que hay es JSON de scraping —que es el HTML ya parseado por ADL— más PDF. Escribir un extractor de HTML sería trabajo muerto.

---

## 2. Inventario y reconciliación

### 2.1 Disco contra índice

| | Archivos |
|---|---:|
| Filas en el índice de ADL | 1.826 |
| Archivos de corpus en disco | 1.837 |
| **Coinciden** | **1.826** |
| **En el índice, ausentes en disco** | **0** |
| En disco, fuera del índice | 11 |

**El corpus está completo: no falta ningún archivo y ninguno está corrupto.** Esto cierra la contradicción **C5** de `CLAUDE.md`, que anotaba un conteo previo de solo ~1.386 archivos reconciliados. El problema era de conteo, no de datos.

Los 11 sobrantes no son corpus y deben excluirse explícitamente:

| Archivo | Qué es |
|---|---|
| `{ceeep,mapp,resdal}_catalogo.json`, `{ceobs,sipri}_full_catalogo.json` | manifiestos del scraper |
| `{ceeep,mapp,resdal}_registro.json`, `{ceobs,sipri}_full_registro.json` | registros de descarga |
| `FASE ORDENADA CODEFEST.xlsx` | archivo de trabajo del equipo |

**Política recomendada:** el índice de ADL es la única lista de entrada. Se itera sobre sus 1.826 filas, no sobre `data/**`. Así los sobrantes quedan fuera por construcción y no por una lista negra que haya que mantener.

Fuera del corpus quedan también los dos archivos de control en la raíz de `data/`: el propio índice y el PDF de consultas.

### 2.2 Composición por formato

| Formato | Archivos | % | Volumen de texto |
|---|---:|---:|---|
| JSON | 954 | 52,2 % | ~633 K palabras ⚠️ |
| PDF | 759 | 41,6 % | 36.828 páginas |
| PBF (vector tiles) | 73 | 4,0 % | atributos de mapa |
| CSV | 26 | 1,4 % | tablas |
| Imagen (jpg/avif) | 9 | 0,5 % | solo OCR |
| XLSX | 4 | 0,2 % | tablas |
| TXT | 1 | 0,1 % | 1.686 palabras |

⚠️ **Corregido el 2026-08-04 (ADR-001).** La cifra original de este sondeo era ~1,27 M
palabras: estaba inflada por doble conteo de `body_paragraphs` (632.558) y `body_text`
(592.659), que son el mismo texto. El volumen único es ~633 K; la extracción real produce
676.560 palabras al añadir `alerta_meta`/`fields` y las palabras clave.

### 2.3 Por fenómeno

| Fenómeno | Observatorios | Archivos | PDF | JSON |
|---|---:|---:|---:|---:|
| F1 — IA y Capacidades Estratégicas | 8 | 459 | 231 | 205 |
| F2 — Seguridad del Entorno Espacial | 5 | 479 | 237 | 230 |
| F3 — Dinámicas Territoriales | 8 | 888 | 291 | 519 |
| **Total** | **21** | **1.826** | **759** | **954** |

F3 concentra el 49 % del corpus y es también donde están los casos difíciles: los PDF escaneados y los tiles PBF.

---

## 3. Estrategias de extracción, por prioridad

### 3.1 JSON — 954 archivos, prioridad 1

Los esquemas **difieren por observatorio**. Hay tres familias, y confundirlas produce chunks vacíos o basura.

#### Familia A — artículo web (509 archivos, el caso rentable)

Claves: `url`, `title`, `date`, `authors`, `excerpt`, `body_paragraphs`, `body_text`, `tags`/`topics`, `pdf_links`, `images`.

| Observatorio | Archivos | Mediana palabras | Máx. |
|---|---:|---:|---:|
| Atlantic_Council | 186 | 2.032 | 14.914 |
| CSIS_Aerospace | 103 | 716 | 92.523 |
| SWF_Counterspace | 56 | 1.341 | 4.212 |
| INPE | 55 | 806 | 4.667 |
| SIPRI | 53 | 463 | 823 |
| CEOBS | 20 | 353 | 15.803 |
| ESA_Space_Debris | 16 | 2.050 | 9.814 |

**Estrategia:** concatenar `title` + `body_paragraphs` en orden. `body_text` suele ser el mismo contenido aplanado — usar uno **u** otro, nunca ambos, o cada documento se duplica y se sesga la puntuación por documento. `url`, `date`, `authors`, `tags` van a metadata, no al cuerpo (§2.1). Los `body_paragraphs` ya vienen segmentados por párrafo: son una frontera de chunking gratuita y de buena calidad.

Ojo con CSIS: mediana 716 palabras pero máximo 92.523. La cola larga obliga a que el chunker no asuma documentos cortos.

#### Familia B — metadata bibliográfica sin cuerpo (95 archivos)

- **CEEEP** (80): `url`, `title`, `date`, `authors`, `abstract`, `keywords`, `doi`, `pdf_url`. Mediana 168 palabras: es el *abstract*, no el artículo.
- **CENIA** (15): `sections`, `lists`, `links`, `images` y una bandera `contenido_limitado`. Mediana 145 palabras.

**Estrategia:** indexar `title` + `abstract` + `keywords` como documento propio. Es poco texto pero denso y muy indicativo del tema; funciona bien para recuperación. En CENIA hay que aplanar `sections` con cuidado y respetar `contenido_limitado` como señal de baja densidad.

#### Familia C — manifiestos, **sin texto útil** (7 archivos)

`Amazon_Underworld` (listado de 262 tiles), `DAIO` (35 estudios), `MAPP_OEA` (78), `RESDAL` (95), `RutaN_GEIAL` (2) y `Defensa21_LatAm` (2 archivos que son **listas JSON vacías**). Todos rinden **0 palabras**: son inventarios de descarga con `filename`, `status`, `size_bytes`, `scraped_at`.

**Estrategia:** no son documentos de contenido. Dos opciones: excluirlos, o extraer solo los campos descriptivos (`title`, `country`, `year`) como texto mínimo. Excluirlos es más limpio, pero **cada `doc_id` excluido es un documento que jamás podrá recuperarse**; si el ground truth lo incluye, es F1@3 perdido de forma irrecuperable. Recomendación: generar un chunk pobre pero no vacío a partir de los títulos, y anotar la decisión.

En total hay **20 JSON con menos de 20 palabras** que caerán en este mismo dilema.

#### Alertas_Tempranas (363 archivos) — caso aparte, el grupo más numeroso

Esquema propio: `url`, `title`, `fields`, `body_paragraphs`, `pdf_links`, `doc_links`, `alerta_meta`. **Mediana de solo 90 palabras** (máx. 775). Son fichas de alerta, no artículos.

Es el 38 % de los JSON pero apenas ~40.000 palabras. **Estrategia:** el objeto `fields` y `alerta_meta` son estructurados (municipio, fecha, tipo de alerta) y probablemente valen más como texto indexable que el cuerpo. Un documento de 90 palabras produce **un solo chunk** — y aquí conecta directo con la métrica: §9.2.1 permite rellenar hasta 250 palabras concatenando vecinos, y con 90 palabras se desperdician 160. Vale la pena evaluar si conviene concatenar `fields` al cuerpo.

### 3.2 PDF — 759 archivos, 36.828 páginas

Se abrieron **los 759**: **0 corruptos**. Densidad típica 280–390 palabras/página, coherente con informes institucionales.

**48 PDF no tienen ni una palabra extraíble en sus primeras 5 páginas: son escaneados.**

| Observatorio | Escaneados | Páginas |
|---|---:|---:|
| Alertas_Tempranas | 45 | ~540 |
| CSIS_Aerospace | 2 | |
| CSET_Georgetown | 1 | |
| **Total** | **48** | **582** |

**Estrategia:** extracción por página preservando orden de lectura, y detección automática de escaneo con un umbral de palabras/página (p. ej. < 40). Los 48 detectados van a una cola de OCR.

Sobre si vale la pena el OCR: 582 páginas es un volumen manejable, pero `pytesseract` exige instalar el binario de Tesseract y 45 de los 48 son de un mismo observatorio de F3. **Es una decisión de coste/beneficio con número concreto: 48 documentos de 1.826, un 2,6 %.** Si se omite el OCR, esos 48 `doc_id` quedan sin recuperar; conviene al menos indexar su título y su metadata del índice para que no sean invisibles.

⚠️ **Precisado el 2026-08-09 (ADR-003).** Los 48 documentos sin ninguna palabra extraíble
siguen siendo 48 — ese número no cambió. Lo que sí cambió es que la clasificación de
"escaneado" pasó de ser un promedio sobre las primeras 5 páginas del documento a una decisión
**por página, sobre las 36.828 páginas del corpus completo**: un documento largo con una sola
página de portada casi en blanco también queda marcado, así que la cuenta agregada de
documentos "con alguna página de baja densidad" es hoy 458, no 66 — son cosas distintas, no una
contradicción. Ninguna página con texto nativo se descarta por la clasificación del documento.

Tamaños extremos a considerar: ILIA_Latam tiene PDFs de ~189 páginas y 8,5 MB de mediana.

### 3.3 PBF — 73 archivos: son **vector tiles**, no OpenStreetMap

La duda que quedaba abierta en `CONSIDERACIONES.md` está resuelta por inspección binaria. Los primeros bytes son `1a d4 a0 08 0a 10 61 75 5f 63 6f 6d`: campo 3 (*layers*) del protobuf de Mapbox Vector Tile, seguido del nombre de capa `au_com`. **No hay cabecera `OSMHeader` ni compresión gzip.**

- Ruta: `F3_Dinamicas_Territoriales/Amazon_Underworld/tiles/{z}/{x}/{y}/`
- Tamaños: 735 B – 1,05 MB (mediana 228 KB)
- 13 nombres distintos para 73 archivos

→ La librería correcta es **`mapbox-vector-tile`**, no `osmium`. Se puede cerrar ese punto en `CONSIDERACIONES.md`.

**Estrategia:** recorrer capas y *features*, volcar los atributos como pares `atributo: valor`. La §2.1 advierte que el mismo elemento se repite en varios niveles de zoom: **deduplicar por identidad de feature antes de indexar**, o se inflará el índice con copias y se distorsionará la agregación a documento.

Realismo: 73 tiles con atributos de mapa difícilmente responden consultas en lenguaje natural sobre dinámicas territoriales. Es el 4 % del corpus con, probablemente, el peor retorno por hora invertida. **Sugerencia: dejarlo para el final**, después de que JSON y PDF estén cerrados.

### 3.4 CSV / XLSX — 30 archivos

Casi todos de AI_Index_Stanford (17 CSV + 4 XLSX). Los CSV de *clinical trials* tienen 27 columnas (`Rank`, `NCT Number`, `Title`, `Acronym`, …); los XLSX son estrechos, de 2–3 columnas (`pmid`/`title`/`journal`, `Author`/`Author ID`).

**Estrategia:** conforme a §2.1, cada fila es una unidad de fragmentación con formato `columna: valor`, omitiendo celdas vacías. Cuidado con dos cosas: los XLSX de 2 columnas tipo `Author | Author ID` **no tienen contenido semántico recuperable** —son tablas de identificadores— y generarían miles de chunks basura que compiten con contenido real en el índice. Conviene filtrar por columnas textuales antes de decidir si un tabular entra.

  ⚠️ **Corregido el 2026-08-09 (ADR-005).** La afirmación de que estos XLSX "no tienen
  contenido semántico recuperable" era una generalización excesiva: solo `Author ID`,
  `Conference ID` y (en el XLSX de lit-covid) `pmid` son identificadores sin valor de búsqueda.
  `Author`, `Conference Name`, `title`, `journal`, `Fields` y `Status` sí lo tienen y se
  conservan. La estrategia final excluye solo esas tres columnas por whitelist explícita, no el
  archivo ni la fila entera.

### 3.5 Imágenes (9) y TXT (1)

- **8 JPG + 1 AVIF**, todas de SWF_Counterspace. ~~Son fotos de misiones espaciales, no infografías: el OCR daría ruido. Recomendación: `texto=""` y solo metadata.~~ Nota técnica: **AVIF no lo abre Pillow sin `pillow-avif-plugin`.**

  ⚠️ **Corregido el 2026-08-04.** Esta conclusión era falsa y se tomó sin abrir los archivos.
  Al inspeccionarlos, **4 de las 9 llevan texto**, y dos de ellas son el contenido más denso
  del subconjunto:

  | `doc_id` | Contenido real | Texto |
  |---|---|---|
  | F2-SWF-076 | Tabla 5-1: pruebas ASAT, 17 filas × 9 columnas (fecha, país, interceptor, altitud, basura rastreada) | mucho |
  | F2-SWF-077 | Matriz de evaluación 2026: 13 países × 7 capacidades contraespaciales + leyenda | mucho |
  | F2-SWF-089 | Gráfico de barras "ASAT Tests by Country (2026)", 4 países con cifras | poco, denso |
  | F2-SWF-084 | Portada: *Global Counterspace Capabilities — An Open Source Assessment, 04/2026* | título |
  | F2-SWF-065/066/067/068/071 | Retratos y fotos de archivo de la NASA | ninguno |

  Son contenido de alto valor temático para el fenómeno 2. Con 4 archivos, transcribir a mano
  es más rápido y más fiable que afinar un motor de OCR. Ver
  `docs/guias/como-escribir-un-parser.md`.
- **`SWF_full-text.txt`**, 1.686 palabras. Caso trivial.

---

## 4. Riesgos abiertos

| # | Riesgo | Impacto | Mitigación propuesta |
|---|---|---|---|
| R1 | Consultas 100 % en español, corpus 45 % en inglés | Alto — afecta las dos métricas | Priorizar alineación cross-lingual al elegir encoder; probar es→en explícitamente |
| R2 | `fuente` mal construida por nombres duplicados | Alto — rompe el emparejamiento de F1@3 | Usar siempre la ruta relativa del índice |
| R3 | 48 PDF escaneados sin OCR | Medio — 2,6 % de documentos irrecuperables | Decidir OCR con número en mano; indexar al menos títulos |
| R4 | 7 manifiestos + 20 JSON casi vacíos | Medio | Chunk mínimo desde títulos en vez de excluir |
| R5 | Duplicación `body_text` + `body_paragraphs` | Medio — sesga la agregación a documento | Elegir uno de los dos, nunca ambos |
| R6 | Tiles PBF repetidos entre niveles de zoom | Medio — infla el índice | Deduplicar por feature |
| R7 | XLSX de identificadores sin semántica | Bajo — ruido en el índice | Filtrar por columnas textuales |
| R8 | Documentos muy largos (92 K palabras, 189 páginas) | Bajo | El chunker no puede asumir documentos cortos |

---

## 5. Orden de trabajo sugerido

1. **Cargador del índice de ADL** — itera las 1.826 filas y entrega `doc_id`, `fuente`, `fenomeno`, `formato`. Es la espina dorsal: todo lo demás cuelga de aquí.
2. **JSON familia A** (509 archivos, ~1,2 M palabras) — el mejor retorno por hora del corpus.
3. **PDF con texto** (711 archivos, ~36 K páginas) — el otro gran bloque.
4. **Alertas_Tempranas** (363 JSON) — esquema propio, decidir qué hacer con `fields`.
5. **JSON familias B y C**, CSV/XLSX, TXT — cola corta.
6. **OCR** y **PBF** — solo si Fase 1 está cerrada y sobra tiempo.

Con los pasos 1–3 se cubre el **66 % de los documentos** y la práctica totalidad del texto recuperable. Coherente con `CLAUDE.md` §6: una entrega válida completa primero, optimización después.

---

## 6. Qué actualizar en otros documentos

- **`CLAUDE.md` C5** (inventario sin reconciliar): resuelto — 1.826/1.826, cero faltantes.
- **`CLAUDE.md` C4** (distribución de formatos): confirmada, y añadir que **no hay HTML**.
- **`CONSIDERACIONES.md` §6** (PBF ambiguo): resuelto — vector tiles, `mapbox-vector-tile`.
- **`CONSIDERACIONES.md` §7** (OCR): ahora tiene número — 48 PDF, 582 páginas.
- **`docs/decisions/`**: los puntos R1, R3, R4 y R5 merecen ADR propio antes de indexar.
