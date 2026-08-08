# ADR-001: Catálogo de ADL como entrada del pipeline y parser JSON

- **Fecha**: 2026-08-04
- **Estado**: aceptada
- **Responsable**: Davinson Arteaga

## Contexto

Arranca la Fase 1 y hay que escribir el primer parser. El sondeo (`docs/sondeo-corpus.md`)
dejó el terreno medido: 1.826 documentos, 954 JSON (52 %), 759 PDF (42 %), y el resto en
formatos de cola larga. Restricciones aplicables: metadata obligatoria de la Tabla 1,
`fuente` como clave de emparejamiento del ground truth (C1), `formato` como extensión real
(C2), y la regla de que un error de extracción se loggea y se salta (CLAUDE.md §7).

Dos decisiones había que tomar antes de escribir una línea: **de dónde sale la lista de
documentos** y **qué formato se ataca primero**.

## Opciones consideradas

### Lista de entrada

| Opción | A favor | En contra |
|---|---|---|
| Recorrer `data/**` | trivial de escribir | arrastra 11 manifiestos del scraper, `.DS_Store` y el índice mismo; obliga a mantener una lista negra; `doc_id` y `fenomeno` habría que inferirlos de la ruta |
| **Iterar el índice de ADL** ✅ | los sobrantes quedan fuera *por construcción*; `doc_id`, `fuente`, `fenomeno` y observatorio vienen dados; el orden es reproducible | depende de `openpyxl` (ya en `dev`) |

### Primer parser

| Opción | Archivos | Texto | Dificultad | Riesgo |
|---|---:|---|---|---|
| **JSON** ✅ | 954 (52 %) | ~633 K palabras | baja: ya viene parseado por ADL, sin marcado ni layout | esquemas distintos por observatorio |
| PDF | 759 (42 %) | 36.828 páginas | media: orden de lectura, columnas, 48 escaneados | OCR, PDFs de 189 páginas |
| PBF / CSV / imágenes | 113 (6 %) | marginal | alta o nula rentabilidad | — |

## Decisión

**La lista de entrada son las 1.826 filas de `Indice_Datos_Codefest.xlsx`, y el primer
parser es el de JSON.** Verificado: el índice empalma 1.826/1.826 con el disco, cero
faltantes.

Decisiones de diseño que van con ello:

1. **Contrato `RawDoc` con bloques, no con texto plano.** La extracción entrega unidades
   naturales del documento (párrafo, sección, fila) en orden de lectura. `body_paragraphs`
   ya viene segmentado por ADL: son fronteras de chunking gratuitas y mejores que cualquier
   corte por tamaño. El chunker decide si las respeta, las une o las parte; la extracción no
   se lo impone.
2. **Enrutado por forma del contenido, no por observatorio.** El parser mira qué claves trae
   el objeto (`body_paragraphs` → artículo, `abstract` → ficha, `sections` → CENIA, lista →
   manifiesto). Una lista de observatorios sería una tabla que mantener cada vez que ADL
   cambie un scraper.
3. **`body_paragraphs` o `body_text`, nunca ambos** (riesgo R5). 485 archivos traen los dos
   con el mismo contenido; sumarlos duplicaría cada documento y sesgaría la agregación a
   documento de §4.5.
4. **Ningún documento sin bloques** (riesgo R4). Los 7 manifiestos y los ~20 JSON casi vacíos
   reciben un bloque mínimo con título, observatorio y campos descriptivos. Un `doc_id` sin
   chunk es un documento irrecuperable para siempre: F1@3 perdido sin remedio.
5. **`alerta_meta` y `fields` se indexan como texto.** En Alertas_Tempranas (363 archivos,
   mediana 174 palabras) el municipio, el tipo de alerta y el tema clave son más informativos
   que el cuerpo, y las consultas van en español sobre esos mismos términos.
6. **Normalización conservadora.** NFC, borrado de invisibles de ancho cero y colapso de
   blancos. No se baja a minúsculas ni se quitan tildes: destruiría señal que los encoders
   multilingües sí aprovechan.

Decidido **sin medición de recuperación** — todavía no hay ground truth ni proxy. Lo que sí
está medido es el volumen (§ siguiente). Deuda anotada: los puntos 4 y 5 deben pasar por
`docs/ablaciones.md` cuando exista el set de desarrollo.

## Consecuencias

**Qué se gana.** 954 documentos extraídos, 0 fallidos, 676.552 palabras en 10.475 bloques.
Mediana de 212 palabras por documento. El 52 % del corpus queda listo para chunking.

El paquete son 266 líneas en cuatro archivos: `core.py` (contratos + catálogo + limpieza),
`json_docs.py` (el parser), `__init__.py` (el dict `PARSERS` y el recorrido tolerante a
fallos) y `__main__.py` (CLI). Añadir PDF es una entrada en `PARSERS` y un módulo que
devuelva `RawDoc`.

**Corrección al sondeo.** El sondeo estimaba ~1,27 M palabras en los JSON. La cifra estaba
inflada por doble conteo: `body_paragraphs` suma 632.558 palabras y `body_text` otras
592.659, y son el mismo texto. El volumen único real es ~633 K.

**Qué queda pendiente de verificar.**

- Los títulos de Alertas_Tempranas son basura del scraper (`"Mapa"` en muchos casos); el
  cuerpo está bien, pero no conviene apoyarse en `title` para esa fuente.
- Algunos `body_paragraphs` de Atlantic_Council empiezan con la firma del autor (`"By ..."`).
  Ruido menor; medir si conviene filtrar antes de decidirlo.
- 13 documentos quedaron con contenido mínimo y 24 por debajo de 20 palabras. Son
  irrecuperables en la práctica salvo por coincidencia de título.

**Qué habría que revisar si esto resulta equivocado.** Si el chunker acaba necesitando el
texto aplanado en vez de los bloques, el cambio es en `RawDoc` — y `RawDoc` es contrato de
equipo: se avisa antes de tocarlo.
