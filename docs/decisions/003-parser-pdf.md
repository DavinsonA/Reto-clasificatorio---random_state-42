# ADR-002: Parser de PDF — pagina como bloque, escaneados sin OCR por defecto

- **Fecha**: 2026-08-08
- **Estado**: aceptada
- **Responsable**: Juan Villegas

## Contexto

759 de 1.826 documentos (42 %) son PDF, 36.828 paginas en total (`docs/sondeo-corpus.md`
§3.2). El sondeo ya habia detectado 48 PDF sin una sola palabra extraible en sus primeras 5
paginas: son fotos de texto, no texto. Restricciones aplicables: invariante de "nunca cero
bloques" (CLAUDE.md §10, guia de parsers), prohibicion de decoders/OCR generativo (CLAUDE.md
§2.1), `pymupdf` ya declarado en `[dependency-groups] dev` (nunca llega a `generador.py`).

Dos decisiones de diseño y un hallazgo no previsto en el sondeo:

1. Que cuenta como "bloque" en un PDF.
2. Que hacer con los PDF escaneados.
3. (Hallazgo en la verificacion, no en el sondeo) varios PDF renderizan negrita duplicando
   cada palabra sin espacio (`RESDALRESDAL`, `AIAI System-to-ModelSystem-to-Model`), un
   glitch de generacion de PDF, no un problema de extraccion.

## Opciones consideradas

### Granularidad del bloque

| Opción | A favor | En contra |
|---|---|---|
| **Una pagina por bloque** ✅ | unidad natural de lectura en un PDF (guia conceptual, Parte 0); preserva orden; documentos de 189 paginas (ILIA_Latam) no generan un solo bloque gigante | una pagina puede exceder o quedarse corta del presupuesto de 250 palabras del chunker; eso lo resuelve el chunker, no el parser |
| Todo el PDF como un bloque | trivial | pierde granularidad; el chunker no puede aprovechar los cortes naturales del documento |
| Por parrafo (via layout) | mas fino | mucho mas caro de implementar bien (columnas, notas al pie) para un beneficio no medido aun |

### PDF escaneados

| Opción | A favor | En contra | Licencia | Costo |
|---|---|---|---|---|
| **Deteccion + bloque minimo, OCR opcional desactivado** ✅ | desbloquea la Fase 1 hoy; cero dependencias nuevas de sistema; el `doc_id` sigue siendo recuperable por titulo/metadata | 48 documentos (582 paginas, 2,6 % del corpus) quedan con muy poco texto hasta que se decida el OCR | — | ninguno |
| OCR con Tesseract activado por defecto | recupera texto real de 582 paginas | exige instalar el binario en cada maquina del equipo hoy mismo, sin medir si compensa; no aplica al chunking/encoder todavia | Apache 2.0 (Tesseract) | medio-alto, sin medir retorno |
| Transcripcion manual (como en el ADR de imagenes) | maxima fidelidad | 582 paginas, no 4 imagenes: no escala (Parte 3, guia conceptual) | — | alto |

Mismo criterio que la guia de parsers usa para las imagenes (§7.4): dejar el camino OCR
**escrito, opcional y apagado por defecto** (`PDF_OCR=1`), y decidir con la ADR de imagenes
como precedente cuando haya tiempo de medirlo.

## Decisión

**Una pagina de PDF = un bloque**, extraida con PyMuPDF (`fitz`, ya en `dev`) usando
`get_text("text", sort=True)` para respetar el orden de lectura en documentos con columnas.

**Deteccion de escaneados:** promedio de palabras/pagina por debajo de 40 en las primeras 5
paginas (mismo criterio que el sondeo). Verificado sobre el corpus completo: detecta 66 PDF,
de los cuales 48 tienen **cero** palabras en la muestra — exactamente los 48 del sondeo. Los
18 adicionales tienen texto residual (etiquetas de ejes, encabezados sueltos en paginas casi
en blanco): mismo problema practico, umbral mas estricto que "cero literal".

**OCR:** implementado como rama opcional (`_ocr_pages`, Tesseract via `pytesseract`),
desactivada por defecto vía variable de entorno `PDF_OCR=1`. Import perezoso: el modulo
importa sin que `pytesseract`/Pillow esten instalados. **No se añaden a `dev` todavia** —
activar esto es una decision de equipo pendiente (582 paginas, instalar un binario de
sistema), no algo que se decide dentro de este parser.

**Hallazgo no previsto — "negrita sintetica":** 230 de 759 documentos (30 %) traian palabras
duplicadas sin espacio por como el PDF simula negrita (dibuja el texto dos veces
superpuesto). Verificado a mano sobre 25 muestras al azar: siempre era el glitch (encabezados
de pagina, "TOTAL", nombres de observatorio repetidos en cada pagina), nunca una palabra que
se repite de verdad. Se colapsa con una regex (`\b(\w{2,})\1\b` → `\1`) aplicada por pagina
antes de la limpieza general. Bajó de 2.480 bloques afectados a 12 (residuo: casos con
guiones internos como `System-to-Model`, que rompen el limite de palabra de la regex — no se
persiguio mas alla por rendimiento decreciente).

Decidido **con medicion sobre el corpus completo**, no sobre una muestra: los 759 PDF se
corrieron de punta a punta antes de aceptar esta version (§ Consecuencias).

## Consecuencias

**Qué se gana.** 759 documentos extraidos, 0 fallidos, ~12,6 M palabras en bloques (36.828
paginas), coherente con la densidad de 280–390 palabras/pagina del sondeo. El 42 % del corpus
queda listo para chunking, sumado al 52 % de JSON del ADR-001: 94 % del corpus cubierto.

**Qué se pierde o queda pendiente.**

- 66 documentos (3,6 % del corpus) quedan con un bloque minimo (titulo + observatorio) hasta
  que el equipo decida activar OCR. Son recuperables por titulo, no por contenido.
- El titulo de PDF sin metadata de titulo incrustada cae al nombre de archivo — igual que el
  parser de JSON e imagenes; no siempre es legible (ver ADR de imagenes sobre IDs de NASA).
- Quedan ~12 bloques con duplicacion residual por guiones internos: impacto marginal, no
  bloqueante.
- Algunos PDF de RESDAL muestran caracteres acentuados como `�` (fuente sin `ToUnicode` CMap
  legible): perdida de informacion en el archivo original, no reparable por software sin
  reconstruir la fuente. No se intento arreglar.

**Qué habria que revisar si esto resulta equivocado.** Si el chunker mide que las paginas son
demasiado grandes o demasiado desiguales en tamaño para el presupuesto de 250 palabras, el
ajuste va en el chunker, no aqui — el contrato de `RawDoc` es bloques en orden de lectura, la
particion fina es responsabilidad de la siguiente etapa. Si se activa `PDF_OCR=1` en el
futuro, falta añadir `pytesseract` y `pillow` a `dev` y documentar la instalacion de Tesseract
en el README, tal como advierte la guia de parsers §6.
