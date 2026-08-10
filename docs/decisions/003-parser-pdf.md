# ADR-003: Parser de PDF — pagina como bloque, OCR por pagina y no destructivo

- **Fecha**: 2026-08-08 (revisada 2026-08-09)
- **Estado**: aceptada
- **Responsable**: Juan Villegas (version inicial); revision de la seccion "OCR
  por pagina" por Davinson Arteaga

> Nota de numeracion: el titulo original de este documento decia "ADR-002" por
> un error de copiado; el archivo siempre se llamo `003-parser-pdf.md` y esa es
> la numeracion correcta (002 es `002-parser-imagenes-y-ocr.md`). Corregido en
> esta revision.
>
> **Que cambio en la revision del 2026-08-09.** La version inicial clasificaba
> el documento **completo** como escaneado (promedio de palabras en sus
> primeras 5 paginas) y, si OCR estaba apagado, sustituia `pages` por `[]`
> **para el documento entero**. Verificado sobre el corpus: eso es destructivo.
> F2-CSIS-155 tiene 4 palabras nativas en sus primeras 5 paginas (0,8
> palabras/pagina, clasificado "escaneado") pero una de sus 9 paginas si tiene
> texto real — la version anterior lo habria descartado igual. La decision de
> nativo-vs-OCR pasa a ser **por pagina**; la seccion "OCR por pagina" de este
> documento reemplaza la logica original. El resto del ADR (granularidad de
> bloque, negrita sintetica) sigue vigente sin cambios.

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

### PDF escaneados (revisado: la decision es por pagina)

La version inicial clasificaba **el documento entero** por el promedio de palabras en sus
primeras 5 paginas. Verificado sobre el corpus: eso descarta texto real. F2-CSIS-155 tiene 4
palabras nativas en sus primeras 5 paginas (0,8 palabras/pagina, "escaneado" con ese criterio)
pero 9 paginas en total, y una de las que quedan fuera de la muestra si tiene texto extraible.
Clasificar por documento y no por pagina lo habria borrado igual.

| Opción | A favor | En contra |
|---|---|---|
| Clasificar el documento completo (version inicial) | simple, un solo promedio | **destructivo**: un documento con paginas mixtas (algunas densas, algunas casi vacias) pierde las densas si la muestra inicial sale baja |
| **Decision por pagina, OCR opcional y no destructivo** ✅ | cada pagina conserva su nativo salvo que el OCR la mejore de verdad; nunca se borra texto sin reemplazo | una pagina mas de logica por iteracion; coste ya medido (ver Consecuencias) |
| OCR con Tesseract activado por defecto | recupera texto real de las paginas de baja densidad | exige instalar el binario en cada maquina del equipo hoy mismo, sin medir si compensa |
| Transcripcion manual (como en el ADR de imagenes) | maxima fidelidad | miles de paginas, no 4 imagenes: no escala (Parte 3, guia conceptual) |

Mismo criterio que la guia de parsers usa para las imagenes (§7.4): dejar el camino OCR
**escrito, opcional y apagado por defecto** (`PDF_OCR=1`), pero la decision nativo-vs-OCR
ahora vive en `_decide_page`, no en una clasificacion previa del documento completo.

## Decisión

**Una pagina de PDF = un bloque**, extraida con PyMuPDF (`fitz`, ya en `dev`) usando
`get_text("text", sort=True)` para respetar el orden de lectura en documentos con columnas.

**Decision nativo-vs-OCR, por pagina (`MIN_WORDS_PAGE = 40`, mismo umbral que la version
inicial, ahora aplicado pagina a pagina en vez de a un promedio de 5 paginas):**

1. Si el texto nativo de la pagina tiene ≥ 40 palabras, se usa el nativo y no se llama a OCR.
2. Si tiene menos y OCR esta apagado, se conserva el nativo tal cual, por corto que sea.
3. Si tiene menos y OCR esta encendido, se intenta OCR; el resultado solo reemplaza al nativo
   si es no vacio y tiene mas palabras. Si el OCR falla, sale vacio o es peor, se conserva el
   nativo. Ninguna combinacion borra un texto nativo no vacio sin un reemplazo mejor.

`extra["escaneado"]` deja de ser un promedio sobre 5 paginas: ahora es "el documento tiene al
menos una pagina por debajo del umbral" (`extra["paginas_baja_densidad"]`), calculado sobre
**todas** las paginas. Es agregado y no destructivo — no decide que bloques existen.

**OCR:** implementado como `_decide_page` + `_ocr_page` en `pdf_docs.py`, que reutiliza
`images.ocr()` (ahora con `scale`/`config` explicitos) rasterizando a 300 ppp con
`scale=1` — la pagina ya sale a la resolucion que Tesseract espera, reescalar otra vez seria
redundante. Config `--psm 3` (segmentacion automatica), no `--psm 6` como en imagenes: los PDF
del corpus mezclan columnas, tablas e infografias, y `--psm 6` (bloque uniforme) es peor ahi.
Activado con `PDF_OCR=1`. Import perezoso: el modulo importa sin que `pytesseract`/Pillow
esten instalados. **Ahora si estan en `dev`** (`uv add --group dev pillow pytesseract`, junto
con `pytest`), pero el **binario** de Tesseract sigue siendo instalacion aparte del sistema
operativo — ver el README para las instrucciones.

**Hallazgo no previsto — "negrita sintetica":** 230 de 759 documentos (30 %) traian palabras
duplicadas sin espacio por como el PDF simula negrita (dibuja el texto dos veces
superpuesto). Verificado a mano sobre 25 muestras al azar: siempre era el glitch (encabezados
de pagina, "TOTAL", nombres de observatorio repetidos en cada pagina), nunca una palabra que
se repite de verdad. Se colapsa con una regex (`\b(\w{2,})\1\b` → `\1`) aplicada por pagina
antes de la limpieza general. Bajó de 2.480 bloques afectados a 12 (residuo: casos con
guiones internos como `System-to-Model`, que rompen el limite de palabra de la regex — no se
persiguio mas alla por rendimiento decreciente).

Decidido **con medicion sobre el corpus completo**, no sobre una muestra: los 759 PDF se
corrieron de punta a punta, con OCR apagado, antes de aceptar esta version (§ Consecuencias).

## Consecuencias

**Qué se gana.** 759 documentos extraidos, 0 fallidos, **12.834.862 palabras** en bloques
(759 PDF, ~7 minutos con OCR apagado), coherente con la densidad de 280–390 palabras/pagina
del sondeo. El 42 % del corpus queda listo para chunking, sumado al 52 % de JSON del ADR-001:
94 % del corpus cubierto. F2-CSIS-155 (el caso que motivo la revision) conserva sus 4 palabras
nativas de la pagina 2 en vez de perderlas por la clasificacion global — verificado, no
supuesto.

**Qué cambio con la revision por pagina.** `extra["escaneado"]` ahora marca **458** documentos
(antes 66), porque se calcula sobre todas las paginas del documento y no sobre un promedio de
las primeras 5: cualquier PDF largo con una sola pagina de portada o divisor casi en blanco
entra en la cuenta. Es una medida mas honesta, no una regresion — esas paginas ya conservaban
su texto nativo antes y despues del cambio; lo que cambio es que ahora quedan **registradas**
en vez de promediadas y ocultas. Los **48** documentos con `contenido_minimo` (cero texto
nativo en absoluto) no cambiaron: son los mismos 48 del sondeo original.

**OCR probado con Tesseract real (2026-08-09).** Se instalo Tesseract 5.5.3 (winget,
`tesseract-ocr.tesseract`) con paquetes `spa`/`eng`/`por` (via `TESSDATA_PREFIX` propio del
usuario, sin tocar `Program Files` — no requiere admin) y se corrieron las 3 muestras con
`PDF_OCR=1`:

| `doc_id` | Antes (sin OCR) | Con OCR real | Tiempo |
|---|---|---|---|
| F3-ALERTAS-364 (español, 11 pag.) | 4 palabras (bloque minimo) | **5.974 palabras**, 11 bloques | 23,5 s |
| F1-CSET-096 (infografia, 1 pag.) | 4 palabras (bloque minimo) | **318 palabras**, 1 bloque | 7,2 s |
| F2-CSIS-155 (inglés, 9 pag.) | 4 palabras (1 pag. nativa) | **5.322 palabras**, 9 bloques | 16,8 s |

Las tres mejoraron de forma drastica. La calidad no es perfecta — el escaneado de ALERTAS-364
(un oficio institucional con membrete/logo) trae ruido tipico de OCR sobre membretes
("Wy Defensoria o del Pueblo 20 Lit, CUIS ot om sia pá") mezclado con texto correcto y legible
("Bogotá D.C., 18 de diciembre de 2018", "MINISTERIO DEL INTERIOR", "Fecha de Radicación") —
pero es sustancialmente mejor que un bloque minimo de 4 palabras, que era la alternativa.
F1-CSET-096 y F2-CSIS-155 (ingles, texto de informe sin membretes complejos) salieron limpios.

**Costo medido para el corpus completo (2026-08-09).** Conteo real, sin OCR (391,4 s), sobre
los 759 PDF / 36.828 paginas:

| Metrica | Valor |
|---|---:|
| Paginas totales del corpus | 36.828 |
| Paginas bajo el umbral (candidatas a OCR) | **3.381** (9,2 %) |
| Documentos con alguna pagina baja | 458 |
| Documentos con `contenido_minimo` (0 nativo en absoluto) | 48 |

A ~2,3 s/pagina (medido sobre las 21 paginas de las 3 muestras): **~130 minutos** de OCR puro
para las 3.381 paginas candidatas, ~155 minutos con 20 % de margen. Es mucho mas que el piso de
~22 minutos que se habia estimado solo con los 48 documentos totalmente escaneados — la mayoria
de las 3.381 paginas vienen de los otros 410 documentos que tienen *alguna* pagina debil
(portadas, divisores, paginas con una figura grande) dentro de un documento por lo demas nativo,
igual que F2-CSIS-155.

**Riesgo residual actualizado:** ya no es "no se pudo probar" — se probo y funciona, y el costo
ya no es una incognita. Lo que sigue pendiente es decidir, con el equipo, si ~2-2,5 horas de
computo (calidad imperfecta en
escaneados con membretes institucionales) compensa frente a otras prioridades de la Fase 2
antes del congelamiento del 7 de agosto.

## OCR ejecutado sobre el corpus completo (2026-08-10) — resultado final

Corrido con el comando de arriba, sin supervision, entorno persistido el dia anterior sin
pasos extra. Resultado, comparado con la corrida sin OCR:

| Metrica | Sin OCR | Con OCR |
|---|---:|---:|
| Documentos con `contenido_minimo` | **48** | **0** |
| Documentos con alguna pagina reconocida por OCR | 0 | 381 |
| Paginas reconocidas por OCR (de 3.381 candidatas) | 0 | 1.966 |
| Palabras totales en bloques de PDF | 12.834.862 | **13.302.268** (+467.406) |

**Los 48 documentos que antes no tenian ni una palabra recuperable ahora tienen contenido
real.** De las 3.381 paginas candidatas (bajo el umbral de 40 palabras), el OCR gano en 1.966
— el resto (1.415) probo OCR y perdio frente al nativo existente o salio vacio, y el
comportamiento no destructivo se sostuvo: se quedaron con lo que ya tenian, no con nada.

Verificado con una muestra de 5 documentos al azar entre los 53 que quedaron 100 %
reconocidos por OCR: la calidad es desigual pero util. Los oficios institucionales escaneados
de Alertas_Tempranas (mismo formato en varios de los 48, membrete + logo) traen ruido de OCR
sobre el membrete ("ps na Bogotá D.C.", "Oy oir Defensoría del Pueblo") pero el cuerpo
sustantivo sale legible (fechas, nombres, "Ministra del Interior", numero de radicado).
Documentos de texto corrido sin membrete complejo (CSIS, CSET) salieron limpios. Nadie
reviso los 759 documentos linea por linea — es una muestra, no una auditoria completa.

**Decisión: se acepta el resultado.** El costo (una corrida nocturna sin supervision) fue
bajo y la ganancia es real y verificada: 48 documentos que eran invisibles para la
recuperacion ahora tienen texto indexable. `data/interim/raw_pdf_ocr.jsonl` queda como el
volcado de referencia para la siguiente etapa (chunking), reemplazando cualquier volcado de
PDF sin OCR.

**Qué se pierde o queda pendiente.**

- **Resuelto (2026-08-10):** los 48 documentos sin texto nativo alguno ya no existen como tal
  — con `PDF_OCR=1` sobre el corpus completo, los 759 documentos quedaron con 0 en
  `contenido_minimo`. Ver "OCR ejecutado sobre el corpus completo" mas arriba.
- La calidad del OCR sobre escaneados con membrete institucional (Alertas_Tempranas) trae
  ruido en encabezados/logos; el cuerpo sustantivo es legible. Verificado sobre una muestra de
  5 de 53 documentos totalmente reconocidos, no sobre los 759 — no hay una auditoria de
  calidad completa.
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
particion fina es responsabilidad de la siguiente etapa. El OCR ya demostro calidad util sobre
las paginas de baja densidad (§ arriba): vale la pena revisar si conviene pasarlo tambien por
las 5 imagenes sin transcripcion manual del ADR-002, ahora que el costo de instalacion ya esta
pagado y el entorno queda listo.
