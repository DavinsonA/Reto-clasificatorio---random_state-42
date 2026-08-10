# ADR-004: CSV timeline — representacion lossless, sin narrativa sintetizada

- **Fecha**: 2026-08-09
- **Estado**: aceptada (reemplaza la narrativa sintetizada de la version inicial de `csv_docs.py`)
- **Responsable**: Davinson Arteaga

## Contexto

5 de los 26 CSV del corpus son series temporales de PubMed (AI/CV/ML/NLP/robotics,
`F1-AIINDEX-055/058/060/062/064`), dos columnas (`Year`, `Count`), 44 a 67 filas cada uno.

La version inicial del parser trataba estos 5 archivos distinto del resto: en vez de una fila
= un bloque (como el resto de CSV/XLSX del corpus), generaba **un solo bloque narrativo** por
archivo con valor inicial, valor final, pico y total acumulado, y guardaba la tabla cruda en
`extra` sin que fuera un chunk buscable.

Esa representación es lossy en dos sentidos: los años intermedios (la mayoria de las 44-67
filas) dejan de ser recuperables por busqueda semantica, y el total/pico son agregaciones que
el archivo original no afirma — son un calculo del parser, no un dato de la fuente.

## Opciones consideradas

| Opción | A favor | En contra |
|---|---|---|
| Narrativa sintetizada (version inicial) | un bloque legible tipo resumen | lossy: los años intermedios no son recuperables; inventa afirmaciones (total, pico) que el CSV no contiene |
| **Misma representacion que el resto de CSV: una fila = un bloque** ✅ | lossless: cada año queda recuperable por si mismo; cero logica especial que mantener aparte | una consulta sobre "la tendencia" no tiene un bloque unico que la resuma — eso es trabajo del chunker/agregacion, no de la extraccion |

## Decisión

**Los timeline usan la misma ruta de lectura que cualquier CSV: una fila = un bloque**, en el
mismo formato `columna: valor` (`Year: 2020 | Count: 6828`), **sin reordenar** — se conserva el
orden del archivo tal como viene (los 5 archivos reales vienen en orden descendente por año,
verificado, pero el parser no lo asume ni lo fuerza).

Se elimino el codigo que existia solo para la narrativa: `TOPIC_HINTS`, `_extract_timeline`,
`_parse_series`, `_narrativa`, `_fmt`, `_topic`, `_humanize`. `_is_timeline()` se conserva
porque sigue teniendo un consumidor: marcar metadata.

Metadata añadida cuando `_is_timeline(entry)` es verdadero:

- `extra["serie_temporal"] = True`
- `extra["num_puntos"]` = filas que produjeron un bloque (mismo valor que `num_filas`, con el
  nombre que pide la evidencia cuantitativa para el informe)
- `extra["orden_temporal"]` = `"descendente"` o `"ascendente"` **solo si se puede determinar
  sin ambigüedad** (todos los valores de la primera columna parsean como numero y la secuencia
  es monotona). Si algun valor no parsea o el orden no es monotono, el campo simplemente no se
  añade — no se afirma un orden que no se pudo verificar. Es deteccion, nunca reordenamiento:
  los `blocks` jamas se tocan.

## Consecuencias

**Qué se gana.** Verificado sobre los 5 archivos reales: 67+58+57+44+62 = 288 filas, todas
producen un bloque, ningun año desaparece, los 5 se detectan en orden descendente
(`orden_temporal: "descendente"` en los 5). Cero narrativa sintetica, cero `total = sum(...)`.
26/26 CSV del corpus se siguen procesando sin regresion.

**Qué se pierde.** La legibilidad de "en una frase, como crecio la investigacion en ML" ya no
la da la extraccion. Si el equipo quiere ese resumen, es trabajo de una etapa posterior
(agregacion a documento o un post-filtro sobre `extra["serie_temporal"]`), nunca sintetizado
dentro del parser — sintetizar ahi es exactamente lo que este ADR revierte.

**Qué habria que revisar si esto resulta equivocado.** Si el chunker mide que 50-60 bloques
casi identicos en forma (`Year: XXXX | Count: NNNN`) compiten mal entre si en el indice para
una consulta tipo "tendencia", la solucion es una estrategia de agregacion o fusion en el
chunker/recuperacion — no volver a sintetizar texto en la extraccion.
