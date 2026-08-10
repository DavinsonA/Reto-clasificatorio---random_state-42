# ADR-005: XLSX — excluir solo identificadores tecnicos, nada mas

- **Fecha**: 2026-08-09
- **Estado**: aceptada
- **Responsable**: Davinson Arteaga

## Contexto

4 XLSX en el corpus, todos de AI_Index_Stanford: `F1-AIINDEX-042` (lit-covid, ~8.866 filas de
`pmid`/`title`/`journal`), `F1-AIINDEX-043` (`Author`/`Author ID`, lifecycle de autores),
`F1-AIINDEX-044` (`Conference Name`/`Conference ID`), `F1-AIINDEX-045` (`Fields`/`Status`).

`pmid`, `Author ID` y `Conference ID` son identificadores de PubMed/Microsoft Academic Graph:
numeros sin valor semantico propio. Indexarlos como `pmid: 32634823` en el texto es ruido puro
— ningun encoder va a acercar una consulta en lenguaje natural a un numero de PubMed suelto.

## Opciones consideradas

| Opción | A favor | En contra |
|---|---|---|
| No filtrar nada (version inicial) | simple | `pmid`, `Author ID`, `Conference ID` son ruido en ~8.900 bloques |
| Heuristica amplia (`"id" in columna.lower()`) | atrapa cualquier variante futura | excluiria columnas legitimas por accidente (`Conference Name` no, pero cualquier futura columna con "id" en el nombre si, sin control) |
| **Whitelist explicita de 3 columnas** ✅ | exactamente lo que se decidio excluir, nada mas; nueva columna con "id" en el nombre no se excluye por accidente | hay que mantener la lista si aparece un XLSX nuevo con otro identificador tecnico — costo bajo, son 4 archivos |

## Decisión

`EXCLUDED_COLUMNS = {"pmid", "author id", "conference id"}`, comparado sobre el nombre de
columna normalizado con `clean(...).casefold()` (case-insensitive, tolera variaciones de
mayusculas/espacios del origen). La fila sigue siendo una unidad: se excluyen las columnas
tecnicas de esa fila, no la fila entera.

`title`, `journal`, `Author`, `Conference Name`, `Fields`, `Status` — todo lo que no esta en la
whitelist — se conserva exactamente igual que antes.

Trazabilidad: `extra["columnas_excluidas"]` por hoja, solo los **nombres** de columna excluidos
(`{"lit_covid": ["pmid"]}`), nunca los ~8.866 valores de `pmid` que se excluyeron — guardar eso
en metadata solo trasladaria el ruido de los bloques a `extra`.

## Consecuencias

**Qué se gana.** Verificado sobre los 4 archivos reales: 4/4 se siguen procesando, 0 bloques
vacios, `pmid:`/`Author ID:`/`Conference ID:` no aparecen en ningun bloque, `title:`/`journal:`
/`Author:`/`Conference Name:`/`Fields:`/`Status:` si aparecen. `F1-AIINDEX-042` produce 8.865
bloques (uno por fila de literatura), `F1-AIINDEX-043` 2, `F1-AIINDEX-044` 27, `F1-AIINDEX-045`
6.

**Qué se deja pendiente a proposito.** No se resuelve aqui el volumen de `F1-AIINDEX-042`
(8.865 bloques de un solo `doc_id`, potencial sesgo en agregacion a documento si muchos de esos
bloques rankean bien): es una decision de chunking/agregacion, fuera del alcance de extraccion.
Tampoco se deduplican filas ni se resumen datasets — extraccion es lossless por diseño.

**Qué habria que revisar si esto resulta equivocado.** Si aparece un XLSX nuevo con otro
identificador tecnico obvio (p. ej. `DOI` sin valor de busqueda), se añade a
`EXCLUDED_COLUMNS` con el mismo criterio: whitelist explicita, nunca una regla amplia por
substring.
