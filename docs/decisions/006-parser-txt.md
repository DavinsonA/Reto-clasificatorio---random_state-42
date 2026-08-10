# ADR-006: Parser de TXT — scrape SWF detectado por contenido, boilerplate excluido

- **Fecha**: 2026-08-09
- **Estado**: aceptada
- **Responsable**: Davinson Arteaga

## Contexto

Un unico TXT en el corpus: `F2-SWF-113`, `SWF_full-text.txt` (~1.686 palabras). Es un scrape de
la pagina web de Secure World Foundation, no un documento de texto plano limpio: trae una
cabecera de scraping (`SOURCE:`, `SCRAPED:`), luego ~150 lineas de navegacion (menu, footer,
newsletter) antes de llegar al contenido real (Background, resumen del reporte 2026,
actualizaciones), y despues del contenido mas boilerplate (publicaciones relacionadas con
parrafos de relleno "Lorem ipsum" literales, newsletter, footer).

Complicacion adicional: dentro del nucleo util, el scraper puso cada hipervinculo del HTML
original en su propia linea — el parrafo que menciona a los dos editores del reporte llega
partido en 5 lineas porque sus nombres son enlaces.

## Opciones consideradas

| Opción | A favor | En contra |
|---|---|---|
| Indexar el archivo completo tal cual | trivial | ~1.086 de las ~1.686 palabras son navegacion/footer/relleno; serian ruido compitiendo con contenido real en el indice |
| Cortar por posicion fija (ej. "las primeras N lineas") | simple | fragil: cualquier cambio en el largo del menu de navegacion rompe el corte; no generaliza a otro TXT |
| **Detectar el patron de scrape por señales de contenido + fallback generico** ✅ | robusto a cambios de longitud del boilerplate; un TXT que no matchee el patron cae a un fallback razonable en vez de fallar | dos rutas de codigo en vez de una |

## Decisión

`_is_swf_scrape()` detecta el patron por señales estables (`SOURCE:`/`SCRAPED:` en las dos
primeras lineas + presencia de los anchors `Background` y `Major Updates in 2026:`), **no** por
`doc_id` — si algun dia hay un segundo TXT con el mismo formato de scrape, se detecta igual.
Cualquier TXT que no matchee cae al fallback generico (parrafos por linea en blanco, igual que
el resto del pipeline).

**Nucleo util:** las lineas entre el anchor `Background` y la linea de licencia (`Global
Counterspace Capabilities © ...`), que se excluye del cuerpo — es licencia, no contenido.

**Reconstruccion de lineas partidas:** `_join_wrapped()` une lineas consecutivas mientras la
anterior no termine en puntuacion terminal (`. ! ? :`). Es una sola funcion generica, no
logica distinta por seccion: donde el scraper ya entrego una linea = una oracion completa
(los parrafos de Background, cada actualizacion de "Major Updates"), la funcion no cambia nada;
donde partio una oracion a la mitad (los nombres de los editores como hipervinculos), la
reconstruye. Ningun bloque queda cortado a mitad de oracion (spec §3.3).

**Metadata:** `source_url` y `scraped_at` desde la cabecera; `título` = la primera linea con
forma "AAAA ... Report" antes del nucleo (`2026 Global Counterspace Capabilities Report`), con
fallback al nombre de archivo humanizado si no aparece nada con esa forma.

## Consecuencias

**Qué se gana.** Verificado contra el archivo real: 14 bloques desde el nucleo (2 parrafos de
Background, el parrafo reconstruido de "The 2026 Report" con los dos editores intactos, 11
actualizaciones de "Major Updates" cada una como bloque propio). `Lorem ipsum`, el menu de
navegacion, el footer y el texto de la licencia no aparecen en ningun bloque. Titulo correcto
extraido del contenido: `2026 Global Counterspace Capabilities Report`.

**Qué se deja pendiente.** El fallback generico solo se ejercito con archivos de prueba
sinteticos (no hay un segundo TXT real en el corpus para validarlo). Si aparece un TXT nuevo
con un formato de scrape distinto, probablemente necesite su propia deteccion especifica en vez
de depender del fallback generico — igual que se hizo aqui para SWF.

**Qué habria que revisar si esto resulta equivocado.** Si el chunker necesita mas contexto por
bloque en las actualizaciones de "Major Updates" (son oraciones sueltas, algunas sin mencionar
explicitamente "2026" o "counterspace"), la solucion es prefijar con el encabezado de seccion
en el parser — cambio de una linea, discutido y descartado en esta version porque cada bullet
ya es una oracion autocontenida (menciona paises, tecnologias y eventos especificos).
