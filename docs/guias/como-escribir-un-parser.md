# Cómo escribir un parser en este pipeline

> Guía general para cualquier formato de archivo, con el **parser de imágenes** como caso de
> estudio. No hay código: hay preguntas, conceptos y criterios. El código lo escribes tú.

**Fecha:** 2026-08-04 · **Para:** cualquiera del equipo que vaya a añadir un formato

---

## Cómo usar esta guía

Léela una vez entera antes de tocar nada. Luego vuelve al **Paso 1** y ve bajando, contestando
por escrito las preguntas de cada paso. Si una pregunta no la puedes contestar, ese es
exactamente el trabajo que falta: no lo saltes escribiendo código "provisional", porque el
código provisional de un parser sobrevive hasta la entrega.

Al final hay un [checklist](#parte-4--checklist-para-cualquier-formato-nuevo) y una
[lista de recursos](#parte-5--recursos-para-aprender) por tema.

---

## Parte 0 — Qué es un parser aquí, exactamente

En este proyecto un parser es **una función con un contrato muy estrecho**:

> Recibe una fila del catálogo de ADL (un `CatalogEntry`: qué archivo es, dónde está, qué
> `doc_id` tiene, de qué fenómeno es) y devuelve un `RawDoc` (el texto de ese archivo,
> partido en bloques, más metadata opcional).

Eso es todo. No decide cómo se fragmenta, no calcula embeddings, no sabe que existe FAISS.

**Por qué el contrato es tan estrecho.** Porque el pipeline es una cadena y cada eslabón
tiene que poder cambiarse sin romper los demás. Si el parser de imágenes decidiera por su
cuenta el tamaño de los fragmentos, cambiar la estrategia de chunking obligaría a tocar todos
los parsers. Esto no es purismo: el 7 de agosto hay congelamiento y no habrá tiempo para
refactores en cascada.

**Los dos conceptos del contrato que hay que entender antes de seguir:**

**Bloque.** Una unidad natural de texto del documento, en orden de lectura. En un artículo
web es un párrafo; en un PDF, una página; en un CSV, una fila. No es "un fragmento del
índice" — es materia prima para que el chunker decida después. La regla mental: *un bloque
es el trozo más pequeño que sigue teniendo sentido por sí solo.*

**El invariante de "nunca cero bloques".** Ningún documento puede salir del parser sin al
menos un bloque de texto. La razón es de puntuación, no de estilo: la métrica de documentos
(F1@3) se calcula sobre los `doc_id` que devolvemos. Un documento que no produjo ningún
fragmento **jamás** podrá aparecer en una respuesta, ni por casualidad. Si el ground truth
lo consideraba relevante para alguna consulta, ese punto está perdido y no hay forma de
recuperarlo después. Por eso, cuando un archivo no tiene texto extraíble, el parser fabrica
un bloque mínimo con lo que sea que lo identifique (título, observatorio, nombre de archivo).

---

## Parte 1 — Los siete pasos

### Paso 1. Inventariar antes de escribir una sola línea

**Preguntas que tienes que contestar:**

- ¿Cuántos archivos de este formato hay exactamente, según el índice de ADL?
- ¿Cómo se reparten por fenómeno y por observatorio?
- ¿Cuánto pesan? ¿Hay outliers (uno de 300 KB y otro de 40 MB)?
- ¿Vienen todos del mismo sitio, o de fuentes distintas que probablemente tengan estructuras
  distintas?

**El concepto.** Un inventario no es burocracia: es lo que te dice **cuánto vale** el trabajo
que vas a hacer. Escribir un parser cuesta lo mismo para 9 archivos que para 900; el retorno
no. Y saber que todos vienen del mismo observatorio te ahorra escribir código defensivo para
variaciones que no existen.

**Por qué aquí en particular.** El corpus son 1.826 archivos de 21 observatorios distintos
descargados por scrapers distintos. La heterogeneidad es la norma, no la excepción. El
inventario te dice si te enfrentas a un formato o a cinco disfrazados del mismo.

**Lo que salió en el caso de las imágenes:** 9 archivos, todos del mismo observatorio
(SWF_Counterspace), todos en la misma carpeta `images/`, todos del fenómeno 2, entre 15 KB y
343 KB. Un solo origen, un solo scraper, tamaños uniformes. Eso simplifica mucho las cosas.

---

### Paso 2. Abrir los archivos y mirarlos con tus propios ojos

Este es el paso que la gente se salta y el que más caro sale.

**Preguntas:**

- ¿Qué hay *realmente* dentro de estos archivos?
- ¿La extensión del nombre corresponde al contenido real?
- ¿Son todos la misma clase de cosa, o hay subgrupos?
- ¿Los nombres de archivo dicen algo? (Muchas veces sí.)

**Concepto 1: la extensión miente.** El nombre de un archivo es solo una etiqueta que alguien
escribió. El contenido real se identifica por los primeros bytes del archivo, llamados
**número mágico** o *magic bytes*: una firma corta y fija que cada formato pone al principio.
Un JPEG empieza por `FF D8 FF`; un PNG por `89 50 4E 47`; un PDF por `%PDF`. Herramientas como
`file` en Linux/Mac hacen exactamente eso: ignoran la extensión y leen la firma.

Esto ya nos pasó en este proyecto: los 73 archivos `.pbf` no eran OpenStreetMap como todo el
mundo asumía. Se resolvió mirando los primeros bytes y viendo que eran *vector tiles* de
Mapbox. Esa comprobación de 30 segundos ahorró escribir el parser equivocado entero.

**Concepto 2: contenedor y códec no son lo mismo.** Un archivo de imagen tiene dos capas: el
*contenedor* (cómo se organiza el archivo, dónde está la metadata) y el *códec* (cómo están
comprimidos los píxeles). AVIF, por ejemplo, es un contenedor tipo HEIF con compresión AV1.
Esto importa por una razón muy práctica: **una librería puede saber leer el contenedor y no
tener el códec**, y entonces falla. Pillow, la librería estándar de imágenes en Python, no
abre AVIF sin un plugin adicional. Si tu parser asume que todas las imágenes se abren igual,
un archivo de nueve te revienta.

**Concepto 3: los nombres de archivo son metadata gratis.** Los scrapers suelen conservar el
nombre original del recurso web, y ese nombre lo puso un humano que sabía qué era la imagen.

**Lo que salió en el caso de las imágenes.** Los nombres ya lo gritaban: `table-5-1-web.jpg`,
`stoplight-chart-execsummary-web.jpg`, `asat-by-country-2026.jpg`. Al abrirlas, esto es lo que
hay de verdad:

| `doc_id` | Qué es realmente | ¿Lleva texto? |
|---|---|---|
| F2-SWF-076 | Tabla de datos completa: 17 filas × 9 columnas, pruebas ASAT con fechas, países, interceptores y cifras de basura orbital | **Muchísimo** |
| F2-SWF-077 | Matriz de evaluación: 13 países × 7 capacidades contraespaciales, con leyenda | **Muchísimo** |
| F2-SWF-089 | Gráfico de barras "ASAT Tests by Country (2026)": 4 países con cifras | **Poco pero denso** |
| F2-SWF-084 | Portada del informe: *Global Counterspace Capabilities — An Open Source Assessment, 04/2026* | **El título** |
| F2-SWF-066 | Foto: la ISS y una Soyuz sobre Europa de noche | No |
| F2-SWF-071 | Retrato de una persona | No |
| F2-SWF-065 | Retrato (AVIF) | No |
| F2-SWF-067, F2-SWF-068 | Fotos de archivo de la NASA (misiones del transbordador) | No |

**Y aquí está la lección más importante de toda la guía:** el sondeo del corpus decía de estas
9 imágenes que *"son fotos de misiones espaciales, no infografías: el OCR daría ruido"*, y
recomendaba dejarlas sin texto. **Era falso para 4 de las 9**, incluidas las dos que más
contenido tienen de todo el subconjunto. Nadie las había abierto.

Esa tabla ASAT es exactamente la clase de contenido que respondería una consulta como
*"¿qué países han realizado pruebas antisatélite y cuánta basura orbital generaron?"* — una
consulta perfectamente plausible en el fenómeno 2.

**Moraleja general: no confíes en la descripción de segunda mano de un formato, ni siquiera
si la escribió tu equipo. Ábrelo.**

---

### Paso 3. Preguntarte "¿qué cuenta como texto en este formato?"

Este es el salto conceptual de verdad, y es distinto para cada formato.

**Preguntas:**

- Si le tuviera que describir este archivo a alguien por teléfono, ¿qué le leería?
- ¿Qué parte de este archivo respondería una pregunta de un usuario?
- ¿Hay información que *es* texto pero no está guardada como texto?
- ¿Hay texto guardado que **no** debería indexarse (navegación, pies de página, avisos de
  cookies, identificadores internos)?

**El concepto.** "Extraer texto" suena obvio hasta que el formato no tiene texto dentro. Ahí
la pregunta se convierte en: *¿cómo convierto esta información en frases que un buscador
semántico pueda comparar con una pregunta en español?*

Fíjate cómo cambia la respuesta según el formato:

| Formato | ¿Qué es "el texto"? | La dificultad real |
|---|---|---|
| JSON de artículo | Los campos de cuerpo, en orden | Elegir bien los campos y no duplicar |
| PDF nativo | El texto de cada página | Orden de lectura, columnas, pies de página |
| PDF escaneado | **No hay texto**: hay una foto de un texto | Necesita OCR |
| CSV / XLSX | Cada fila como `columna: valor` | Distinguir tablas con semántica de tablas de identificadores |
| Vector tiles (PBF) | Los atributos de cada elemento del mapa | El mismo elemento se repite en varios niveles de zoom |
| **Imagen** | **Depende de si es una foto o un gráfico** | Decidir cuál es cuál |

**Por qué esto importa tanto en este proyecto.** La evaluación juzga la relevancia sobre el
**contenido textual** del fragmento que entregamos, no sobre su identificador. Si convertimos
una tabla en una lista de números sueltos sin sus encabezados, el texto resultante no se
parece a ninguna pregunta que un humano vaya a hacer, y el encoder no lo va a acercar a la
consulta. Por eso la especificación insiste en el formato `columna: valor` para tablas: cada
dato conserva pegado el nombre de lo que significa. "Russia" solo no dice nada; "country:
Russia. interceptor: Nudol. date: Nov. 15, 2021" sí.

**La pregunta específica del caso de imágenes:** una foto de la ISS no tiene texto, pero
**sí tiene descripción**. ¿Vale la pena inventarle una? Cuidado aquí: describir una imagen
con un modelo generativo está **prohibido** (ver Paso 5). Lo que sí es legítimo es usar lo
que ya existe: el nombre del archivo, el texto alternativo (`alt`) que el scraper guardó en
el JSON del artículo que contenía la imagen, o el pie de foto. Eso no es generación, es
recuperación de metadata que ya estaba escrita por un humano.

---

### Paso 4. Calcular si vale la pena, con números

**Preguntas:**

- ¿Cuántos documentos de los 1.826 gano si hago esto? ¿Qué porcentaje es?
- ¿Cuántas horas me cuesta?
- ¿Qué pasa si **no** lo hago? (¿Quedan invisibles, o quedan indexados con poco texto?)
- ¿Hay otra cosa pendiente que dé más puntos por la misma hora de trabajo?

**El concepto: coste de oportunidad.** Con fecha límite fija, cada hora que dedicas a un
formato es una hora que no dedicas a otro. La pregunta nunca es "¿esto es mejor que nada?"
(casi siempre lo es); la pregunta es "¿esto es lo mejor que puedo hacer con esta hora?".

**Cómo hacer la aritmética en este reto.** Las dos métricas se promedian sobre 50 consultas.
Un documento vale, como muchísimo, lo que valga en las consultas donde sea relevante. Con
1.826 documentos y 50 consultas, un documento suelto tiene poca probabilidad de aparecer en
el ground truth… **salvo que sea muy específico de un tema que seguro se pregunta.**

Y ahí está el matiz que cambia la respuesta para las imágenes: 9 archivos son el 0,5 % del
corpus, un número ridículo. Pero dos de esos 9 son *la* tabla de pruebas ASAT y *la* matriz
de capacidades contraespaciales por país, en un reto con un fenómeno entero dedicado a
seguridad espacial. La probabilidad de que alguna de las 50 consultas toque ese tema no es
del 0,5 %: es alta.

**Compáralo con el otro extremo:** los 73 tiles PBF también son el 4 % del corpus, pero son
atributos de mapa (geometrías, códigos de municipio) que difícilmente responden una pregunta
en lenguaje natural. Mismo porcentaje, retorno completamente distinto.

**La conclusión metodológica:** cuenta documentos, pero pondera por **densidad temática**. No
todos los documentos tienen la misma probabilidad de ser preguntados.

---

### Paso 5. Elegir la herramienta

**Preguntas — en este orden, porque el orden importa:**

1. **¿La licencia es permisiva?** (Apache 2.0, MIT, BSD, CC BY.) Si es GPL/AGPL/LGPL fuerte,
   **descarta y sigue**. No hay discusión ni excepción: descalifica la entrega entera.
2. **¿Es un modelo generativo (decoder)?** Si lo es, está prohibido. Ver más abajo.
3. ¿Necesita instalar algo que no sea un paquete de Python?
4. ¿La usa `generador.py`, o solo se usa al construir el índice?
5. ¿Cuánto tarda? ¿Cabe en el hardware que tenemos?

**Concepto: la frontera entre dependencias de construcción y de ejecución.** El evaluador va
a instalar `entrega/requirements.txt` en su máquina y correr `generador.py`. Ese script
**solo carga el índice ya construido y procesa las 50 consultas** — no vuelve a extraer nada.
Por lo tanto, todo lo que use tu parser va al grupo `dev` y **nunca** a las dependencias de
runtime.

Esto es una liberación enorme y conviene entenderla bien: puedes usar herramientas pesadas,
lentas, o que requieran instalar binarios del sistema, porque **solo corren en nuestras
máquinas**. Lo que no puedes es meterlas en `requirements.txt`, porque cada dependencia de
runtime es una forma más de que la entrega falle en una máquina que no controlamos.

**Concepto: dependencias que no son de Python.** Algunas librerías de Python son solo un
envoltorio fino sobre un programa escrito en C++ que hay que instalar aparte. El caso clásico
en OCR: el paquete de Python es diminuto, pero necesita el motor real instalado en el sistema
operativo. Consecuencia práctica: *"me funciona a mí"* no significa que le funcione al resto
del equipo. Si eliges algo así, documenta la instalación en el README el mismo día.

**⚠️ El punto que puede invalidar la entrega: OCR moderno y la prohibición de decoders.**

La especificación prohíbe absolutamente cualquier modelo generativo en cualquier etapa de
indexación o recuperación. Y aquí hay una trampa poco obvia:

- **Los motores de OCR clásicos** (el tipo Tesseract) usan redes que leen una línea y emiten
  caracteres. No son modelos generativos de lenguaje. **Permitidos.**
- **Muchos OCR "de nueva generación" son modelos multimodales con un decoder de lenguaje
  dentro** — literalmente un LLM que mira la imagen y *escribe* lo que ve. Aunque los llamen
  "OCR", arquitectónicamente son decoders generativos. Usarlos para construir el índice es
  **exactamente** lo que la regla prohíbe.

Esta distinción no es un tecnicismo: es la diferencia entre entregar y ser descalificado. La
pregunta que tienes que hacerle a cualquier herramienta de OCR antes de usarla es
**"¿esto lee o esto escribe?"**. Si por dentro genera texto token a token con un modelo de
lenguaje, no entra. Si tienes duda, la duda misma es motivo para no usarlo: no vale la pena
arriesgar la entrega entera por 4 imágenes.

Lo mismo aplica, con más razón, a "describir la foto de la ISS con un modelo de visión".
Eso es generación de texto. Está prohibido.

---

### Paso 6. Diseñar la salida antes de programarla

**Preguntas:**

- ¿Qué es un bloque en este formato? ¿Un archivo entero, o varias unidades?
- ¿Qué metadata puedo rescatar que no sea el cuerpo?
- ¿Cómo garantizo que nunca salgan cero bloques?
- ¿Qué pasa cuando el archivo está corrupto, vacío, o la herramienta falla?
- ¿El texto que produzco se puede cortar en oraciones completas?

**Concepto: separar el cuerpo de la metadata.** El cuerpo es lo que se indexa y se compara
con la consulta. La metadata (fecha, autor, URL, dimensiones) va aparte, porque sirve para
filtrar y para trazabilidad pero **ensucia** el texto si se mezcla. Una fecha suelta en medio
de un párrafo no ayuda al encoder; como campo de metadata, en cambio, permite post-filtros,
que sí están permitidos.

La excepción interesante: cuando el documento casi no tiene cuerpo, la metadata *es* el
contenido. Es lo que hicimos con las fichas de alerta, donde municipio y tipo de alerta dicen
más que el cuerpo. Criterio: *si sin esa metadata el documento es irreconocible, la metadata
es cuerpo.*

**Concepto: fallar por archivo, no por lote.** Con 1.826 archivos heterogéneos, algunos van a
fallar. La política del proyecto es que un fallo se registre en el log y el recorrido siga.
Un parser que lanza una excepción y detiene todo obliga a reprocesar 1.800 archivos por culpa
de uno. Diseña asumiendo que **algo va a estar roto** y que no lo sabrás hasta que lo corras.

**Concepto: la completitud lingüística y por qué las tablas son un caso especial.** La regla
obligatoria dice que ningún fragmento puede contener oraciones cortadas. Con prosa es claro:
al cortar, retrocedes al final de la última oración completa. Pero **una tabla no tiene
oraciones.** ¿Qué significa "no cortar" en una fila de datos?

La respuesta razonable — y que conviene dejar escrita en el ADR para poder defenderla en el
informe — es que **la unidad indivisible de una tabla es la fila**. Cortar a mitad de fila
produce datos huérfanos ("Nov. 15, 2021. country:" sin el país) que son peores que inútiles:
son engañosos. Si tratas cada fila como atómica y nunca la partes, cumples el espíritu de la
regla aunque el texto no tenga puntos.

**Concepto: el presupuesto de 250 palabras.** Cada fragmento entregado puede tener hasta 250
palabras, y está permitido concatenar un fragmento con su vecino del mismo documento para
rellenar. Un fragmento de 40 palabras desperdicia 210. Esto no es un detalle: es una palanca
directa sobre la métrica de fragmentos. Cuando diseñes los bloques, ten en la cabeza que
bloques diminutos y aislados aprovechan mal ese presupuesto.

---

### Paso 7. Verificar con números y dejar la decisión escrita

**Preguntas:**

- ¿Cuántos documentos procesé, cuántos fallaron, cuántas palabras salieron?
- ¿Abrí una muestra de la salida y la leí? ¿Se entiende?
- ¿Cuántos documentos quedaron con contenido mínimo o vacío?
- ¿Sé decir *por qué* elegí esto, en tres frases, dentro de seis días?

**El concepto.** "Funciona" no es un resultado, es una impresión. El resultado es
*"954 documentos, 0 fallidos, 676.552 palabras, mediana 212 por documento"*. Con números
puedes comparar; con impresiones solo puedes discutir.

**Por qué en este proyecto, específicamente.** El informe técnico tiene 8 páginas y hay que
justificar las decisiones de diseño. Esa justificación se escribe desde el registro de
decisiones, no de memoria. Si el 7 de agosto hay que reconstruir por qué se hizo OCR de 4
imágenes y no de 9, se pierde tiempo y se pierden puntos en el criterio de documentación.

Una cosa más, muy barata y muy útil: **lee la salida con tus ojos.** Diez ejemplos al azar.
Los números te dicen que hay texto; solo mirarlo te dice si el texto tiene sentido.

---

## Parte 2 — Conceptos que conviene tener claros

### Qué hace el OCR por dentro, en cristiano

OCR significa *reconocimiento óptico de caracteres*: convertir la foto de un texto en texto.
Por dentro son cuatro etapas, y cada una puede fallar:

1. **Preprocesado.** Enderezar la imagen, subir el contraste, y **binarizar**: convertir todo
   a blanco y negro puro decidiendo, píxel a píxel, si es tinta o es papel. Suena trivial y
   es donde más se pierde: una imagen con fondo degradado o texto claro sobre oscuro puede
   binarizarse mal y borrar letras enteras.
2. **Análisis de disposición** (*layout*). Encontrar dónde hay bloques de texto, en qué orden
   se leen, dónde están las columnas. Aquí es donde una tabla se rompe: si el motor no
   entiende que hay una rejilla, lee de izquierda a derecha atravesando columnas y mezcla
   datos de campos distintos.
3. **Segmentación en líneas y palabras.**
4. **Reconocimiento.** Convertir cada línea en caracteres.

**Los tres ajustes que más cambian el resultado:**

- **Resolución.** Los motores clásicos esperan una densidad equivalente a unos 300 puntos por
  pulgada. Una imagen web de 400 píxeles de ancho tiene mucha menos: **agrandarla antes de
  pasarla por OCR suele mejorar el resultado más que cualquier otro ajuste.**
- **Modo de segmentación.** Decirle al motor si le estás dando una página completa, una sola
  columna, una sola línea o una sola palabra. El modo por defecto asume página completa y es
  mala elección para un gráfico.
- **Idioma.** Los motores usan modelos por idioma. Pasarle inglés a un modelo de español
  degrada el resultado. En nuestro corpus esto importa: hay documentos en tres idiomas.

### Por qué una tabla es el caso difícil

Un humano lee una tabla en dos dimensiones: sabe que la celda "Russia" está en la columna
*country* porque está debajo de ese encabezado. El OCR clásico lee en una dimensión, línea a
línea. Sin un paso explícito de reconocimiento de estructura, una tabla sale como una tira de
palabras sueltas donde se perdió justo lo que le daba sentido: la relación entre cada dato y
su encabezado.

Por eso, para tablas, o usas una herramienta que reconozca estructura de tabla, o aceptas
que el texto va a ser de peor calidad, o —opción muy razonable con 3 imágenes— **transcribes
a mano.** Con 4 archivos, transcribir es más rápido y más fiable que afinar un motor de OCR.
Y transcribir a mano no viola ninguna regla: la prohibición es de modelos generativos, no de
seres humanos con un teclado.

### Metadata incrustada (EXIF y compañía)

Las imágenes pueden traer metadata dentro del archivo: fecha de captura, cámara, a veces
título o descripción escritos por el autor. Casi nunca hay nada útil en imágenes descargadas
de web (los optimizadores suelen borrarla), pero mirar cuesta un minuto y a veces aparece un
título completo. Es información que ya existe: usarla no es generar nada.

### Detección de idioma

Consiste en adivinar en qué idioma está un texto a partir de su estadística de letras y
palabras. Es fácil y barato, pero **falla con textos muy cortos**: con 5 palabras la
predicción es poco fiable. Guardarlo como metadata es útil (permite post-filtros, que están
permitidos) siempre que se guarde también la confianza y no se trate como verdad absoluta.

Este es además uno de los dos huecos pendientes de la etapa de limpieza de la especificación:
*"detectar y marcar idioma predominante"* y *"eliminar boilerplate"*. Ninguno está hecho.

### Boilerplate

Es el texto de relleno que aparece en todos los documentos de una misma fuente y no dice nada
del contenido: menús de navegación, "Todos los derechos reservados", "Compartir en Twitter",
números de página. Ensucia el índice porque hace que documentos distintos se parezcan entre
sí, que es exactamente lo contrario de lo que necesita un buscador.

La forma sencilla de detectarlo, sin nada sofisticado: **buscar bloques de texto idénticos
que se repiten en muchos documentos del mismo observatorio.** Si la misma frase aparece en
180 de 186 documentos, es plantilla.

---

## Parte 3 — El caso concreto: tu parser de imágenes

Ya tienes los datos del Paso 2. Lo que queda son decisiones, y son **tuyas**. Estas son las
preguntas abiertas, ordenadas de más a menos importante:

**1. ¿Se hace OCR o no?**
Los datos dicen que 4 de 9 imágenes tienen texto valioso y que 2 de esas 4 son contenido de
alto valor temático para el fenómeno 2. Los datos también dicen que son 9 archivos de 1.826.
Decide, y escribe el porqué.

**2. Si se hace, ¿cómo se distingue una foto de una infografía?**
Opciones: por el nombre del archivo (`table`, `chart`, `figure` frente a IDs numéricos de la
NASA); por una medida de la imagen (las infografías suelen tener pocos colores y mucho blanco,
las fotos muchos colores); o **a mano, porque son nueve**. Con 9 archivos, cualquier
heurística automática es más código y más riesgo que una lista explícita. Con 900 sería al
revés. Ese cambio de respuesta según la escala es el aprendizaje.

**3. ¿Qué haces con las 5 fotos sin texto?**
No pueden quedar sin bloques (invariante). ¿Qué es lo mínimo identificable que tienen? Piensa
en el nombre del archivo, el observatorio, el fenómeno, y en si el JSON del artículo que
contenía esa imagen guardó un `alt` o un pie de foto — porque si lo guardó, ahí hay una
descripción escrita por un humano, gratis.

**4. ¿Qué haces con el AVIF?**
Es un archivo de nueve y necesita un plugin extra. Preguntas: ¿instalas la dependencia por un
solo archivo? ¿Lo conviertes una vez y trabajas con el resultado? ¿Lo tratas como foto sin
texto (que es lo que es: un retrato) y te ahorras el problema entero? Ojo con el índice de
ADL, que lo clasifica en el cajón "Otro" mientras su extensión real es `avif` — un recordatorio
de que el `formato` de la metadata sale de la extensión real, no de la columna del índice.

**5. Si transcribes una tabla, ¿cómo la estructuras?**
Vuelve al concepto de `columna: valor` y al de la fila como unidad atómica. Piensa qué texto
se parecería más a una pregunta escrita en español por un evaluador.

**6. ¿Cuántos bloques produce una imagen?**
¿La tabla entera es un bloque, o cada fila es un bloque? Piensa en el presupuesto de 250
palabras y en cómo va a quedar el fragmento que se entregue.

---

## Parte 4 — Checklist para cualquier formato nuevo

Antes de escribir código:

- [ ] Sé cuántos archivos son, de qué observatorios y de qué fenómenos
- [ ] Abrí al menos 5 con mis propios ojos, no solo su descripción
- [ ] Verifiqué que la extensión corresponde al contenido real
- [ ] Sé qué subgrupos distintos hay dentro del formato
- [ ] Contesté "¿qué es texto aquí?" por escrito
- [ ] Calculé el retorno frente al coste, con números
- [ ] La licencia de cada herramienta es permisiva — verificada, no supuesta
- [ ] Ninguna herramienta que uso es un modelo generativo
- [ ] Sé si mis dependencias van al grupo `dev` (casi siempre) o a runtime (casi nunca)

Al escribir:

- [ ] Devuelvo bloques en orden de lectura, no una masa de texto
- [ ] Ningún documento sale con cero bloques
- [ ] La metadata va separada del cuerpo
- [ ] Un archivo roto no tumba el recorrido
- [ ] `doc_id`, `fuente`, `formato` y `fenomeno` salen del catálogo, no los invento

Al terminar:

- [ ] Tengo números: procesados, fallidos, palabras, mediana
- [ ] Leí 10 salidas al azar y tienen sentido
- [ ] Registré la decisión en `docs/decisions/` con su porqué
- [ ] Actualicé lo que este trabajo haya demostrado falso en otros documentos

---

## Parte 5 — Recursos para aprender

> Los enlaces marcados con ✓ los verifiqué al escribir esta guía. El resto son páginas
> oficiales y estables, pero si alguno se ha movido, búscalo por el título.

### Para entender el problema de fondo (recuperación de información)

- ✓ **[Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)** — Manning,
  Raghavan y Schütze. Gratis y completo. **Los capítulos 1, 2 y 6** explican por qué el
  preprocesamiento decide la calidad de la búsqueda. Es el libro de referencia del área.
- **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** — Jurafsky y
  Martin, 3.ª edición, borrador gratuito. El capítulo de embeddings vectoriales explica qué
  hace realmente un encoder cuando "entiende" un texto.
- **[Canal de James Briggs en YouTube](https://www.youtube.com/@jamesbriggs)** — vídeos
  prácticos sobre búsqueda semántica, embeddings y bases vectoriales.

### Para entender OCR

- ✓ **[Optical Character Recognition (OCR) — Computerphile](https://www.youtube.com/watch?v=ZNrteLp_SvY)**
  — 15 minutos, explicación conceptual sin matemáticas. El mejor punto de partida.
- ✓ **[Documentación oficial de Tesseract](https://tesseract-ocr.github.io/tessdoc/)** — el
  motor de OCR clásico de referencia, licencia Apache 2.0.
- ✓ **[Improving the quality of the output](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)**
  — la página que hay que leer **antes** de concluir que "el OCR no funciona". Resolución,
  binarización, inclinación, modos de segmentación.
- **[pytesseract](https://pypi.org/project/pytesseract/)** — el envoltorio de Python. Recuerda
  que necesita el motor instalado aparte.
- ✓ **[Comparativa de motores OCR de código abierto (2026)](https://unstract.com/blog/best-opensource-ocr-tools/)**
  y ✓ **[la de LlamaIndex](https://www.llamaindex.ai/blog/best-ocr-libraries-for-developers)**
  — útiles para ver el panorama, **pero léelas con la pregunta de la prohibición de decoders
  en la cabeza**: varias de las opciones que recomiendan son modelos multimodales generativos
  y aquí no se pueden usar.
- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)** (Apache 2.0) — incluye
  reconocimiento de estructura de tablas, que es justo el punto débil del OCR clásico.
- **[docTR](https://github.com/mindee/doctr)** (Apache 2.0) — detección y reconocimiento con
  aprendizaje profundo, sin binarios externos.

### Para entender formatos e imágenes

- **[Documentación de Pillow](https://pillow.readthedocs.io/en/stable/)** — la librería
  estándar de imágenes en Python. Mira la lista de formatos soportados y sus límites.
- **[Pillow — ExifTags](https://pillow.readthedocs.io/en/stable/reference/ExifTags.html)** —
  cómo leer la metadata incrustada.
- **[pillow-avif-plugin](https://pypi.org/project/pillow-avif-plugin/)** — el plugin que hace
  falta para AVIF.
- **[Lista de firmas de archivo (magic numbers)](https://en.wikipedia.org/wiki/List_of_file_signatures)**
  — la tabla de "primeros bytes" de cada formato. Guárdala.

### Para entender texto y codificación

- **[The Absolute Minimum Every Software Developer Must Know About Unicode](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/)**
  — Joel Spolsky. Es de 2003 y sigue siendo la mejor explicación que existe de por qué el
  texto se rompe.
- **[Unicode Normalization Forms (UAX #15)](https://unicode.org/reports/tr15/)** — qué es NFC
  y por qué normalizamos antes de indexar. Denso pero es la fuente.
- **[pySBD](https://github.com/nipunsadvilkar/pySBD)** — segmentación de oraciones que sí
  entiende abreviaturas, decimales y siglas. Relevante para la regla de completitud
  lingüística.

### Para escribir mejor código de pipeline

- **[A Philosophy of Software Design](https://web.stanford.edu/~ouster/cgi-bin/book.php)** —
  John Ousterhout. Corto. El capítulo sobre "módulos profundos" (interfaz simple, mucha
  complejidad escondida detrás) es exactamente lo que buscamos en un parser: una función,
  un contrato, todo el desorden dentro.
- **[Canal de ArjanCodes](https://www.youtube.com/@ArjanCodes)** — diseño de software en
  Python, muy accesible. Busca sus vídeos sobre inyección de dependencias y sobre cuándo
  *no* abstraer.
- **[Documentación de logging de Python](https://docs.python.org/3/howto/logging.html)** —
  por qué el pipeline usa logging en vez de imprimir por pantalla.

### Para el asunto de las licencias

- **[Lista de licencias SPDX](https://spdx.org/licenses/)** — el catálogo canónico.
- **[choosealicense.com](https://choosealicense.com/)** — explicación en lenguaje llano de qué
  te obliga cada licencia. Lo que hay que mirar es si es *copyleft fuerte*.

---

## Documentos relacionados en este repo

- `CLAUDE.md` — las reglas duras y las contradicciones resueltas
- `.claude/reference/spec-etapa1.md` — la especificación condensada; la sección 2 es la que
  manda sobre preprocesamiento
- `docs/sondeo-corpus.md` — el inventario del corpus (con la advertencia del Paso 2: sobre las
  imágenes se equivocaba)
- `docs/decisions/001-extraccion-catalogo-y-json.md` — un ADR real, como referencia de formato
- `src/extract/` — el parser de JSON, como ejemplo de la forma que toma todo esto
