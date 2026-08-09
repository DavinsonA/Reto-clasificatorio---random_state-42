# ADR-002: Parser de imágenes y estrategia de OCR

- **Fecha**: 2026-08-04
- **Estado**: aceptada
- **Responsable**: Davinson Arteaga

## Contexto

El corpus trae 9 imágenes (8 JPG + 1 AVIF), todas de SWF_Counterspace, fenómeno 2. El sondeo
las había descartado como *"fotos de misiones espaciales, no infografías: el OCR daría ruido"*
y recomendaba dejarlas sin texto.

**Al abrirlas, esa conclusión resultó falsa para 4 de las 9**: una tabla de datos completa de
pruebas ASAT (16 filas × 9 columnas), una matriz de 13 países × 7 capacidades contraespaciales,
un gráfico de barras y la portada del informe. Las otras 5 sí son retratos y fotos de archivo
sin texto.

Restricción crítica: la prohibición de decoders (§8.3) alcanza al OCR. Los motores clásicos
(Tesseract) leen y están permitidos; los OCR multimodales modernos llevan un decoder de
lenguaje dentro y **escriben** lo que ven — usarlos para indexar es exactamente lo prohibido.

## Opciones consideradas

| Opción | A favor | En contra |
|---|---|---|
| Dejarlas sin texto | cero trabajo | pierde los 2 documentos más densos del subconjunto, en un fenómeno donde seguro hay consultas sobre ASAT |
| OCR con Tesseract | automático, escala | requiere binario del sistema; el OCR clásico destroza tablas porque lee en una dimensión; sobre 4 archivos es más trabajo que el resultado |
| OCR multimodal | mejor con tablas | **prohibido**: es un decoder generativo |
| **Transcripción manual + OCR opcional** ✅ | exacto, sin dependencias, sin riesgo de regla, reproducible | trabajo humano; error de transcripción silencioso |

## Decisión

**Transcripción manual como camino principal, OCR como rama opcional.**

El parser intenta tres cosas en orden: transcripción manual del `doc_id` si existe → OCR si
está disponible → bloque mínimo de metadata. Con 4 archivos, transcribir es más rápido y más
exacto que afinar un motor de OCR. Con 400 la respuesta sería la contraria; el criterio es la
escala, no el gusto.

Decisiones de diseño que van con ello:

1. **`ocr()` acepta una ruta o una imagen ya en memoria.** Un PDF escaneado es una imagen de
   un texto: al rasterizar una página se obtiene exactamente lo que esta función consume. Los
   48 PDF escaneados del corpus reutilizarán esta pieza sin duplicar nada.
2. **Umbral de 20 palabras para aceptar OCR.** Una foto pasada por OCR devuelve caracteres
   sueltos. Por debajo del umbral se descarta y el documento cae al bloque mínimo. Evita meter
   ruido en el índice sin tener que clasificar foto contra infografía.
3. **Las transcripciones viven en `assets/transcripciones/`, no en `data/`.** El `.gitignore`
   ignora `data/*` entero; ahí se perderían en el próximo clone. Son trabajo humano
   irrepetible y tienen que estar versionadas.
4. **Una fila por línea, y la fila es atómica.** Es nuestra lectura del requisito de
   completitud lingüística (§3.3) aplicado a contenido tabular, que no tiene oraciones. Cortar
   a mitad de fila produce datos huérfanos que engañan más de lo que informan.
5. **Formato `columna: valor`.** El mismo que la especificación exige para CSV/XLSX. Cada dato
   lleva pegado el nombre de su columna: `Russia` solo no responde nada.
6. **El AVIF no necesita tratamiento especial.** Pillow no lo abre sin plugin, pero es un
   retrato sin texto: la excepción se captura y cae al bloque mínimo. Instalar una dependencia
   por un archivo del que no se extrae nada es coste puro.

## Consecuencias

**Qué se gana.** 9 documentos, 0 fallidos, 866 palabras (frente a 53 con solo metadata). Los
4 documentos con contenido aportan 839 de esas palabras. El parser son 84 líneas y **no añade
ninguna dependencia**: sin `pytesseract` instalado degrada a bloque mínimo y registra un aviso.

**Qué se pierde o queda por verificar.**

- **Las transcripciones no están verificadas.** Un error de transcripción es silencioso: no
  falla ninguna prueba y acaba en el índice como dato correcto. Las dos tablas suman ~150
  celdas. Está anotado en `assets/transcripciones/README.md`; hace falta que alguien las
  contraste contra la imagen una vez.
- Los valores de F2-SWF-089 son aproximados por construcción: el gráfico no trae etiquetas
  numéricas. El propio bloque lo dice, para que quede en el texto indexado.
- Las 5 fotos quedan con un bloque de 4–7 palabras derivado del nombre de archivo. Cumple el
  invariante pero no vale nada en recuperación cuando el nombre es un identificador de la NASA
  (`38236`). La vía de mejora es rescatar el `alt` o el pie de foto del JSON de SWF que
  enlazaba la imagen — texto escrito por una persona, no generado.

**Qué revisar si esto resulta equivocado.** Si al llegar a los PDF escaneados el OCR demuestra
buena calidad sobre las 582 páginas, conviene reevaluar si merece la pena pasarlo también por
las imágenes. La pieza ya está lista para eso.
