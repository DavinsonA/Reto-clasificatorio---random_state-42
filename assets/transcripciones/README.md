# Transcripciones manuales

Texto transcrito a mano de documentos cuyo contenido es una imagen (tablas, gráficos,
matrices). El parser de imágenes las prefiere sobre el OCR: con pocos archivos, transcribir
es más rápido, más exacto y no depende de instalar nada.

Van aquí y no en `data/` porque el `.gitignore` ignora `data/*` entero: son trabajo humano
irrepetible y tienen que estar versionadas.

## Formato

Un archivo por documento, nombrado con su `doc_id` exacto: `F2-SWF-076.txt`.

**Una unidad por línea.** Cada línea se convierte en un bloque, y un bloque nunca se parte.
Para contenido tabular, una fila por línea en formato `columna: valor` separado por puntos:

```
date: Nov. 15, 2021. country: Russia. interceptor: Nudol. interceptor type: Direct Ascent. target: Cosmos 1408. intercept altitude: 470 km. tracked debris: 1807. debris still on orbit: 5
```

Cada dato lleva pegado el nombre de su columna. `Russia` solo no responde nada;
`country: Russia. interceptor: Nudol` sí se parece a una pregunta en lenguaje natural.

Para texto corrido (una portada, un pie de figura), una frase por línea y ya.

## Estado

| `doc_id` | Qué es | Bloques | Palabras | Estado |
|---|---|---:|---:|---|
| F2-SWF-076 | Tabla 5-1: pruebas ASAT, 16 filas × 9 columnas + total | 18 | 515 | ⚠️ verificar |
| F2-SWF-077 | Matriz 2026: 13 países × 7 capacidades contraespaciales | 9 | 259 | ⚠️ verificar |
| F2-SWF-089 | Gráfico de barras: ASAT Tests by Country (2026), 4 países | 5 | 50 | ⚠️ verificar |
| F2-SWF-084 | Portada: *Global Counterspace Capabilities — An Open Source Assessment* | 1 | 15 | listo |

> ⚠️ **Verificar contra la imagen antes de indexar.** Estas transcripciones se hicieron leyendo
> las imágenes, y un error de transcripción es **silencioso**: no falla ninguna prueba, no
> aparece en ningún log, y acaba en el índice como si fuera un dato correcto. Las dos primeras
> tienen ~150 celdas entre fechas, países y cifras. Que alguien las contraste una vez.
>
> En F2-SWF-089 los valores son aproximados a propósito: el gráfico no trae etiquetas
> numéricas y las barras se leen contra el eje. Está dicho en el texto del propio bloque.

Las otras 5 imágenes del corpus son retratos y fotos de archivo sin texto: no necesitan
transcripción y el parser les genera un bloque mínimo de metadata.
