# Cómo escribir un parser — versión técnica

> Manual de implementación. La versión conceptual está en
> [`como-escribir-un-parser.md`](como-escribir-un-parser.md): qué preguntas hacerte y por qué.
> Esta es la otra mitad: **qué contrato respetar y cómo se escribe el código**.
>
> Lee la conceptual primero. Si escribes el código sin haber contestado sus preguntas, vas a
> escribir un parser correcto para el problema equivocado.

**Fecha:** 2026-08-04 · **Caso de ejemplo:** el parser de imágenes

---

## 1. El contrato, en código

Todo parser es **una función**. Esta es su firma, y no admite variantes:

```python
def extract(entry: CatalogEntry) -> RawDoc: ...
```

Los dos tipos están en [`src/extract/core.py`](../../src/extract/core.py) y son `frozen`
(inmutables) a propósito: nadie puede modificar un documento después de crearlo.

### Lo que recibes

```python
@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """Fila del indice de ADL resuelta contra el disco."""

    doc_id: str          # "F2-SWF-076", literal de ADL. NUNCA lo inventes.
    fuente: str          # ruta relativa POSIX, clave del ground truth
    formato: str         # extension real en minusculas, sin punto
    fenomeno: int        # 1, 2 o 3
    observatory: str     # "SWF_Counterspace"
    path: Path           # ruta absoluta al archivo
```

Los cuatro primeros campos **se copian tal cual** al `RawDoc`. No los recalculas, no los
normalizas, no los deduces de la ruta. Vienen del índice de ADL porque el emparejamiento con
el ground truth depende de que sean exactos.

### Lo que devuelves

```python
@dataclass(frozen=True, slots=True)
class RawDoc:
    """Texto de un documento sin fragmentar. Invariante: nunca cero bloques."""

    doc_id: str
    fuente: str
    formato: str
    fenomeno: int
    title: str                  # puede ser "" si el formato no tiene titulo
    blocks: tuple[str, ...]     # >= 1 elemento, ninguno vacio, en orden de lectura
    extra: dict[str, Any] = field(default_factory=dict)
```

**Las tres reglas que el tipo no puede imponer solo y tú sí tienes que cumplir:**

1. `blocks` **nunca** está vacío.
2. Ningún bloque es cadena vacía ni solo espacios.
3. Los bloques van en **orden de lectura** del documento original.

La tercera importa más de lo que parece: el chunker asigna el campo `posicion` de la Tabla 1
según ese orden, y la regla de las 250 palabras permite concatenar un fragmento con su vecino
inmediato. Si el orden está mal, concatenar produce texto incoherente.

---

## 2. La plantilla mínima

Esto es un parser completo y válido. Todo lo demás son variaciones sobre esto:

```python
"""Parser de <FORMATO>: <una frase sobre que hay dentro de estos archivos>."""

from __future__ import annotations

from typing import Any

from .core import CatalogEntry, RawDoc, clean


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un <FORMATO> del corpus en `RawDoc`."""
    raw = _read(entry.path)                       # lo especifico del formato
    blocks = [block for block in map(clean, raw) if block]

    extra: dict[str, Any] = {"observatorio": entry.observatory}
    if not blocks:
        # Un doc_id sin chunk no se puede recuperar jamas: F1@3 perdido (R4).
        blocks = [f"{entry.path.stem}. Observatorio: {entry.observatory}"]
        extra["contenido_minimo"] = True

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title="",
        blocks=tuple(blocks),
        extra=extra,
    )
```

Fíjate en el patrón de tres líneas que se repite en todos los parsers:

```python
blocks = [block for block in map(clean, raw) if block]   # 1. limpiar y descartar vacios
if not blocks:                                            # 2. red de seguridad
    blocks = [...]                                        # 3. bloque minimo
```

`clean()` está en `core.py` y hace lo mismo para todos los formatos: normaliza a NFC, borra
invisibles de ancho cero, quita caracteres de control y colapsa espacios. Devuelve `""` si le
pasas algo que no es una cadena, y por eso `map(clean, raw)` es seguro aunque `raw` traiga
`None` o números sueltos.

**No escribas tu propia limpieza.** Si tu formato necesita una limpieza extra que ningún otro
necesita, va dentro de tu parser; si la necesitan dos o más, va a `core.py`.

---

## 3. Registrar el parser

Una línea en [`src/extract/__init__.py`](../../src/extract/__init__.py):

```python
from .images import extract as extract_image

PARSERS = {
    "json": extract_json,
    "jpg": extract_image,
    "jpeg": extract_image,
    "png": extract_image,
    "avif": extract_image,
}
```

Las claves son **extensiones reales en minúscula y sin punto**, porque así se construye el
campo `formato` en `load_catalog()`. Un formato puede tener varias extensiones; regístralas
todas o los archivos se saltan en silencio.

Con eso ya funciona:

```bash
uv run python -m src.extract --formato jpg avif
```

No hay nada más que tocar. `extract_all()` busca en `PARSERS`, y lo que no está registrado
simplemente no se procesa.

---

## 4. Manejo de errores: qué capturar y qué no

`extract_all()` ya envuelve cada archivo:

```python
try:
    doc = parser(entry)
except Exception:
    failed += 1
    logger.warning("fallo | %s | %s", entry.doc_id, entry.fuente, exc_info=True)
    continue
```

**Consecuencia para ti: tu parser no necesita `try/except` para "que no se caiga".** Deja que
la excepción suba. El recorrido la captura, la registra con traza completa y sigue con el
siguiente archivo.

Solo captura una excepción cuando **sabes qué hacer con ella**:

```python
# BIEN: sabes que hacer y lo haces.
try:
    image = Image.open(entry.path)
except UnidentifiedImageError:
    # Sin pixeles legibles no hay texto, pero el documento no puede desaparecer.
    return _solo_metadata(entry)

# MAL: esconde el error y devuelve un documento silenciosamente roto.
try:
    ...
except Exception:
    return None
```

Devolver `None` está prohibido por el contrato: la firma dice `-> RawDoc`. Si no puedes
producir un documento, **lanza**; si puedes producir uno pobre, **prodúcelo**.

### Nunca uses `print()`

El pipeline usa `logging`. Un `print` en un recorrido de 1.826 archivos hace ilegible la
salida y no se puede filtrar por nivel.

```python
import logging

logger = logging.getLogger(__name__)

logger.debug("OCR aplicado | %s | %d palabras", entry.doc_id, len(text.split()))
logger.warning("imagen ilegible | %s", entry.fuente)
```

Usa los marcadores `%s` en vez de f-strings: si el nivel está desactivado, el mensaje ni
siquiera se construye.

---

## 5. Convenciones del repo que aplican al código

| Regla | Detalle |
|---|---|
| Idioma | Código, funciones y variables en **inglés**. Docstrings y comentarios en **español**. |
| Nombres | Módulos `snake_case`, clases `PascalCase`, constantes `UPPER_SNAKE_CASE` |
| Type hints | Obligatorios en funciones públicas |
| Docstrings | Una línea basta. Los bloques `Args:`/`Returns:` sobran si la firma ya lo dice. |
| Privado | Un guion bajo delante: `_read_transcript`. Solo `extract` es público. |
| Rutas | Nunca absolutas en el código. Se derivan de `entry.path` o de las constantes de `core.py`. |
| Comentarios | Solo para lo que el código **no puede** decir: un porqué, un riesgo, una decisión. |

**Sobre Python 3.9:** esa restricción aplica **solo** a `entrega/generador.py`, que corre en la
máquina del evaluador. En `src/` puedes usar todo lo de 3.11+ sin problema — el walrus `:=`,
`X | Y` en anotaciones, `slots=True` en dataclasses.

Antes de dar por terminado:

```bash
uv run --with ruff ruff format --line-length 100 src/
uv run --with ruff ruff check --line-length 100 src/
```

---

## 6. Dependencias: dónde va cada cosa

La pregunta única: **¿lo importa `entrega/generador.py`?**

```bash
uv add --group dev pillow          # SI: tu parser corre al construir el indice
uv add pillow                      # NO HAGAS ESTO salvo que generador.py lo importe
```

`generador.py` solo carga el índice ya construido y procesa las 50 consultas. **Ningún parser
corre en la máquina del evaluador.** Por eso puedes usar herramientas pesadas, lentas o que
necesiten binarios del sistema: solo corren aquí.

Antes de añadir cualquier cosa, verifica la licencia. Copyleft fuerte (GPL/AGPL/LGPL)
descalifica la entrega:

```bash
uv run python -c "import importlib.metadata as m; print(m.metadata('pillow')['License-Expression'])"
```

**Recordatorio del punto crítico:** si la herramienta es un modelo generativo (un decoder, un
modelo multimodal que *escribe* lo que ve), está prohibida en cualquier etapa de indexación.
Los motores de OCR clásicos leen; los VLM escriben. Ver §5 de la guía conceptual.

---

## 7. Caso completo: el parser de imágenes

Recuerda el diagnóstico: 9 imágenes, **4 con texto valioso** (una tabla de datos, una matriz
de países, un gráfico de barras, una portada) y 5 fotos sin texto.

### 7.1 La decisión de diseño y por qué

Con 4 archivos, **transcribir a mano gana a montar OCR**: es más rápido, el resultado es
perfecto, no añade dependencias, no arriesga la regla de decoders, y es reproducible por
definición. Con 400 archivos la respuesta sería la contraria.

Entonces el parser hace tres cosas, en orden de preferencia:

1. Si existe una **transcripción manual** para ese `doc_id`, la usa.
2. Si no, y está activado el OCR opcional, lo intenta.
3. Si no, produce un bloque mínimo con la metadata (invariante de "nunca cero bloques").

### 7.2 Dónde viven las transcripciones

En `assets/transcripciones/<doc_id>.txt`, **fuera de `data/`**. Motivo concreto: el
`.gitignore` ignora `data/*` entero, así que una transcripción ahí no se commitea y se pierde
en el próximo clone. Y estas transcripciones son trabajo humano irrepetible: tienen que estar
versionadas.

```
assets/transcripciones/
  F2-SWF-076.txt     # tabla 5-1, una fila por linea
  F2-SWF-077.txt     # matriz de capacidades
  F2-SWF-089.txt     # grafico de barras
  F2-SWF-084.txt     # portada
```

Formato del contenido: **una unidad por línea**, en `columna: valor` separado por puntos.
Ejemplo de una línea de `F2-SWF-076.txt`:

```
date: Nov. 15, 2021. country: Russia. interceptor: Nudol. interceptor type: Direct Ascent. target: Cosmos 1408. intercept altitude: 470 km. tracked debris: 1807. debris still on orbit: 5. total debris lifespan: 4.3 years
```

Por qué así: cada dato lleva pegado el nombre de su columna. "Russia" solo no responde nada;
"country: Russia. interceptor: Nudol" sí se parece a una pregunta en lenguaje natural. Es la
misma forma que la especificación exige para CSV/XLSX y que ya usamos en `alerta_meta`.

**Una fila por línea, y la fila es atómica.** Nunca se parte. Es nuestra interpretación de la
regla de completitud lingüística para contenido tabular, que no tiene oraciones — y hay que
dejarla escrita en el ADR para poder defenderla en el informe.

### 7.3 El parser

```python
"""Parser de imagenes: 9 archivos, 4 con texto (tablas y graficos) y 5 fotos."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .core import ROOT, CatalogEntry, RawDoc, clean

logger = logging.getLogger(__name__)

TRANSCRIPTS = ROOT / "assets" / "transcripciones"


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte una imagen del corpus en `RawDoc`."""
    transcript = TRANSCRIPTS / f"{entry.doc_id}.txt"
    extra: dict[str, Any] = {"observatorio": entry.observatory}

    if transcript.is_file():
        raw = transcript.read_text(encoding="utf-8").splitlines()
        extra["transcripcion"] = "manual"
    else:
        raw = []

    blocks = [block for block in map(clean, raw) if block]
    if not blocks:
        # Una foto no tiene texto, pero su doc_id no puede quedar sin chunk (R4).
        blocks = [_describe(entry)]
        extra["contenido_minimo"] = True

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title=clean(entry.path.stem.split("-", 1)[-1].replace("-", " ")),
        blocks=tuple(blocks),
        extra=extra,
    )


def _describe(entry: CatalogEntry) -> str:
    """Bloque minimo para una imagen sin texto: lo poco que la identifica."""
    name = entry.path.stem.split("-", 1)[-1].replace("-", " ")
    return f"Imagen: {name}. Observatorio: {entry.observatory}"
```

Son 35 líneas y no tiene ninguna dependencia nueva. El scraper de SWF prefija los nombres con
un hash (`SWF_69cac182...-table-5-1-web`), y por eso el `split("-", 1)[-1]` se queda con la
parte legible.

**Corrido de verdad sobre las 9 imágenes, esto es lo que produce** — y muestra un límite que
conviene ver antes de confiarse:

```
F2-SWF-076  title='table 5 1 web'          -> transcripcion manual
F2-SWF-066  title='38236'                  -> "Imagen: 38236. Observatorio: SWF_Counterspace"
F2-SWF-068  title='sts063 712 072medium'   -> "Imagen: sts063 712 072medium. ..."
```

El nombre de archivo es buena metadata cuando lo escribió un humano (`table-5-1-web`) y basura
cuando es un identificador de archivo de la NASA (`38236`). El bloque mínimo cumple el
invariante, pero para esas fotos es texto sin valor de recuperación. **Si quieres que valgan
algo, la vía es buscar el `alt` o el pie de foto en el JSON del artículo de SWF que enlazaba
esa imagen** — ahí hay una descripción escrita por una persona, y usarla no es generar nada.
Eso es trabajo extra; decide con el criterio de coste/beneficio de la guía conceptual.

### 7.4 Si aun así quieres OCR

Añádelo como **una rama opcional**, nunca como el camino principal, y desactivado por defecto:

```bash
uv add --group dev pillow pytesseract
```

`pytesseract` es solo un envoltorio: necesita el binario de Tesseract instalado en el sistema
operativo. Documenta esa instalación en el README el mismo día que lo añadas, o al resto del
equipo le va a fallar sin entender por qué.

```python
def _ocr(path: Path, lang: str = "eng") -> list[str]:
    """OCR opcional. Requiere el binario de Tesseract, no solo el paquete de Python."""
    import pytesseract
    from PIL import Image

    with Image.open(path) as image:
        # Las imagenes web rondan las 96 ppp; Tesseract espera ~300. Escalar x3
        # mejora el resultado mas que cualquier otro ajuste.
        scaled = image.convert("L").resize((image.width * 3, image.height * 3))
        text = pytesseract.image_to_string(scaled, lang=lang, config="--psm 6")
    return text.splitlines()
```

Tres detalles que explican el 90 % de los malos resultados de OCR:

- **`resize`**: subir la resolución antes de reconocer. Es el ajuste de mayor impacto.
- **`convert("L")`**: pasar a escala de grises antes de que el motor binarice.
- **`--psm 6`**: "trata esto como un bloque uniforme de texto". El modo por defecto asume una
  página completa con columnas y destroza los gráficos.

El `import` va **dentro** de la función a propósito: así el módulo se puede importar aunque
Pillow no esté instalado, y quien no use OCR no necesita la dependencia.

### 7.5 El AVIF

Un solo archivo, y es un retrato sin texto. Pillow no abre AVIF sin `pillow-avif-plugin`.
Como no tiene texto, **no hace falta abrirlo**: cae en la rama de bloque mínimo y todo
funciona. Instalar una dependencia por un archivo del que no vas a extraer nada es coste puro.

---

## 8. Verificar antes de dar por terminado

### 8.1 Correrlo

```bash
uv run python -m src.extract --formato jpg avif
```

Lo que tiene que salir: número de procesados, cero fallidos, y un total de palabras que
tenga sentido con lo que sabes del formato.

### 8.2 Comprobar los invariantes

Este script vale para cualquier parser. Es el mínimo antes de commitear:

```python
from src.extract import load_catalog, extract_all

entries = [e for e in load_catalog() if e.formato in ("jpg", "avif")]
docs = list(extract_all(entries))

assert len(docs) == len(entries), "se perdio algun documento"
assert all(d.blocks for d in docs), "hay documentos sin bloques"
assert all(b.strip() for d in docs for b in d.blocks), "hay bloques vacios"
assert len({d.doc_id for d in docs}) == len(docs), "doc_id duplicado"
assert all(d.fenomeno in (1, 2, 3) for d in docs), "fenomeno invalido"

print(len(docs), "docs |", sum(len(" ".join(d.blocks).split()) for d in docs), "palabras")
```

### 8.3 Leerlo con los ojos

Los `assert` te dicen que hay texto. Solo mirarlo te dice si el texto **sirve**:

```python
for doc in docs[:5]:
    print(doc.doc_id, "|", len(doc.blocks), "bloques |", doc.blocks[0][:200])
```

Pregúntate: ¿esto se parece a algo que alguien preguntaría en español? Si el bloque es
`"1807 5 4.3 years"`, el encoder no lo va a acercar a ninguna consulta y acabas de añadir
ruido al índice.

### 8.4 Comparar contra la corrida anterior

Cuando **modifiques** un parser existente, guarda el volcado antes y compara después. Así
sabes exactamente qué cambió y no descubres una regresión el día de la entrega:

```bash
uv run python -m src.extract --formato json -o data/interim/antes.jsonl
# ... cambios ...
uv run python -m src.extract --formato json -o data/interim/despues.jsonl
```

```python
import json

antes = {json.loads(l)["doc_id"]: json.loads(l) for l in open("data/interim/antes.jsonl", encoding="utf-8")}
despues = {json.loads(l)["doc_id"]: json.loads(l) for l in open("data/interim/despues.jsonl", encoding="utf-8")}

distintos = [k for k in antes if antes[k]["blocks"] != despues[k]["blocks"]]
print(f"{len(distintos)} de {len(antes)} documentos cambiaron")
```

### 8.5 Registrar la decisión

Un archivo en `docs/decisions/` siguiendo la plantilla de su `README.md`. Los números de §8.1
van en la sección de consecuencias. El informe técnico se escribe desde ahí.

---

## 9. Anti-patrones, con el contraejemplo al lado

**Inventar identificadores**

```python
doc_id = f"IMG-{index:03d}"                    # MAL: rompe el emparejamiento
doc_id = entry.doc_id                          # BIEN
```

**Construir `fuente` con el nombre del archivo**

```python
fuente = entry.path.name                       # MAL: 186 archivos comparten nombre
fuente = entry.fuente                          # BIEN: ruta relativa del indice
```

**Deducir el formato del contenido o de la carpeta**

```python
formato = "imagen"                             # MAL
formato = entry.formato                        # BIEN: extension real, en minusculas
```

**Devolver un documento vacío**

```python
return RawDoc(..., blocks=())                  # MAL: doc_id irrecuperable para siempre
return RawDoc(..., blocks=(minimo,))           # BIEN
```

**Meter la metadata dentro del cuerpo**

```python
blocks = [f"url: {url}", f"fecha: {fecha}", cuerpo]     # MAL: ensucia el texto indexado
extra = {"url": url, "date": fecha}                     # BIEN
```

La excepción legítima: cuando el documento casi no tiene cuerpo y la metadata **es** el
contenido. Es lo que hacemos con `alerta_meta` en las fichas de alerta. Criterio: *si sin esa
metadata el documento es irreconocible, la metadata es cuerpo.*

**Duplicar el mismo contenido desde dos campos**

```python
blocks = paragraphs + [body_text]              # MAL: duplica el documento y sesga la
blocks = paragraphs or split(body_text)        # BIEN: agregacion a nivel documento
```

**Reordenar o filtrar después**

Cualquier reordenamiento posterior rompe la correspondencia entre `metadata.jsonl` y los ids
internos de FAISS, y con ella toda la trazabilidad. El orden que produce el parser es el orden
que se indexa.

---

## 10. Documentación de referencia

> Los enlaces marcados con ✓ los verifiqué al escribir esta guía.

**Del propio repo — léelos, son la fuente de verdad**

- [`src/extract/core.py`](../../src/extract/core.py) — los contratos, `clean()` y el catálogo
- [`src/extract/json_docs.py`](../../src/extract/json_docs.py) — un parser real, 99 líneas
- [`src/extract/__init__.py`](../../src/extract/__init__.py) — `PARSERS` y `extract_all`
- [`CLAUDE.md`](../../CLAUDE.md) — reglas duras y convenciones
- [`.claude/reference/spec-etapa1.md`](../../.claude/reference/spec-etapa1.md) — la §2 manda
  sobre preprocesamiento; la Tabla 1 sobre metadata

**Python**

- [dataclasses](https://docs.python.org/3/library/dataclasses.html) — qué hacen `frozen` y
  `slots`, y por qué los usamos
- [pathlib](https://docs.python.org/3/library/pathlib.html) — `Path` en vez de concatenar
  cadenas: es lo que hace que el código funcione igual en Windows y en Linux
- [Logging HOWTO](https://docs.python.org/3/howto/logging.html) — niveles y por qué `%s`
- [typing](https://docs.python.org/3/library/typing.html) y
  [`from __future__ import annotations`](https://docs.python.org/3/library/__future__.html)

**Herramientas**

- [Documentación de ruff](https://docs.astral.sh/ruff/) — el linter y formateador del repo
- [Documentación de uv](https://docs.astral.sh/uv/) — grupos de dependencias y `uv export`
- [Pillow](https://pillow.readthedocs.io/en/stable/) ·
  [ExifTags](https://pillow.readthedocs.io/en/stable/reference/ExifTags.html) ·
  [pillow-avif-plugin](https://pypi.org/project/pillow-avif-plugin/)
- ✓ [Tesseract — documentación](https://tesseract-ocr.github.io/tessdoc/) ·
  ✓ [Improving the quality of the output](https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html)
  — la página de los modos `--psm` y la resolución
- [pytesseract](https://pypi.org/project/pytesseract/)
- [pySBD](https://github.com/nipunsadvilkar/pySBD) — segmentación de oraciones, para la fase
  de chunking

**Conceptos**

- ✓ [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/), cap. 2 — por
  qué el preprocesamiento decide la calidad de la búsqueda
- [Unicode UAX #15](https://unicode.org/reports/tr15/) — qué es NFC y por qué normalizamos
- [A Philosophy of Software Design](https://web.stanford.edu/~ouster/cgi-bin/book.php) —
  "módulos profundos": interfaz mínima, complejidad escondida. Es exactamente la forma de un
  parser: una función pública, todo el desorden dentro.
