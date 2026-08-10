# CODEFEST Ad Astra 2026 — Desafio Clasificatorio

> Equipo `random_state = 42`

Base de conocimiento vectorial (FAISS + encoders de HuggingFace) para recuperar
documentos y fragmentos relevantes ante 50 consultas en lenguaje natural.

## Integrantes

- Davinson Arteaga
- Daniela Castaño
- Gian Mendoza
- Juan Villegas

## Estructura

```
├── src/                      # pipeline
├── data/                     # corpus crudo
├── notebooks/                # exploración con Jupyter
├── entrega/                  # ARTEFACTO: lo que se empaqueta y se entrega
│   ├── resultados.jsonl      #   50 líneas: 3 documentos + 10 fragmentos por consulta
│   ├── generador.py          #   reproduce resultados.jsonl a partir del índice
│   ├── informe_tecnico.pdf   #   decisiones de diseño (máx. 8 páginas)
│   ├── requirements.txt      #   GENERADO — ver "Empaquetar la entrega"
│   ├── README.md             #   instrucciones para el evaluador
│   └── base_vectorial/
│       ├── encoder_1/        #   index.faiss + metadata.jsonl
│       ├── encoder_2/        #   index.faiss + metadata.jsonl
│       └── grafo/            #   grafo.graphml (bonus)
├── pyproject.toml            # dependencias
└── uv.lock
```

`entrega/` es la **salida del pipeline**, basicamente lo que vera el evaluador. 

Lo trabajado vive en `src/`; `src/indexar.py` escribe dentro de `entrega/base_vectorial/`.

## Pipeline

Extracción y limpieza → chunking → embeddings → índice FAISS → recuperación.

## Entorno

Gestionado con [uv](https://docs.astral.sh/uv/). Requiere Python 3.11 o superior.

Para usar la versión del enviroment usando CUDA:

```powershell
uv sync --extra gpu
```

Verificar que la GPU quedó activa:

```powershell
uv run --extra gpu python -c "import torch; print(torch.cuda.is_available())"
```

Para trabajar sin GPU usar `uv sync --extra cpu` en su lugar.

> **Usar siempre `--extra gpu` también en `uv run`.** `uv run` sincroniza el
> entorno antes de ejecutar; sin el flag te revierte torch a la versión de CPU.

## OCR opcional (Tesseract)

Los parsers de PDF e imagenes (`src/extract/pdf_docs.py`, `src/extract/images.py`) pueden usar
OCR clasico (Tesseract, no generativo — cumple la prohibicion de decoders de `CLAUDE.md` §2.1)
para paginas o imagenes sin texto extraible. Esta **apagado por defecto**; se activa con:

```powershell
$env:PDF_OCR = "1"
uv run --extra gpu python -m src.extract --formato pdf
```

```bash
PDF_OCR=1 uv run --extra gpu python -m src.extract --formato pdf
```

`pillow` y `pytesseract` ya estan en `[dependency-groups] dev` (build-time, nunca llegan a
`generador.py`), pero **Tesseract es ademas un binario del sistema operativo** que `pip` no
instala. Sin el binario, el pipeline sigue corriendo — cada pagina/imagen conserva su texto
nativo o cae al bloque minimo, nunca falla (ver `docs/decisions/003-parser-pdf.md`).

Probado en este repo (2026-08-09, ver `docs/decisions/003-parser-pdf.md`): las 3 muestras de
PDF escaneado pasaron de un bloque minimo de 4 palabras a cientos o miles de palabras reales.

Para instalar el binario y los paquetes de idioma (español, inglés, portugués — los tres del
corpus):

**Windows — via winget (recomendado, sin instalador manual):**

```powershell
winget install --id tesseract-ocr.tesseract
```

El paquete de winget solo trae el idioma ingles. **Si no tienes permisos de administrador**
para escribir en `C:\Program Files\Tesseract-OCR\tessdata\` (el caso mas comun en un equipo
compartido), no lo intentes con `sudo`/elevacion: usa una carpeta propia y la variable de
entorno `TESSDATA_PREFIX`, que Tesseract respeta sin tocar la instalacion del sistema:

```powershell
mkdir $env:LOCALAPPDATA\tessdata
Copy-Item "C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata" $env:LOCALAPPDATA\tessdata\
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata_fast/raw/main/spa.traineddata" -OutFile "$env:LOCALAPPDATA\tessdata\spa.traineddata"
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata_fast/raw/main/por.traineddata" -OutFile "$env:LOCALAPPDATA\tessdata\por.traineddata"

$env:PATH = "C:\Program Files\Tesseract-OCR;" + $env:PATH
$env:TESSDATA_PREFIX = "$env:LOCALAPPDATA\tessdata"
tesseract --list-langs   # debe listar eng, spa, por
```

Alternativa con instalador grafico (trae los 3 idiomas marcables en el propio instalador, pero
si requiere permisos de administrador):
[UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

**Linux (Debian/Ubuntu):**

```bash
sudo apt install tesseract-ocr tesseract-ocr-spa tesseract-ocr-eng tesseract-ocr-por
```

En todos los casos, `PATH`/`TESSDATA_PREFIX` son variables de entorno de cada maquina, nunca
rutas escritas en el código — `src/extract/images.py` y `src/extract/pdf_docs.py` solo llaman
`tesseract` por nombre.

## Dependencias

| Dónde | Descripción | Usado por evaluador |
| --- | --- | --- |
| `[project.dependencies]` | usado en `generador.py` | **sí** |
| `[dependency-groups] dev` | extracción, chunking, notebooks | no |
| `[project.optional-dependencies]` | variante de torch (`cpu` / `gpu`) | no |


Al añadir una dependencia, considerar si su uso también se dara en `generador.py`. Si no, va al
grupo `dev`:

```powershell
uv add --group dev <paquete>
```

## Uso

```powershell
uv run --extra gpu python entrega/generador.py
```

## Empaquetar la entrega

`entrega/requirements.txt` es un **archivo generado**: no se edita a mano. Se
regenera antes de entregar, y cada vez que cambien las dependencias de runtime:

```powershell
uv export --extra cpu --no-dev --no-hashes --no-emit-project --emit-index-url -o entrega/requirements.txt
```

- `--extra cpu` fija `torch==2.13.0+cpu`, para no descargar librerias pesadas de CUDA.
- `--no-dev` deja fuera `pymupdf`, `pandas`, Jupyter y demás herramientas inncesarias para el script.
- `--emit-index-url` añade el índice de PyTorch, sin el cual `pip` no encuentra la rueda `+cpu`.

## Evaluación

| Nivel      | Métrica  | Salida             |
| ---------- | -------- | ------------------ |
| Fragmento  | NDCG@10  | 10 chunks ≤ 250 palabras |
| Documento  | F1@3     | 3 `doc_id`         |
