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
