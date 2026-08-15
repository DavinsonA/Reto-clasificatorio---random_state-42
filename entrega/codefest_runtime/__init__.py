"""Runtime de recuperacion de la entrega CODEFEST Ad Astra 2026 — equipo `random_state = 42`.

Paquete AUTOCONTENIDO: no importa nada del repositorio de desarrollo (`src/`, `data/`, `tests/`).
Copiar la carpeta `entrega/` a cualquier ubicacion y ejecutar `python generador.py` debe bastar.

Modulos, en el orden en que los usa el pipeline:

    config          constantes congeladas de la arquitectura
    queries         carga y validacion de `consultas.jsonl`
    encoder         BGE-M3 con la configuracion exacta que construyo el indice
    index_store     carga de `index.faiss` + `metadata.jsonl` y sus lecturas
    materialization M4: que texto se entrega para un chunk recuperado
    textseg         segmentacion linguistica de salida (pysbd), sin cortar oraciones
    normalization   normalizacion oficial a <= 250 palabras
    preflight       verificacion de la base vectorial antes de recuperar nada
    pipeline        orquestacion y contratos de salida

Los imports son perezosos: `import codefest_runtime` no carga torch, FAISS ni el modelo, para que
`python generador.py --help` responda al instante.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
