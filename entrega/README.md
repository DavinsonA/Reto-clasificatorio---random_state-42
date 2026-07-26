# CODEFEST Ad Astra 2026 — Etapa 1

> Equipo `random_state = 42`

Base de conocimiento vectorial para recuperación de documentos y fragmentos.
Este directorio contiene la entrega completa.

## Contenido

| Archivo | Descripción |
| --- | --- |
| `resultados.jsonl` | 50 líneas (`q001`–`q050`), una por consulta |
| `generador.py` | reproduce `resultados.jsonl` a partir del índice |
| `informe_tecnico.pdf` | decisiones de diseño |
| `base_vectorial/` | índices FAISS y metadata, una subcarpeta por encoder |
| `requirements.txt` | dependencias fijadas |

## Reproducir los resultados

Requiere Python 3.11 o superior. **No requiere GPU.**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python generador.py
```

Esto regenera `resultados.jsonl` en este mismo directorio.

## Notas

- **La primera ejecución descarga el encoder desde HuggingFace** (varios cientos
  de MB) y puede tardar unos minutos sin mostrar progreso. Las siguientes usan la
  caché local y son inmediatas.
- `requirements.txt` incluye el índice de PyTorch además de PyPI, para instalar la
  rueda de torch sin CUDA. Todas las versiones están fijadas con `==`.
- El índice se carga con `faiss.read_index()`, sin dependencias adicionales.
