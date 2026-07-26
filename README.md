# CODEFEST Ad Astra 2026 — Etapa 1

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
entrega/
├── resultados.jsonl          # 50 líneas: 3 documentos + 10 fragmentos por consulta
├── generador.py              # reproduce resultados.jsonl a partir del índice
├── informe_tecnico.pdf       # decisiones de diseño (máx. 8 páginas)
└── base_vectorial/
    ├── encoder_1/            # index.faiss + metadata.jsonl
    ├── encoder_2/            # index.faiss + metadata.jsonl
    └── grafo/                # grafo.graphml (bonus)
```

## Pipeline

Extracción y limpieza → chunking → embeddings → índice FAISS → recuperación.

## Uso

```bash
pip install -r requirements.txt
python entrega/generador.py
```

## Evaluación

| Nivel      | Métrica  | Salida             |
| ---------- | -------- | ------------------ |
| Fragmento  | NDCG@10  | 10 chunks ≤ 250 palabras |
| Documento  | F1@3     | 3 `doc_id`         |
