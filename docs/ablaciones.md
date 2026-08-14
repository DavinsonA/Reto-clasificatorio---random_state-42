# Bitácora de ablaciones

Toda variante del pipeline se registra aquí antes de adoptarse. Sin fila, no entra.

**Reglas** (ver `CLAUDE.md` §4.6 y el subagente `evaluador`):

- Una sola variable cambia por fila. Mismo set de consultas, mismo corpus, misma semilla.
- Se reportan **las dos** métricas proxy: nivel fragmento y nivel documento. El leaderboard es Borda sobre ambas; una mejora que sube una y hunde la otra puede empeorar el puesto final.
- No hay ground truth. Todo valor aquí es un proxy y puede no correlacionar con la métrica oficial. Anotar qué proxy se usó.
- Delta dentro del ruido del proxy → **descartar**. Cada componente extra es riesgo de fallo en la máquina del evaluador.

## Baseline vigente

| Campo | Valor |
|---|---|
| Fecha | _pendiente_ |
| Extracción | |
| Chunking | format-aware, `target=200 / soft_min=120 / max=250`, overlap 0, salida `240/250` (ADR-007). **Sin medir**: adoptado por el perfil estructural del corpus, no por métrica de recuperación |
| Encoder(s) | |
| Índice | |
| Fusión | |
| Agregación a documento | |
| Proxy fragmento | |
| Proxy documento | |

## Registro

**Pendiente de recuperación.** Existe un development set interno
(`data/interim/benchmarks/prechunk/devset.jsonl`: 9 consultas, 8 con gold, 15 fragmentos, 11
documentos, todos PDF, F1 y F3, **sin F2**; limitaciones en research §4.1), pero **no hay encoder,
ni índice, ni pipeline de recuperación**, así que ninguna variante de chunking se puede puntuar
todavía. El chunking implementado (ADR-007) trae un
perfil **estructural** (número de chunks, palabras por chunk, concentración documental,
conservación de contenido) que se regenera con:

```bash
uv run --extra cpu python -m src.chunking --profile <ruta>
```

Ese perfil **no es** NDCG@10 ni F1@3 y no se convierte en una fila de esta tabla. Las hipótesis
E1–E6 de `docs/research/chunking-best-practices-codefest.md` §19 entran aquí cuando exista con qué
medirlas.

### V5 — ablación de chunking (2026-08-13)

Ya existe pipeline de recuperación, así que las hipótesis E1–E3 del research por fin se pueden
puntuar. Contexto: V4 midió que el **techo de representación** del chunking vigente es 4/15 y que
las 11 evidencias perdidas son irrepresentables, no mal rankeadas — en los 11 casos ningún par
adyacente cabe en 250 palabras (chunk medio 177,9 palabras, pair-fit 0,31 %).

- **Proxy fragmento**: `representation-aware evidence recall@100` (micro, 15 unidades) — el único
  comparable contra el techo de representación, que también es micro.
- **Proxy documento**: `F1@3` (macro por consulta).
- Etapa A (6 variantes) sin embeddings; Etapa B (2 finalistas) con BGE-M3 `5617a9f6`, fp16,
  `IndexFlatIP`, misma configuración que el índice vigente. Solo cambia el chunking.
- **C0 reproduce `format_aware_v1.jsonl` bitwise** (sha256 idéntico): la comparación tiene
  referencia verificada.

| Fecha | Variante | Qué cambia vs. baseline | Proxy | Frag. | Doc. | Δ frag. | Δ doc. | Decisión | Responsable |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-13 | C1 `target=160` | solo granularidad | techo repr. (Etapa A) | 1/15 | — | −3 ev. | — | **descartada**: regresión; zona muerta entre "cabe en un chunk" y "cabe el par" | Daniela Castaño |
| 2026-08-13 | C3 `overlap=1 unidad` | solo solapamiento (E3) | techo repr. (Etapa A) | 6/15 | — | +2 ev. | — | **no finalista**: gana poco y a 1,28x chunks + 1,28x texto duplicado | Daniela Castaño |
| 2026-08-13 | C4 `target=160 + overlap=1` | ambas | techo repr. (Etapa A) | 4/15 | — | 0 ev. | — | **descartada**: sin ganancia, 1,60x chunks | Daniela Castaño |
| 2026-08-13 | C2 `target=120` | solo granularidad | ReprAware R@100 / F1@3 | 0,4667 | 0,1542 | +0,2667 | +0,0292 | **candidata**: techo 11/15, 1,53x chunks, sin duplicación | Daniela Castaño |
| 2026-08-13 | C5 `target=120 + overlap=1` | ambas | ReprAware R@100 / F1@3 | 0,6000 | 0,1958 | +0,4000 | +0,0708 | **candidata preferida por cobertura**: techo 12/15, pero 1,90x chunks y 1,26x texto | Daniela Castaño |

Baseline de referencia para las dos últimas filas: BGE-M3 sobre C0, ReprAware R@100 = 0,2000 y
F1@3 = 0,1250 (`data/interim/retrieval_benchmark_v2/metrics.json`,
`data/interim/retrieval_benchmark_v4/recall_saturation.json`).

**Advertencia sobre `ProxyNDCG@10` en esta ablación.** Cae de 0,1420 (C0) a **0,0000** en C2 y C5.
No es una degradación semántica: `ProxyNDCG@10` compara el texto **crudo** del chunk contra la
evidencia, y en C2/C5 la representabilidad raw es **0/15** por construcción (ninguna evidencia
cabe entera en un chunk de ~120 palabras). La métrica no puede ser distinta de cero ahí. Medir la
calidad de fragmento de estos chunkings exige materializar la salida con `previous/next_if_fits`
—que §9.2.1 permite y que estas variantes por fin hacen viable— y eso **no** se ejecutó en V5.
No comparar esas dos columnas entre chunkings con distinta representabilidad raw.

Ninguna variante se adopta todavía: falta la medición productiva de fragmento sobre el nuevo
universo de chunks. Artefactos: `data/interim/chunking_benchmark_v5/`.
