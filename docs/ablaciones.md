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

### V5.1 — materialización productiva sobre C2/C5 (2026-08-13)

Cierra el hueco que V5 dejó abierto: se materializa el `text` con cinco políticas que **no ven el
gold**, sobre el mismo ranking BGE y los mismos índices de V5 (nada se reconstruye). Se añade el
merge consciente del solapamiento para C5 (dedup de la unidad repetida en la frontera, solo
igualdad exacta) y una política nueva, `best_bge_similarity_adjacent_if_fits` (M4), que elige
vecino por similitud BGE consulta↔vecino en vez de por rank.

- **Proxy fragmento**: `ProxyNDCG@10` sobre el texto **materializado** y `EvR@100` micro.
- **Proxy documento**: `F1@3` macro. No depende de la materialización (verificado).
- Regresión: las métricas documentales por consulta reproducen V5 exactamente (18/18).

| Fecha | Variante | Qué cambia vs. baseline | Proxy | Frag. | Doc. | Δ frag. | Δ doc. | Decisión | Responsable |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-13 | C2 + `raw` | chunking 120 | ProxyNDCG@10 / F1@3 | 0,0000 | 0,1542 | −0,1420 | +0,0292 | descartada: representabilidad raw 0/15 | Daniela Castaño |
| 2026-08-13 | C2 + `next_if_fits` | chunking 120 + materialización | ProxyNDCG@10 / F1@3 | 0,0625 | 0,1542 | −0,0795 | +0,0292 | **máx. cobertura**: EvR@100 7/15, captura 100 % del oráculo | Daniela Castaño |
| 2026-08-13 | C2 + `best_bge_similarity` | chunking 120 + M4 | ProxyNDCG@10 / F1@3 | 0,0878 | 0,1542 | −0,0542 | +0,0292 | mejor orden que M2 pero solo 4/15 | Daniela Castaño |
| 2026-08-13 | C5 + `previous_if_fits` | chunking 120 + overlap | ProxyNDCG@10 / F1@3 | 0,0505 | 0,1958 | −0,0915 | +0,0708 | 6/15 | Daniela Castaño |
| 2026-08-13 | C5 + `best_bge_similarity` | chunking 120 + overlap + M4 | ProxyNDCG@10 / F1@3 | **0,1294** | **0,1958** | −0,0126 | +0,0708 | **mejor en las dos métricas puntuadas** | Daniela Castaño |

Baseline C0 (BGE, V2/V3): `ProxyNDCG@10` = 0,1420 y `F1@3` = 0,1250.

**Tensión que la decisión automática no resuelve.** La regla declarada en el prompt elige por
`EvR@100` y devuelve `RECOMMEND_C2` (7 evidencias frente a 6). Pero `EvR@100` es un proxy de
disponibilidad de candidatos, **no** una métrica del leaderboard. En las dos que sí puntúan, C5
gana: `ProxyNDCG@10` 0,1294 vs 0,0625 y `F1@3` 0,1958 vs 0,1542 (además `Hit@3` 0,500 vs 0,375 y
`MRR` 0,3057 vs 0,2797). Y **ninguna configuración recupera todavía el `ProxyNDCG@10` del
baseline** (0,1420). Ver `CLAUDE.md` §5 sobre Borda: la decisión final es del equipo, no del
criterio automático. Artefactos: `data/interim/chunking_benchmark_v5_1/`.

**`decision.json` no se modifica** — es el resultado correcto del criterio que V5.1 declaró, y
reescribirlo falsearía la historia de esa fase. La decisión de equipo posterior está en
`docs/decisions/008-promocion-chunking-format-aware-v2.md`.

### Promoción a producción — ADR-008 (2026-08-13)

Por Borda sobre las dos métricas del leaderboard (`ProxyNDCG@10` y `F1@3`, ambas favorables a
C5 en la tabla de arriba), el equipo adopta **C5 como `format_aware_v2`**, no `RECOMMEND_C2`.
Config productiva: `FORMAT_AWARE_V2_CONFIG` en `src/chunking/core.py`
(`target=120/soft_min=72/max=250/overlap=1`, huella `f2c665528a008aa9` — idéntica a
`c5_smaller_120_overlap`). Materialización elegida para producción:
`best_bge_similarity_adjacent_if_fits` (M4), sin reimplementarla. Detalle completo, impacto en
conteo de chunks (1,90x) y duplicación (1,26x), y limitaciones del devset: ver el ADR. No se
construyó ningún índice FAISS ni se ejecutaron embeddings en esta fase; el artefacto productivo
es `data/interim/chunking/format_aware_v2.jsonl` con su manifest de procedencia.
