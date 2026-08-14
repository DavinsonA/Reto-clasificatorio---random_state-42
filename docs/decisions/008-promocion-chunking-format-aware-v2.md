# ADR-008: Promoción de C5 a `format_aware_v2`

- **Fecha**: 2026-08-13
- **Estado**: aceptada
- **Responsable**: Daniela Castaño (RAG & LLM Engineer)

## Contexto

`format_aware_v1` (ADR-007, `target_words=200 / soft_min=120 / max=250`, `overlap_units=0`) fue
el baseline de implementación: sin medición de recuperación, porque en ese momento no existía
pipeline encoder → embeddings → FAISS → recuperación contra el que compararlo.

Ese pipeline ya existe (V2/V3) y una vez existió se pudo diagnosticar y medir el chunking:

- **V4** (`docs/ablaciones.md`, `f76cf7c`) midió que el techo de representación del baseline es
  **4/15** evidencias del devset: las 11 restantes son irrepresentables, no mal rankeadas — en
  ninguna de ellas un par de chunks adyacentes cabe en las 250 palabras que permite entregar
  `§9.2.1`. Causa: el chunk medio del baseline ocupa 177,9 de esas 250 palabras.
- **V5** (`681de79`) barrió seis variantes de granularidad y solapamiento. Con `target_words=120`
  el techo sube a 11/15 (C2, sin overlap) y a **12/15** con `overlap_units=1` (C5). Midiendo
  materialización productiva con BGE (Etapa B): `ProxyNDCG@10` cae a 0 en ambas porque la
  representabilidad *cruda* es 0/15 por construcción (ningún chunk de ~120 palabras contiene una
  evidencia entera) — la calidad de fragmento exige materializar con `previous/next_if_fits`, que
  V5 no ejecutó todavía.
- **V5.1** (`2b635ba`) cerró ese hueco: materializó las cinco políticas que no ven el gold sobre
  los mismos índices de V5 (sin reconstruir nada), añadió el merge consciente del solapamiento
  para C5 (dedup por igualdad exacta en la unidad repetida) y una política nueva,
  `best_bge_similarity_adjacent_if_fits` (M4). Resultado relevante para esta promoción:

  | | C2 | C5 |
  |---|---|---|
  | Oracle@100 (techo productivo) | 7/15 | **9/15** |
  | Mejor política productiva | `next_if_fits` | `best_bge_similarity_adjacent_if_fits` |
  | `ProxyNDCG@10` | 0,0625 | **0,1294** |
  | `F1@3` | 0,1542 | **0,1958** |
  | `Hit@3` / `MRR` | 0,375 / 0,2797 | **0,500 / 0,3057** |
  | Capture ratio@100 (vs. oráculo) | 1,000 | 0,667 |

## Decisión

**Se promueve C5 (`c5_smaller_120_overlap`) al candidato productivo `format_aware_v2`.**

```text
target_words        = 120
soft_min_words       = 72
max_words            = 250
overlap_units        = 1
cross_block_packing  = True   # sin cambios respecto a format_aware_v1
output_target_words  = 240
output_max_words     = 250
```

Config congelada en `FORMAT_AWARE_V2_CONFIG` (`src/chunking/core.py`), con la misma huella
(`config_fingerprint`) que ya tiene commiteada `c5_smaller_120_overlap` en
`data/interim/chunking_benchmark_v5/variant_configs.json`
(`f2c665528a008aa9`) — es la MISMA config, no una aproximación.

### Por qué el `decision.json` de V5.1 dice `RECOMMEND_C2` y aquí se adopta C5

`decision.json` (V5.1, `data/interim/chunking_benchmark_v5_1/`) aplica el criterio declarado en
esa fase — elegir por `EvR@100` (recall de candidatos disponibles) — y ese criterio produce
`RECOMMEND_C2` porque C2 captura el 100 % de su propio techo (7/7) mientras C5 captura 6/9 con
la mejor política encontrada. **`EvR@100` es un proxy de disponibilidad de candidatos, no una
métrica del leaderboard.** En las dos métricas que sí puntúan (`CLAUDE.md` §5), C5 gana con
margen: `ProxyNDCG@10` más del doble (0,1294 vs 0,0625) y `F1@3` superior (0,1958 vs 0,1542),
además de `Hit@3` y `MRR`. Por Borda sobre las dos tablas del leaderboard, C5 queda por delante.
Esa tensión ya está documentada en `docs/ablaciones.md` (sección V5.1) y es la base de esta
decisión — una decisión de equipo, no del criterio automático de V5.1, tal como esa misma fase
advertía que correspondía (§40 de su prompt: *"no usar un score compuesto arbitrario"*).

**`decision.json` no se modifica.** Es el resultado honesto de un criterio explícito aplicado
correctamente; cambiarlo sería reescribir la historia de V5.1. Esta promoción es una decisión
posterior y separada, registrada aquí.

### Materialización: M4 se elige, no se reimplementa

La política de materialización elegida para producción es
**`best_bge_similarity_adjacent_if_fits` (M4)**: sobre C5 iguala el recall de la mejor política
simple (`previous_if_fits`, EvR@100 = 0,4000) con más del doble de `ProxyNDCG@10` (0,1294 vs
0,0505). M4 ya existe en `src/retrieval/productive_materialization.py` (V5.1) — esta fase de
promoción **no la reimplementa ni la reconecta**: solo dejaba constancia de cuál se usará cuando
se enganche `generador.py`, tarea de una fase posterior (§ Próxima fase).

## Impacto medido en el chunking (corpus completo, `data/interim/chunking_benchmark_v5/`)

| | C0 (`format_aware_v1`) | C5 (`format_aware_v2`) |
|---|---|---|
| Chunks | 171.780 | 326.866 (**1,90x**) |
| Palabras medias por chunk | 177,9 | 117,9 |
| `duplication_ratio` (palabras de chunk / palabras de fuente) | 1,00 | **1,26** — esperado por `overlap_units=1`, no una fuga |
| Techo de representación (15 evidencias) | 4/15 | 12/15 |

El aumento de 1,90x en vectores y el 26 % de palabras duplicadas son el costo medido y aceptado
de subir el techo de representación de 4/15 a 12/15: más vectores en el índice FAISS y más
tiempo de embedding en la fase siguiente.

## Limitaciones del devset (heredadas de V4/V5/V5.1, no resueltas aquí)

El devset interno (`data/interim/benchmarks/prechunk/devset.jsonl`) tiene **9 consultas, 8 con
gold, 15 fragmentos, 11 documentos, todos PDF**, fenómenos 1 y 3, **sin cobertura de F2**. Todos
los números de esta promoción — techo de representación, `ProxyNDCG@10`, `F1@3` — son proxies
sobre esa muestra diminuta y no cubren tabular (CSV/XLSX/PBF) ni imágenes. Ninguna cifra aquí es
la métrica oficial del comité.

## Qué NO decide este ADR

Encoder (sigue BGE-M3, sin cambios), tipo de índice, fusión multi-encoder, reranking, agregación
a documento, ni la conexión de M4 a `generador.py`. `format_aware_v1.jsonl` y los índices FAISS
vigentes **no se tocan**: esta promoción solo materializa
`data/interim/chunking/format_aware_v2.jsonl` y su manifest de procedencia
(`format_aware_v2.manifest.json`) como artefacto nuevo, en paralelo.

## Próxima fase

Reconstruir los índices FAISS definitivos (BGE y GTE) consumiendo exclusivamente
`format_aware_v2.jsonl`, reintegrar RRF y el reranker sobre ese universo de chunks, y enganchar
M4 a `generador.py`. Ninguno de esos pasos se ejecuta en esta fase.

## Consecuencias

**Qué se gana**

- El chunking productivo pasa de un techo de representación medido de 4/15 a 12/15, con ganancia
  neta en las dos métricas del leaderboard (`ProxyNDCG@10`, `F1@3`) frente al baseline vigente en
  V5.1, aunque `ProxyNDCG@10` de C5 (0,1294) todavía no alcanza el de `format_aware_v1` puro
  (0,1420) — ver la advertencia de V5 en `docs/ablaciones.md` sobre por qué esa comparación cruda
  no es la que importa una vez que se materializa con vecinos.
- Config con identidad productiva propia (`FORMAT_AWARE_V2_CONFIG`), sin que `src/retrieval/` ni
  el futuro `generador.py` dependan de `src.chunking.ablation.VARIANTS`.
- Manifest de procedencia reproducible: hash del artefacto, huella de config, hashes de cada
  entrada, revisión de git y estado del working tree.

**Qué se pierde o queda pendiente**

- 1,90x más vectores que el baseline: costo de embedding e índice en la fase siguiente, no
  medido todavía en tiempo de reloj.
- La comparación C2 vs. C5 se decidió con Borda sobre dos proxies de un devset de 15 evidencias;
  con más datos de desarrollo la decisión podría revertirse.
- M4 depende de similitud BGE consulta↔vecino calculada en V5.1 contra el índice de esa fase; al
  reconstruir el índice definitivo sobre `format_aware_v2.jsonl` completo (no solo los documentos
  gold) hay que revalidar que M4 se comporta igual, no asumirlo.

**Qué habría que revisar si esta decisión resulta equivocada**

Si al reconstruir los índices definitivos `ProxyNDCG@10` de C5 no logra superar al baseline en el
devset ampliado, o si el fenómeno 2 (sin cobertura en el devset actual) se comporta distinto, la
vuelta atrás es barata: `format_aware_v1.jsonl` no se tocó y `DEFAULT_CONFIG` sigue siendo
`200/120/250/overlap=0` sin cambios.
