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

**Pendiente de recuperación.** No hay encoder, ni índice, ni conjunto de consultas de desarrollo:
ninguna variante de chunking se puede puntuar todavía. El chunking implementado (ADR-007) trae un
perfil **estructural** (número de chunks, palabras por chunk, concentración documental,
conservación de contenido) que se regenera con:

```bash
uv run --extra cpu python -m src.chunking --profile <ruta>
```

Ese perfil **no es** NDCG@10 ni F1@3 y no se convierte en una fila de esta tabla. Las hipótesis
E1–E6 de `docs/research/chunking-best-practices-codefest.md` §19 entran aquí cuando exista con qué
medirlas.

| Fecha | Variante | Qué cambia vs. baseline | Proxy | Frag. | Doc. | Δ frag. | Δ doc. | Decisión | Responsable |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
