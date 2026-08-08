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
| Chunking | |
| Encoder(s) | |
| Índice | |
| Fusión | |
| Agregación a documento | |
| Proxy fragmento | |
| Proxy documento | |

## Registro

| Fecha | Variante | Qué cambia vs. baseline | Proxy | Frag. | Doc. | Δ frag. | Δ doc. | Decisión | Responsable |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |
