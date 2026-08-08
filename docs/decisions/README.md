# Registro de decisiones (ADR)

Una decisión de peso = un archivo `00X-titulo-corto.md`. El informe técnico (≤ 8 páginas) se escribe desde aquí: si las decisiones no están registradas, el 7 de agosto habrá que reconstruirlas de memoria y se pierden puntos en el criterio de documentación.

Registrar como mínimo: elección de encoder(s), estrategia de chunking, tipo de índice FAISS, estrategia de fusión y agregación a documento, y política ante archivos faltantes o corruptos.

## Plantilla

```markdown
# ADR-00X: <título>

- **Fecha**: AAAA-MM-DD
- **Estado**: propuesta | aceptada | reemplazada por ADR-00Y
- **Responsable**:

## Contexto

Qué problema obliga a decidir. Restricciones aplicables (licencia, límite de tokens del
encoder, CPU en la máquina del evaluador, Python 3.9, prohibición de decoders).

## Opciones consideradas

| Opción | A favor | En contra | Licencia | Costo |
|---|---|---|---|---|

## Decisión

Cuál y por qué. Enlazar la fila de `docs/ablaciones.md` que la sustenta. Si se decidió sin
medición, decirlo explícitamente y marcar la deuda.

## Consecuencias

- Qué se gana.
- Qué se pierde o qué queda pendiente de verificar.
- Qué habría que revisar si esta decisión resulta equivocada.
```
