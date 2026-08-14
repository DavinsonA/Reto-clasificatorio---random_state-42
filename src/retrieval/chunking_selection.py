"""Pareto y seleccion de finalistas de la ablacion de chunking (V5, Etapa A -> Etapa B).

Elegir "el mayor representation ceiling" a secas seria un error: un chunking que duplica el
numero de vectores para ganar una evidencia no es obviamente mejor, y con 15 unidades de gold
ninguna diferencia pequena soporta una decision. Este modulo deja el trade-off explicito en vez
de comprimirlo en un score compuesto inventado (prompt V5 S17/S30/S46).

Tres ejes de decision:

- **beneficio**: `representation_ceiling` (cuantas evidencias pueden representarse);
- **coste de indice**: `chunk_count_ratio_vs_baseline` (cuantos vectores mas hay que embeber);
- **coste de duplicacion**: `duplication_ratio` (cuanto texto se repite en `metadata.jsonl`).

`pair_fit_rate` y `raw_representation_recall` se reportan y se usan para desempatar, pero NO
entran en la dominancia: son mecanismos que explican el beneficio, no beneficios independientes.
Meterlos como ejes haria "no dominada" a casi cualquier variante y la palabra dejaria de informar.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Ganancia minima, en unidades de evidencia, para considerar que una variante justifica pagar
# embeddings (prompt V5 S20). NO es un test estadistico: con 15 evidencias no existe uno honesto.
# Es un criterio de decision declarado: una sola evidencia entra dentro de lo que el ruido de un
# devset de este tamano puede mover, dos ya cambian la lectura del cuello de botella.
MATERIAL_GAIN_MIN_EVIDENCE = 2

# Por encima de este factor la variante se detiene ANTES de embeber (prompt V5 S54).
EXCESSIVE_CHUNK_INFLATION_RATIO = 3.0

MAX_FINALISTS = 2

NO_MEANINGFUL_REPRESENTATION_GAIN = "NO_MEANINGFUL_REPRESENTATION_GAIN"
EXCESSIVE_CHUNK_INFLATION = "EXCESSIVE_CHUNK_INFLATION"
GATE_PASSED = "GATE_PASSED"


@dataclass(frozen=True, slots=True)
class VariantScorecard:
    """Los numeros de una variante que participan en la decision, ya reunidos."""

    variant_id: str
    representable_count: int
    gold_total: int
    raw_representable_count: int
    neighbor_expansion_required_count: int
    chunk_count: int
    chunk_count_ratio: float
    pair_fit_rate: float
    duplication_ratio: float
    overlap_units: int

    @property
    def representation_ceiling(self) -> float:
        return self.representable_count / self.gold_total if self.gold_total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "representable_count": self.representable_count,
            "gold_evidence_total": self.gold_total,
            "representation_ceiling": self.representation_ceiling,
            "raw_representable_count": self.raw_representable_count,
            "neighbor_expansion_required_count": self.neighbor_expansion_required_count,
            "chunk_count": self.chunk_count,
            "chunk_count_ratio_vs_baseline": self.chunk_count_ratio,
            "pair_fit_rate": self.pair_fit_rate,
            "duplication_ratio": self.duplication_ratio,
            "overlap_units": self.overlap_units,
        }


def dominates(a: VariantScorecard, b: VariantScorecard) -> bool:
    """`a` domina a `b`: no es peor en ningun eje de decision y es mejor en al menos uno."""
    not_worse = (
        a.representable_count >= b.representable_count
        and a.chunk_count_ratio <= b.chunk_count_ratio
        and a.duplication_ratio <= b.duplication_ratio
    )
    strictly_better = (
        a.representable_count > b.representable_count
        or a.chunk_count_ratio < b.chunk_count_ratio
        or a.duplication_ratio < b.duplication_ratio
    )
    return not_worse and strictly_better


def _selection_key(card: VariantScorecard) -> tuple[Any, ...]:
    """Orden de preferencia del prompt V5 S18, en ese orden exacto.

    1. mas cobertura; 2. menos chunks; 3. menos overlap; 4. mas representabilidad raw;
    5. mayor pair-fit. El `variant_id` cierra la clave para que el desempate sea reproducible.
    """
    return (
        -card.representable_count,
        card.chunk_count_ratio,
        card.overlap_units,
        -card.raw_representable_count,
        -card.pair_fit_rate,
        card.variant_id,
    )


def build_pareto(
    cards: list[VariantScorecard], baseline_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Marca variantes dominadas y devuelve `(filas, ids_no_dominados)`.

    El baseline participa en la comparacion (es la referencia obligada) pero nunca se selecciona
    como finalista: reconstruirlo no aportaria informacion, sus metricas ya existen desde V4.
    """
    rows: list[dict[str, Any]] = []
    non_dominated: list[str] = []
    for card in cards:
        dominators = [
            other.variant_id for other in cards if other is not card and dominates(other, card)
        ]
        row = card.as_dict()
        row["pareto_dominated"] = bool(dominators)
        row["dominated_by"] = dominators
        row["is_baseline"] = card.variant_id == baseline_id
        rows.append(row)
        if not dominators and card.variant_id != baseline_id:
            non_dominated.append(card.variant_id)
    return rows, non_dominated


def select_finalists(
    cards: list[VariantScorecard],
    baseline_id: str,
    max_finalists: int = MAX_FINALISTS,
    min_gain: int = MATERIAL_GAIN_MIN_EVIDENCE,
) -> dict[str, Any]:
    """Aplica el gate del prompt V5 S20 y elige como maximo `max_finalists` variantes.

    Devuelve siempre la explicacion completa: por que entra cada finalista y por que se descarta
    cada una de las demas. Una seleccion sin motivo no es auditable.
    """
    by_id = {card.variant_id: card for card in cards}
    baseline = by_id.get(baseline_id)
    if baseline is None:
        raise ValueError(f"falta el baseline {baseline_id!r} entre las variantes evaluadas")

    candidates = [card for card in cards if card.variant_id != baseline_id]
    gains = {
        card.variant_id: card.representable_count - baseline.representable_count
        for card in candidates
    }
    best_gain = max(gains.values(), default=0)

    pareto_rows, non_dominated = build_pareto(cards, baseline_id)

    if best_gain < min_gain:
        return {
            "status": NO_MEANINGFUL_REPRESENTATION_GAIN,
            "baseline_representable_count": baseline.representable_count,
            "best_gain_evidence_units": best_gain,
            "min_gain_required": min_gain,
            "selected": [],
            "not_selected": [
                {
                    "variant_id": card.variant_id,
                    "reason": f"gain {gains[card.variant_id]:+d} evidencias sobre el baseline, "
                    f"por debajo del minimo declarado de {min_gain}",
                }
                for card in sorted(candidates, key=_selection_key)
            ],
            "pareto": pareto_rows,
            "note": (
                "Ninguna variante gana suficiente representacion para justificar construir "
                "embeddings. Etapa B no se ejecuta (prompt V5 S20)."
            ),
        }

    eligible = [
        card
        for card in candidates
        if gains[card.variant_id] >= min_gain and card.variant_id in non_dominated
    ]
    ordered = sorted(eligible, key=_selection_key)

    selected: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for card in ordered:
        if len(selected) >= max_finalists:
            break
        if card.chunk_count_ratio > EXCESSIVE_CHUNK_INFLATION_RATIO:
            blocked.append(
                {
                    "variant_id": card.variant_id,
                    "reason": EXCESSIVE_CHUNK_INFLATION,
                    "chunk_count_ratio_vs_baseline": card.chunk_count_ratio,
                    "limit": EXCESSIVE_CHUNK_INFLATION_RATIO,
                }
            )
            continue
        selected.append(
            {
                "variant_id": card.variant_id,
                "why_selected": (
                    f"+{gains[card.variant_id]} evidencias representables sobre el baseline "
                    f"({card.representable_count}/{card.gold_total}), "
                    f"{card.chunk_count_ratio:.2f}x chunks, "
                    f"duplicacion {card.duplication_ratio:.3f}x, "
                    f"pair-fit {100 * card.pair_fit_rate:.1f}%, "
                    f"overlap_units={card.overlap_units}; no dominada en el Pareto"
                ),
                **card.as_dict(),
            }
        )

    selected_ids = {entry["variant_id"] for entry in selected}
    not_selected: list[dict[str, Any]] = []
    for card in sorted(candidates, key=_selection_key):
        if card.variant_id in selected_ids:
            continue
        not_selected.append(
            {
                "variant_id": card.variant_id,
                "reason": _rejection_reason(card, gains, non_dominated, blocked, min_gain, by_id),
                **card.as_dict(),
            }
        )

    return {
        "status": GATE_PASSED,
        "baseline_representable_count": baseline.representable_count,
        "best_gain_evidence_units": best_gain,
        "min_gain_required": min_gain,
        "selected": selected,
        "not_selected": not_selected,
        "blocked": blocked,
        "pareto": pareto_rows,
    }


def _rejection_reason(
    card: VariantScorecard,
    gains: dict[str, int],
    non_dominated: list[str],
    blocked: list[dict[str, Any]],
    min_gain: int,
    by_id: dict[str, VariantScorecard],
) -> str:
    """Motivo concreto y verificable, nunca "no seleccionada" a secas."""
    if any(entry["variant_id"] == card.variant_id for entry in blocked):
        return (
            f"{EXCESSIVE_CHUNK_INFLATION}: {card.chunk_count_ratio:.2f}x chunks, por encima del "
            f"limite de {EXCESSIVE_CHUNK_INFLATION_RATIO}x"
        )
    if gains[card.variant_id] < min_gain:
        return (
            f"gain {gains[card.variant_id]:+d} evidencias sobre el baseline, por debajo del "
            f"minimo declarado de {min_gain}"
        )
    if card.variant_id not in non_dominated:
        dominators = [
            other.variant_id
            for other in by_id.values()
            if other is not card and dominates(other, card)
        ]
        return f"dominada en el Pareto por {', '.join(sorted(dominators))}"
    return (
        "no dominada y por encima del gate, pero por debajo de los finalistas en el orden de "
        "preferencia (cobertura, menos chunks, menos overlap, mas representabilidad raw)"
    )
