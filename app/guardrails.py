"""Deterministic guardrails and normalization for LLM directive interpretations.

Enforces Section 08 Guardrails:
- Strict directive type membership
- 1-to-1 mapping in note_index order (0..N-1)
- Ascending unique hours normalization in [0..23]
- Factor clamping [0.0..1.0] for solar_reduction
- Battery reserve bounds [0.0..capacity_kwh]
- Max grid limits >= 0.0
- Exact applies/null semantics: applies=False iff directive_type=no_op
- Safe failure handling without crashing
"""

import logging
from typing import Any, Dict, List, Optional
from app.schemas import (
    BatteryInput,
    DirectiveInterpretation,
    DirectiveType,
)

logger = logging.getLogger(__name__)


class GuardrailValidationError(Exception):
    """Raised when an LLM directive interpretation violates critical invariants."""
    pass


def normalize_hours(raw_hours: Any) -> List[int]:
    """Extract, filter, deduplicate, and sort hour integers into [0..23]."""
    if not isinstance(raw_hours, list):
        return []
    valid_hours = set()
    for h in raw_hours:
        try:
            h_int = int(h)
            if 0 <= h_int <= 23:
                valid_hours.add(h_int)
        except (ValueError, TypeError):
            continue
    return sorted(list(valid_hours))


def sanitize_directive_interpretation(
    raw_interp: Any,
    note_index: int,
    operator_note: str,
    battery: BatteryInput,
) -> DirectiveInterpretation:
    """Validate and normalize a single directive interpretation entry.

    Applies deterministic guardrails to prevent hallucinations, clamp numeric
    values, ensure ordered hours, and strictly enforce applies semantics.
    """
    default_explanation = f"Interpreted directive for note {note_index}."

    if not isinstance(raw_interp, dict):
        logger.warning(
            "Raw interpretation for note %d is not a dict: %s. Defaulting to no_op.",
            note_index,
            type(raw_interp),
        )
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=f"Malformed LLM output for note {note_index}; treated as no_op.",
        )

    # 1. Resolve directive_type
    raw_type = str(raw_interp.get("directive_type", "")).strip().lower()
    try:
        dtype = DirectiveType(raw_type)
    except ValueError:
        logger.warning(
            "Unrecognized directive_type '%s' for note %d. Defaulting to no_op.",
            raw_type,
            note_index,
        )
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=f"Unsupported directive type '{raw_type}' received; normalized to no_op.",
        )

    explanation = str(raw_interp.get("explanation") or default_explanation).strip()

    # 2. If directive is NO_OP, enforce applies=False and structured_adjustment=None
    if dtype == DirectiveType.NO_OP:
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=explanation or "This note does not affect the 24-hour energy schedule.",
        )

    # 3. For non-no_op directives, applies must be True and structured_adjustment must be present
    raw_adj = raw_interp.get("structured_adjustment")
    if not isinstance(raw_adj, dict):
        logger.warning(
            "Directive %s for note %d missing valid structured_adjustment. Reverting to no_op.",
            dtype.value,
            note_index,
        )
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation="Missing adjustment parameters for directive; treated as no_op.",
        )

    hours = normalize_hours(raw_adj.get("hours"))
    if not hours:
        logger.warning(
            "Directive %s for note %d contains no valid hours in [0..23]. Reverting to no_op.",
            dtype.value,
            note_index,
        )
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation="No valid hours specified for directive; treated as no_op.",
        )

    # 4. Process directive-specific parameters
    if dtype == DirectiveType.SOLAR_REDUCTION:
        raw_factor = raw_adj.get("factor")
        try:
            factor = float(raw_factor)
            factor = max(0.0, min(1.0, factor))
        except (ValueError, TypeError):
            logger.warning("Invalid solar reduction factor: %s. Defaulting to 1.0.", raw_factor)
            factor = 1.0

        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "factor": factor},
            explanation=explanation,
        )

    elif dtype == DirectiveType.MINIMUM_BATTERY_RESERVE:
        raw_min = raw_adj.get("minimum_energy_kwh")
        try:
            min_kwh = float(raw_min)
            min_kwh = max(0.0, min(battery.capacity_kwh, min_kwh))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid minimum_energy_kwh: %s. Clamping to base minimum %f.",
                raw_min,
                battery.minimum_energy_kwh,
            )
            min_kwh = battery.minimum_energy_kwh

        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "minimum_energy_kwh": min_kwh},
            explanation=explanation,
        )

    elif dtype in (DirectiveType.NO_CHARGE_WINDOW, DirectiveType.NO_DISCHARGE_WINDOW):
        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours},
            explanation=explanation,
        )

    elif dtype == DirectiveType.MAX_GRID_WINDOW:
        raw_max = raw_adj.get("max_grid_kwh")
        try:
            max_grid = float(raw_max)
            max_grid = max(0.0, max_grid)
        except (ValueError, TypeError):
            logger.warning("Invalid max_grid_kwh: %s. Reverting to no_op.", raw_max)
            return DirectiveInterpretation(
                note_index=note_index,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                structured_adjustment=None,
                explanation="Invalid max_grid value; treated as no_op.",
            )

        return DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type=dtype,
            structured_adjustment={"hours": hours, "max_grid_kwh": max_grid},
            explanation=explanation,
        )

    # Fallback safe failure
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation="Fallback safe failure normalization.",
    )


def validate_and_align_interpretations(
    raw_interpretations: List[Any],
    operator_notes: List[str],
    battery: BatteryInput,
) -> List[DirectiveInterpretation]:
    """Validate, sanitize, and strictly align directives to operator_notes in order.

    Ensures exactly len(operator_notes) entries are returned, with note_index
    ranging strictly from 0 to N-1.
    """
    cleaned: List[DirectiveInterpretation] = []

    # Map raw entries by note_index if available
    interp_map: Dict[int, Any] = {}
    if isinstance(raw_interpretations, list):
        for item in raw_interpretations:
            if isinstance(item, dict) and "note_index" in item:
                try:
                    idx = int(item["note_index"])
                    if idx not in interp_map:
                        interp_map[idx] = item
                except (ValueError, TypeError):
                    continue

    for i, note in enumerate(operator_notes):
        raw_item = interp_map.get(i)
        # If not keyed by note_index, check list index
        if raw_item is None and isinstance(raw_interpretations, list) and i < len(raw_interpretations):
            raw_item = raw_interpretations[i]

        sanitized = sanitize_directive_interpretation(
            raw_interp=raw_item,
            note_index=i,
            operator_note=note,
            battery=battery,
        )
        cleaned.append(sanitized)

    return cleaned
