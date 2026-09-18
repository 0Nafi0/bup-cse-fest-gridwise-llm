"""Domain models, Pydantic V2 schemas, and enums for GridWise.

Defines the canonical API contracts, request/response structures,
battery actions, and operator directive types according to the
BUP CSE Fest 2026 Problem Statement and Participant Guide.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class DirectiveType(str, Enum):
    """Supported operator directive types."""

    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class BatteryAction(str, Enum):
    """Battery operational status for each hour."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class HourInput(BaseModel):
    """Input definition for a single hour in the 24-hour horizon."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0..23).")
    demand_kwh: float = Field(
        ..., ge=0.0, description="Campus electricity demand in kWh."
    )
    solar_kwh: float = Field(
        ..., ge=0.0, description="Forecasted solar generation in kWh."
    )
    tariff_bdt_per_kwh: float = Field(
        ..., description="Grid tariff rate in BDT per kWh."
    )


class BatteryInput(BaseModel):
    """Battery physical specifications and operational limits."""

    capacity_kwh: float = Field(
        ..., gt=0.0, description="Total battery capacity in kWh."
    )
    initial_energy_kwh: float = Field(
        ..., ge=0.0, description="Initial battery energy at start of hour 0."
    )
    minimum_energy_kwh: float = Field(
        ..., ge=0.0, description="Base minimum reserve energy in kWh."
    )
    max_charge_kwh_per_hour: float = Field(
        ..., ge=0.0, description="Maximum charge rate in kWh per hour."
    )
    max_discharge_kwh_per_hour: float = Field(
        ..., ge=0.0, description="Maximum discharge rate in kWh per hour."
    )

    @model_validator(mode="after")
    def validate_battery_invariants(self) -> "BatteryInput":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError(
                f"minimum_energy_kwh ({self.minimum_energy_kwh}) cannot exceed capacity_kwh ({self.capacity_kwh})"
            )
        if self.initial_energy_kwh > self.capacity_kwh:
            raise ValueError(
                f"initial_energy_kwh ({self.initial_energy_kwh}) cannot exceed capacity_kwh ({self.capacity_kwh})"
            )
        if self.initial_energy_kwh < self.minimum_energy_kwh:
            raise ValueError(
                f"initial_energy_kwh ({self.initial_energy_kwh}) cannot be less than minimum_energy_kwh ({self.minimum_energy_kwh})"
            )
        return self


class OptimizeEnergyRequest(BaseModel):
    """Request payload for POST /optimize-energy."""

    scenario_id: str = Field(..., min_length=1, description="Unique scenario identifier.")
    operator_notes: List[str] = Field(
        ...,
        min_length=1,
        max_length=3,
        description="List of 1 to 3 natural-language operator notes.",
    )
    hours: List[HourInput] = Field(
        ...,
        min_length=24,
        max_length=24,
        description="Hourly inputs for all 24 hours.",
    )
    battery: BatteryInput = Field(..., description="Battery configuration.")

    @field_validator("operator_notes")
    @classmethod
    def validate_operator_notes(cls, notes: List[str]) -> List[str]:
        for i, note in enumerate(notes):
            if not note or not note.strip():
                raise ValueError(f"operator_note[{i}] must not be empty.")
        return notes

    @model_validator(mode="after")
    def validate_hours_sequence(self) -> "OptimizeEnergyRequest":
        hours_seen = [h.hour for h in self.hours]
        if sorted(hours_seen) != list(range(24)):
            raise ValueError(
                f"hours must contain exactly 24 entries covering 0 through 23, got: {sorted(hours_seen)}"
            )
        return self


# ---------------------------------------------------------------------------
# Directive & Adjustment Schemas
# ---------------------------------------------------------------------------


class SolarReductionAdjustment(BaseModel):
    """Structured adjustment for solar_reduction."""

    hours: List[int] = Field(..., description="Affected hours (0..23).")
    factor: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Remaining usable solar fraction (0.0..1.0).",
    )


class MinimumBatteryReserveAdjustment(BaseModel):
    """Structured adjustment for minimum_battery_reserve."""

    hours: List[int] = Field(..., description="Affected hours (0..23).")
    minimum_energy_kwh: float = Field(
        ..., ge=0.0, description="Mandated minimum battery reserve in kWh."
    )


class NoChargeWindowAdjustment(BaseModel):
    """Structured adjustment for no_charge_window."""

    hours: List[int] = Field(..., description="Affected hours (0..23).")


class NoDischargeWindowAdjustment(BaseModel):
    """Structured adjustment for no_discharge_window."""

    hours: List[int] = Field(..., description="Affected hours (0..23).")


class MaxGridWindowAdjustment(BaseModel):
    """Structured adjustment for max_grid_window."""

    hours: List[int] = Field(..., description="Affected hours (0..23).")
    max_grid_kwh: float = Field(
        ..., ge=0.0, description="Maximum permitted grid import in kWh."
    )


StructuredAdjustmentType = Optional[
    Union[
        SolarReductionAdjustment,
        MinimumBatteryReserveAdjustment,
        NoChargeWindowAdjustment,
        NoDischargeWindowAdjustment,
        MaxGridWindowAdjustment,
        Dict[str, Any],
    ]
]


class DirectiveInterpretation(BaseModel):
    """Interpretation result for a single operator note."""

    note_index: int = Field(..., ge=0, description="Zero-based index of operator note.")
    applies: bool = Field(
        ..., description="True if note alters energy schedule, False only for no_op."
    )
    directive_type: DirectiveType = Field(
        ..., description="Standardized directive category."
    )
    structured_adjustment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Machine-checkable parameters, or null for no_op.",
    )
    explanation: str = Field(
        ..., description="Human-readable rationale for the interpretation."
    )


# ---------------------------------------------------------------------------
# Hourly Plan & Response Schemas
# ---------------------------------------------------------------------------


class HourlyPlanEntry(BaseModel):
    """Scheduled energy allocation for a single hour."""

    hour: int = Field(..., ge=0, le=23, description="Hour of the day (0..23).")
    grid_kwh: float = Field(..., ge=0.0, description="Grid import in kWh.")
    solar_used_kwh: float = Field(
        ..., ge=0.0, description="Solar generation consumed in kWh."
    )
    battery_action: BatteryAction = Field(
        ..., description="Battery action: charge, discharge, or idle."
    )
    battery_kwh: float = Field(
        ..., ge=0.0, description="Magnitude of battery charge/discharge in kWh."
    )
    battery_energy_after_kwh: float = Field(
        ..., ge=0.0, description="Battery energy state at end of hour in kWh."
    )


class OptimizeEnergyResponse(BaseModel):
    """Complete response payload for POST /optimize-energy."""

    scenario_id: str = Field(..., description="Echo of request scenario_id.")
    directive_interpretation: List[DirectiveInterpretation] = Field(
        ..., description="Machine-checkable interpretation per note in order."
    )
    hourly_plan: List[HourlyPlanEntry] = Field(
        ..., min_length=24, max_length=24, description="Hourly plan for 24 hours."
    )
    total_grid_kwh: float = Field(
        ..., ge=0.0, description="Sum of grid energy imported across 24 hours."
    )
    total_cost_bdt: float = Field(
        ..., description="Total grid electricity cost in BDT."
    )
    peak_grid_kwh: float = Field(
        ..., ge=0.0, description="Maximum single-hour grid import in kWh."
    )
    plan_summary: str = Field(
        ..., description="Human-readable summary of the optimization strategy."
    )


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = Field(default="ok", description="Service readiness status.")
