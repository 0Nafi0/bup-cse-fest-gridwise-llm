"""24-Hour simulation replay, constraint verification, and metric rollup.

Independently validates the optimizer's hourly schedule against GridWise physical rules,
battery state dynamics, energy balance equations, and active operator directives.
Computes authoritative summary metrics and generates plan summaries.
"""

from typing import List, Tuple
from app.config import settings
from app.optimizer import apply_directives_to_profiles
from app.schemas import (
    BatteryAction,
    BatteryInput,
    DirectiveInterpretation,
    DirectiveType,
    HourInput,
    HourlyPlanEntry,
)


class SimulationReplayError(Exception):
    """Raised when the 24-hour schedule breaks physical or directive constraints."""
    pass


def verify_and_replay_schedule(
    hourly_plan: List[HourlyPlanEntry],
    hours: List[HourInput],
    battery: BatteryInput,
    directives: List[DirectiveInterpretation],
    tolerance: float = 0.01,
) -> Tuple[float, float, float]:
    """Replay and verify the 24-hour schedule against all constraints.

    Returns:
        Tuple of (total_grid_kwh, total_cost_bdt, peak_grid_kwh)

    Raises:
        SimulationReplayError: If any energy balance, rate limit, battery bound,
            directive requirement, or end-of-day neutrality invariant fails.
    """
    if len(hourly_plan) != 24:
        raise SimulationReplayError(f"hourly_plan must have 24 entries, got {len(hourly_plan)}")

    (
        effective_solar,
        active_min_reserve,
        active_charge_limit,
        active_discharge_limit,
        active_grid_max,
    ) = apply_directives_to_profiles(hours, battery, directives)

    current_battery = float(battery.initial_energy_kwh)
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h, plan in enumerate(hourly_plan):
        hour_in = hours[h]
        if plan.hour != h:
            raise SimulationReplayError(f"Expected hour {h}, got {plan.hour}")

        # 1. Non-negativity
        if plan.grid_kwh < -tolerance:
            raise SimulationReplayError(f"Hour {h}: negative grid_kwh {plan.grid_kwh}")
        if plan.solar_used_kwh < -tolerance:
            raise SimulationReplayError(f"Hour {h}: negative solar_used_kwh {plan.solar_used_kwh}")
        if plan.battery_kwh < -tolerance:
            raise SimulationReplayError(f"Hour {h}: negative battery_kwh {plan.battery_kwh}")

        # 2. Solar availability
        if plan.solar_used_kwh > effective_solar[h] + tolerance:
            raise SimulationReplayError(
                f"Hour {h}: solar_used_kwh ({plan.solar_used_kwh}) exceeds effective solar ({effective_solar[h]})"
            )

        # 3. Grid cap directive check
        if plan.grid_kwh > active_grid_max[h] + tolerance:
            raise SimulationReplayError(
                f"Hour {h}: grid_kwh ({plan.grid_kwh}) exceeds max_grid_window limit ({active_grid_max[h]})"
            )

        # 4. Battery action and state transition
        dis = 0.0
        ch = 0.0
        if plan.battery_action == BatteryAction.CHARGE:
            ch = plan.battery_kwh
            if ch > active_charge_limit[h] + tolerance:
                raise SimulationReplayError(
                    f"Hour {h}: charge ({ch}) exceeds hourly rate limit ({active_charge_limit[h]})"
                )
            current_battery += ch
        elif plan.battery_action == BatteryAction.DISCHARGE:
            dis = plan.battery_kwh
            if dis > active_discharge_limit[h] + tolerance:
                raise SimulationReplayError(
                    f"Hour {h}: discharge ({dis}) exceeds hourly rate limit ({active_discharge_limit[h]})"
                )
            current_battery -= dis
        elif plan.battery_action == BatteryAction.IDLE:
            if plan.battery_kwh > tolerance:
                raise SimulationReplayError(
                    f"Hour {h}: battery is idle but battery_kwh is {plan.battery_kwh}"
                )

        # 5. Energy state bounds
        if current_battery < active_min_reserve[h] - tolerance:
            raise SimulationReplayError(
                f"Hour {h}: battery energy ({current_battery:.2f}) dropped below active reserve ({active_min_reserve[h]:.2f})"
            )
        if current_battery > battery.capacity_kwh + tolerance:
            raise SimulationReplayError(
                f"Hour {h}: battery energy ({current_battery:.2f}) exceeded capacity ({battery.capacity_kwh:.2f})"
            )

        # Compare reported battery_energy_after_kwh
        if abs(plan.battery_energy_after_kwh - current_battery) > tolerance:
            raise SimulationReplayError(
                f"Hour {h}: reported battery_energy_after_kwh ({plan.battery_energy_after_kwh}) does not match simulated ({current_battery:.2f})"
            )

        # 6. Energy Balance
        supply = plan.grid_kwh + plan.solar_used_kwh + dis
        demand = hour_in.demand_kwh + ch
        if abs(supply - demand) > tolerance:
            raise SimulationReplayError(
                f"Hour {h}: energy balance violated: supply={supply:.3f} (grid={plan.grid_kwh}, solar={plan.solar_used_kwh}, dis={dis}), demand={demand:.3f} (demand={hour_in.demand_kwh}, ch={ch})"
            )

        # 7. Metric accumulation
        total_grid += plan.grid_kwh
        total_cost += plan.grid_kwh * hour_in.tariff_bdt_per_kwh
        if plan.grid_kwh > peak_grid:
            peak_grid = plan.grid_kwh

    # 8. End-of-day battery neutrality
    if abs(current_battery - battery.initial_energy_kwh) > tolerance:
        raise SimulationReplayError(
            f"End-of-day battery energy ({current_battery:.2f}) does not equal initial energy ({battery.initial_energy_kwh:.2f})"
        )

    return (
        round(total_grid, 2),
        round(total_cost, 2),
        round(peak_grid, 2),
    )


def generate_plan_summary(
    directives: List[DirectiveInterpretation],
    total_grid_kwh: float,
    total_cost_bdt: float,
    peak_grid_kwh: float,
) -> str:
    """Construct an informative, professional strategy summary."""
    active_types = [
        d.directive_type.value for d in directives if d.applies and d.directive_type != DirectiveType.NO_OP
    ]
    ignored_count = sum(1 for d in directives if not d.applies or d.directive_type == DirectiveType.NO_OP)

    directives_desc = ""
    if active_types:
        directives_desc = f"Active operator directives applied: {', '.join(set(active_types))}."
    else:
        directives_desc = "No restrictive operator directives active."

    if ignored_count > 0:
        directives_desc += f" Ignored {ignored_count} non-operational note(s)."

    summary = (
        f"Optimized 24-hour dispatch minimizing grid cost to {total_cost_bdt:,.2f} BDT "
        f"({total_grid_kwh:,.2f} kWh imported, peak {peak_grid_kwh:,.2f} kWh). "
        f"{directives_desc} "
        f"Battery shifts low-cost and solar power to peak-tariff windows while preserving end-of-day neutrality."
    )
    return summary
