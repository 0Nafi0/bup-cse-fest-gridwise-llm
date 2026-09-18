"""High-performance Linear Programming (LP) dispatch optimizer using SciPy HiGHS.

Solves the 24-hour campus energy scheduling problem while respecting all physical,
battery, grid, and operator-directive constraints, minimizing grid electricity cost.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linprog

from app.config import settings
from app.schemas import (
    BatteryAction,
    BatteryInput,
    DirectiveInterpretation,
    DirectiveType,
    HourInput,
    HourlyPlanEntry,
)


class OptimizationError(Exception):
    """Raised when optimization is infeasible or solver fails."""
    pass


def apply_directives_to_profiles(
    hours: List[HourInput],
    battery: BatteryInput,
    directives: List[DirectiveInterpretation],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate effective parameter profiles after applying all valid directives.

    Returns:
        Tuple of:
        - effective_solar (np.ndarray of shape (24,))
        - active_min_reserve (np.ndarray of shape (24,))
        - active_charge_limit (np.ndarray of shape (24,))
        - active_discharge_limit (np.ndarray of shape (24,))
        - active_grid_max (np.ndarray of shape (24,))
    """
    effective_solar = np.array([h.solar_kwh for h in hours], dtype=float)
    active_min_reserve = np.full(24, battery.minimum_energy_kwh, dtype=float)
    active_charge_limit = np.full(24, battery.max_charge_kwh_per_hour, dtype=float)
    active_discharge_limit = np.full(24, battery.max_discharge_kwh_per_hour, dtype=float)
    active_grid_max = np.full(24, np.inf, dtype=float)

    for directive in directives:
        if not directive.applies or not directive.structured_adjustment:
            continue

        adj = directive.structured_adjustment
        adj_hours = adj.get("hours", [])

        if directive.directive_type == DirectiveType.SOLAR_REDUCTION:
            factor = float(adj.get("factor", 1.0))
            for h in adj_hours:
                if 0 <= h < 24:
                    effective_solar[h] *= factor

        elif directive.directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE:
            min_kwh = float(adj.get("minimum_energy_kwh", battery.minimum_energy_kwh))
            for h in adj_hours:
                if 0 <= h < 24:
                    active_min_reserve[h] = max(active_min_reserve[h], min_kwh)

        elif directive.directive_type == DirectiveType.NO_CHARGE_WINDOW:
            for h in adj_hours:
                if 0 <= h < 24:
                    active_charge_limit[h] = 0.0

        elif directive.directive_type == DirectiveType.NO_DISCHARGE_WINDOW:
            for h in adj_hours:
                if 0 <= h < 24:
                    active_discharge_limit[h] = 0.0

        elif directive.directive_type == DirectiveType.MAX_GRID_WINDOW:
            max_grid = float(adj.get("max_grid_kwh", np.inf))
            for h in adj_hours:
                if 0 <= h < 24:
                    active_grid_max[h] = min(active_grid_max[h], max_grid)

    return (
        effective_solar,
        active_min_reserve,
        active_charge_limit,
        active_discharge_limit,
        active_grid_max,
    )


def solve_energy_dispatch(
    hours: List[HourInput],
    battery: BatteryInput,
    directives: List[DirectiveInterpretation],
) -> List[HourlyPlanEntry]:
    """Formulate and solve the 24-hour linear program using HiGHS solver.

    Decision Variables (97 total):
    - 0..23:  grid[h] >= 0
    - 24..47: solar_used[h] >= 0
    - 48..71: charge[h] >= 0
    - 72..95: discharge[h] >= 0
    - 96:     M (peak grid envelope) >= 0

    Objective:
    Min sum(tariff[h] * grid[h]) - 1e-6 * sum(solar_used) + 1e-7 * sum(ch + dis) + lambda * M
    """
    (
        effective_solar,
        active_min_reserve,
        active_charge_limit,
        active_discharge_limit,
        active_grid_max,
    ) = apply_directives_to_profiles(hours, battery, directives)

    demand = np.array([h.demand_kwh for h in hours], dtype=float)
    tariff = np.array([h.tariff_bdt_per_kwh for h in hours], dtype=float)
    capacity = float(battery.capacity_kwh)
    e0 = float(battery.initial_energy_kwh)

    # Cost vector c (97 elements)
    c = np.zeros(97, dtype=float)
    c[0:24] = tariff
    c[24:48] = -1e-6  # incentivize using clean zero-emission solar
    c[48:72] = 1e-7   # minimal tie-breaker to prevent gratuitous cycling
    c[72:96] = 1e-7   # minimal tie-breaker to prevent gratuitous cycling
    c[96] = settings.PEAK_GRID_REGULARIZATION  # regularize peak grid among equal-tariff hours

    # Equality constraints: A_eq @ x = b_eq
    # 1. Energy balance for each hour h (24 constraints)
    #    grid[h] + solar_used[h] - ch[h] + dis[h] = demand[h]
    # 2. End-of-day battery neutrality (1 constraint)
    #    sum(ch) - sum(dis) = 0
    A_eq = []
    b_eq = []

    for h in range(24):
        row = np.zeros(97, dtype=float)
        row[h] = 1.0        # grid[h]
        row[24 + h] = 1.0   # solar_used[h]
        row[48 + h] = -1.0  # -ch[h]
        row[72 + h] = 1.0   # +dis[h]
        A_eq.append(row)
        b_eq.append(demand[h])

    # End of day neutrality: E[23] = e0 => sum_{i=0}^23 (ch[i] - dis[i]) = 0
    row_eod = np.zeros(97, dtype=float)
    row_eod[48:72] = 1.0
    row_eod[72:96] = -1.0
    A_eq.append(row_eod)
    b_eq.append(0.0)

    # Inequality constraints: A_ub @ x <= b_ub
    # Battery state after hour h:
    # E[h] = e0 + sum_{i=0}^h (ch[i] - dis[i])
    # E[h] <= capacity       => sum_{i=0}^h (ch[i] - dis[i]) <= capacity - e0
    # E[h] >= min_reserve[h] => -sum_{i=0}^h (ch[i] - dis[i]) <= e0 - min_reserve[h]
    # Peak grid envelope:
    # grid[h] <= M           => grid[h] - M <= 0
    A_ub = []
    b_ub = []

    for h in range(24):
        # Upper bound: capacity
        row_cap = np.zeros(97, dtype=float)
        row_cap[48 : 48 + h + 1] = 1.0
        row_cap[72 : 72 + h + 1] = -1.0
        A_ub.append(row_cap)
        b_ub.append(capacity - e0)

        # Lower bound: active minimum reserve
        row_min = np.zeros(97, dtype=float)
        row_min[48 : 48 + h + 1] = -1.0
        row_min[72 : 72 + h + 1] = 1.0
        A_ub.append(row_min)
        b_ub.append(e0 - active_min_reserve[h])

        # Peak envelope: grid[h] - M <= 0
        row_peak = np.zeros(97, dtype=float)
        row_peak[h] = 1.0
        row_peak[96] = -1.0
        A_ub.append(row_peak)
        b_ub.append(0.0)

    # Variable bounds
    bounds = []
    for h in range(24):
        bounds.append((0.0, None if np.isinf(active_grid_max[h]) else float(active_grid_max[h])))
    for h in range(24):
        bounds.append((0.0, float(effective_solar[h])))
    for h in range(24):
        bounds.append((0.0, float(active_charge_limit[h])))
    for h in range(24):
        bounds.append((0.0, float(active_discharge_limit[h])))
    bounds.append((0.0, None))  # M >= 0

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise OptimizationError(f"Linear programming solver failed: {res.message}")

    x = res.x
    grid_raw = x[0:24]
    solar_used_raw = x[24:48]
    ch_raw = x[48:72]
    dis_raw = x[72:96]

    # Post-process to eliminate micro-simultaneous charge and discharge
    simultaneous = np.minimum(ch_raw, dis_raw)
    ch = ch_raw - simultaneous
    dis = dis_raw - simultaneous

    hourly_plan: List[HourlyPlanEntry] = []
    current_energy = e0

    for h in range(24):
        g = float(np.round(grid_raw[h], 4))
        s = float(np.round(solar_used_raw[h], 4))
        c_h = float(ch[h])
        d_h = float(dis[h])

        # Action classification
        if c_h > 1e-5:
            action = BatteryAction.CHARGE
            action_kwh = float(np.round(c_h, 4))
            current_energy += action_kwh
        elif d_h > 1e-5:
            action = BatteryAction.DISCHARGE
            action_kwh = float(np.round(d_h, 4))
            current_energy -= action_kwh
        else:
            action = BatteryAction.IDLE
            action_kwh = 0.0

        # Adjust tiny floating point drift on battery energy
        current_energy = max(
            float(active_min_reserve[h]),
            min(capacity, float(np.round(current_energy, 4))),
        )
        if h == 23:
            # Guarantee exact end-of-day neutrality
            current_energy = float(round(e0, 4))

        hourly_plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=g,
                solar_used_kwh=s,
                battery_action=action,
                battery_kwh=action_kwh,
                battery_energy_after_kwh=float(round(current_energy, 4)),
            )
        )

    return hourly_plan
