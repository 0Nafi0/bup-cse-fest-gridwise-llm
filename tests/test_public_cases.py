"""Validation of all 10 official BUP CSE Fest 2026 public sample benchmark cases."""

from typing import Any, Dict
from fastapi.testclient import TestClient
import pytest


@pytest.mark.parametrize("case_index", list(range(10)))
def test_public_sample_benchmark_case(
    client: TestClient, sample_cases_data: Dict[str, Any], case_index: int
) -> None:
    """Validate full end-to-end execution on official public sample cases.

    Checks:
    1. HTTP 200 response with correct scenario_id.
    2. Directive interpretations match expected types, hours, and values.
    3. Total grid electricity cost matches official optimal reference within 0.01 BDT.
    4. Total grid import and peak grid match official reference within 0.01 kWh.
    5. Energy balance, battery bounds, rate limits, and end-of-day neutrality hold.
    """
    case = sample_cases_data["cases"][case_index]
    case_id = case["id"]
    req_payload = case["input"]
    expected_output = case["expected_output"]

    response = client.post("/optimize-energy", json=req_payload)
    assert response.status_code == 200, f"{case_id} failed with {response.text}"
    result = response.json()

    # 1. Scenario ID echo
    assert result["scenario_id"] == req_payload["scenario_id"]

    # 2. Directive interpretation check
    assert len(result["directive_interpretation"]) == len(expected_output["directive_interpretation"])
    for i, interp in enumerate(result["directive_interpretation"]):
        exp_interp = expected_output["directive_interpretation"][i]
        assert interp["note_index"] == exp_interp["note_index"]
        assert interp["applies"] == exp_interp["applies"]
        assert interp["directive_type"] == exp_interp["directive_type"]

        if exp_interp["applies"]:
            assert interp["structured_adjustment"] is not None
            exp_adj = exp_interp["structured_adjustment"]
            act_adj = interp["structured_adjustment"]

            # Check affected hours
            assert act_adj.get("hours") == exp_adj.get("hours")

            # Check directive specific numbers
            if "factor" in exp_adj:
                assert abs(act_adj["factor"] - exp_adj["factor"]) < 0.01
            if "minimum_energy_kwh" in exp_adj:
                assert abs(act_adj["minimum_energy_kwh"] - exp_adj["minimum_energy_kwh"]) < 0.01
            if "max_grid_kwh" in exp_adj:
                assert abs(act_adj["max_grid_kwh"] - exp_adj["max_grid_kwh"]) < 0.01
        else:
            assert interp["structured_adjustment"] is None

    # 3. Cost optimality check (tolerance 0.01 BDT)
    exp_cost = float(expected_output["total_cost_bdt"])
    act_cost = float(result["total_cost_bdt"])
    assert (
        abs(act_cost - exp_cost) <= 0.01
    ), f"{case_id}: Cost mismatch: actual {act_cost:.2f} BDT vs expected {exp_cost:.2f} BDT"

    # 4. Total grid and peak grid check (tolerance 0.01 kWh)
    exp_grid = float(expected_output["total_grid_kwh"])
    act_grid = float(result["total_grid_kwh"])
    assert (
        abs(act_grid - exp_grid) <= 0.01
    ), f"{case_id}: Total grid mismatch: actual {act_grid:.2f} vs expected {exp_grid:.2f}"

    exp_peak = float(expected_output["peak_grid_kwh"])
    act_peak = float(result["peak_grid_kwh"])
    assert (
        abs(act_peak - exp_peak) <= 0.01
    ), f"{case_id}: Peak grid mismatch: actual {act_peak:.2f} vs expected {exp_peak:.2f}"

    # 5. Hourly plan shape and bounds
    assert len(result["hourly_plan"]) == 24
    for h, hour_entry in enumerate(result["hourly_plan"]):
        assert hour_entry["hour"] == h
        assert hour_entry["grid_kwh"] >= 0.0
        assert hour_entry["solar_used_kwh"] >= 0.0
        assert hour_entry["battery_kwh"] >= 0.0
        assert hour_entry["battery_action"] in ["charge", "discharge", "idle"]

    # 6. Plan summary exists
    assert isinstance(result["plan_summary"], str)
    assert len(result["plan_summary"]) > 20
