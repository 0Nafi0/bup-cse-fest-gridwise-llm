"""Integration and contract tests for FastAPI endpoints."""

from typing import Any, Dict
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """GET /health must return HTTP 200 with {'status': 'ok'}."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_optimize_energy_empty_payload(client: TestClient) -> None:
    """POST /optimize-energy with empty body must return 400 Bad Request."""
    response = client.post("/optimize-energy", json={})
    assert response.status_code == 400
    assert "error" in response.json()


def test_optimize_energy_missing_hours(
    client: TestClient, sample_cases_data: Dict[str, Any]
) -> None:
    """POST /optimize-energy with incomplete hours list must return 400."""
    payload = dict(sample_cases_data["cases"][0]["input"])
    # Truncate hours to 23
    payload["hours"] = payload["hours"][:23]

    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_optimize_energy_invalid_battery_bounds(
    client: TestClient, sample_cases_data: Dict[str, Any]
) -> None:
    """POST /optimize-energy with initial_energy exceeding capacity must fail."""
    payload = dict(sample_cases_data["cases"][0]["input"])
    invalid_battery = dict(payload["battery"])
    invalid_battery["initial_energy_kwh"] = invalid_battery["capacity_kwh"] + 100
    payload["battery"] = invalid_battery

    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_optimize_energy_empty_operator_notes(
    client: TestClient, sample_cases_data: Dict[str, Any]
) -> None:
    """POST /optimize-energy with empty notes list must fail."""
    payload = dict(sample_cases_data["cases"][0]["input"])
    payload["operator_notes"] = []

    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400
    assert "error" in response.json()


def test_optimize_energy_schema_contract(
    client: TestClient, sample_cases_data: Dict[str, Any]
) -> None:
    """POST /optimize-energy response must conform strictly to the Problem Statement schema."""
    case_0 = sample_cases_data["cases"][0]
    response = client.post("/optimize-energy", json=case_0["input"])
    assert response.status_code == 200

    data = response.json()
    assert data["scenario_id"] == case_0["input"]["scenario_id"]

    # Directive interpretation checks
    assert isinstance(data["directive_interpretation"], list)
    assert len(data["directive_interpretation"]) == len(case_0["input"]["operator_notes"])
    for i, interp in enumerate(data["directive_interpretation"]):
        assert interp["note_index"] == i
        assert isinstance(interp["applies"], bool)
        assert interp["directive_type"] in [
            "solar_reduction",
            "minimum_battery_reserve",
            "no_charge_window",
            "no_discharge_window",
            "max_grid_window",
            "no_op",
        ]
        if interp["applies"]:
            assert isinstance(interp["structured_adjustment"], dict)
        else:
            assert interp["structured_adjustment"] is None

    # Hourly plan checks
    assert isinstance(data["hourly_plan"], list)
    assert len(data["hourly_plan"]) == 24
    for h, plan in enumerate(data["hourly_plan"]):
        assert plan["hour"] == h
        assert plan["battery_action"] in ["charge", "discharge", "idle"]
        assert plan["grid_kwh"] >= 0.0
        assert plan["solar_used_kwh"] >= 0.0
        assert plan["battery_kwh"] >= 0.0
        assert plan["battery_energy_after_kwh"] >= 0.0

    # Top-level metrics
    assert isinstance(data["total_grid_kwh"], (int, float))
    assert isinstance(data["total_cost_bdt"], (int, float))
    assert isinstance(data["peak_grid_kwh"], (int, float))
    assert isinstance(data["plan_summary"], str)
    assert len(data["plan_summary"]) > 0
