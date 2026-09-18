"""Pytest configuration and shared test fixtures."""

import json
from pathlib import Path
from typing import Any, Dict, Generator
from fastapi.testclient import TestClient
import pytest

from app.main import app

SAMPLE_CASES_PATH = (
    Path(__file__).parent.parent
    / "BUP_CSE_FEST_2026_Participant_Docs"
    / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"
)


@pytest.fixture(scope="session")
def sample_cases_data() -> Dict[str, Any]:
    """Load the official public sample cases JSON pack."""
    assert SAMPLE_CASES_PATH.exists(), f"Sample cases file not found at: {SAMPLE_CASES_PATH}"
    with open(SAMPLE_CASES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient instance."""
    with TestClient(app) as test_client:
        yield test_client
