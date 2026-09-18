"""FastAPI service for the BUP CSE Fest 2026 GridWise energy optimization challenge.

Exposes canonical endpoints:
- GET /health: Health readiness check returning {"status": "ok"}
- POST /optimize-energy: LLM-assisted directive interpretation, LP dispatch optimization,
  and 24-hour schedule simulation.
"""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.guardrails import validate_and_align_interpretations
from app.llm_parser import extract_directives_from_notes
from app.optimizer import OptimizationError, solve_energy_dispatch
from app.replay import (
    SimulationReplayError,
    generate_plan_summary,
    verify_and_replay_schedule,
)
from app.schemas import (
    HealthResponse,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("gridwise.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Service lifecycle management."""
    logger.info(
        "Starting GridWise Service (Provider: %s, Model: %s, Host: %s, Port: %d)",
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        settings.HOST,
        settings.PORT,
    )
    yield
    logger.info("Shutting down GridWise Service.")


app = FastAPI(
    title="GridWise Campus Energy Optimization API",
    version="1.0.0",
    description="LLM-Assisted Smart Campus Energy Scheduling Service for BUP CSE Fest 2026.",
    lifespan=lifespan,
)

# Enable CORS for open judging accessibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Format request validation errors into a clean 400 response."""
    logger.warning("Request validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Malformed or structurally invalid request.",
            "details": [
                {
                    "loc": [str(loc) for loc in err["loc"]],
                    "msg": err["msg"],
                    "type": err["type"],
                }
                for err in exc.errors()
            ],
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Controlled internal error handler preventing raw stack trace or secret leakage."""
    logger.error("Unhandled error on %s: %s", request.url.path, str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "An internal error occurred during energy optimization.",
            "message": "Optimization pipeline failed in a controlled state.",
        },
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health readiness check",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Return status 'ok' when the service is ready to receive requests."""
    return HealthResponse(status="ok")


@app.post(
    "/optimize-energy",
    response_model=OptimizeEnergyResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize 24-hour campus energy schedule with operator note directives",
    tags=["Optimization"],
)
async def optimize_energy(request: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    """Process operator notes and schedule 24-hour campus energy dispatch.

    1. Extract structured directives from natural-language notes via LLM.
    2. Deterministically validate and normalize directives via guardrails.
    3. Solve the 24-hour Linear Programming energy dispatch via SciPy HiGHS.
    4. Independently replay and verify all constraints.
    5. Roll up authoritative metrics and return the complete schedule.
    """
    logger.info("Received optimization request for scenario: %s", request.scenario_id)

    # Step 1 & 2: LLM Extraction & Deterministic Guardrails
    raw_directives = await extract_directives_from_notes(
        request.operator_notes, request.battery
    )
    sanitized_directives = validate_and_align_interpretations(
        raw_directives, request.operator_notes, request.battery
    )

    # Step 3: HiGHS LP Solver Optimization
    try:
        hourly_plan = solve_energy_dispatch(
            request.hours, request.battery, sanitized_directives
        )
    except OptimizationError as e:
        logger.error("Optimization failed for scenario %s: %s", request.scenario_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization solver failed: {e}",
        )

    # Step 4: 24-Hour Simulation Replay & Constraint Rollup
    try:
        total_grid, total_cost, peak_grid = verify_and_replay_schedule(
            hourly_plan, request.hours, request.battery, sanitized_directives
        )
    except SimulationReplayError as e:
        logger.error("Simulation replay failed for scenario %s: %s", request.scenario_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Schedule replay verification failed: {e}",
        )

    # Step 5: Summary Synthesis & Final Response Construction
    plan_summary = generate_plan_summary(
        sanitized_directives, total_grid, total_cost, peak_grid
    )

    return OptimizeEnergyResponse(
        scenario_id=request.scenario_id,
        directive_interpretation=sanitized_directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=plan_summary,
    )
