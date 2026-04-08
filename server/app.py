"""
ESG-Audit-Gym: FastAPI Application Factory
===========================================
Implements the OpenEnv HTTP API with endpoints for:
  POST /reset     — Start a new episode
  POST /step      — Execute an action
  GET  /state     — Retrieve full internal state
  GET  /tasks     — List available tasks
  GET  /health    — Health check
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError

try:
    from ..models import Action, Observation, State
    from .environment import ESGAuditEnvironment
except (ImportError, ValueError):
    from models import Action, Observation, State
    from server.environment import ESGAuditEnvironment

logger = logging.getLogger("esg_audit_gym")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# ── Request / Response Schemas ──────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = Field(None, description="Task identifier to start.")


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    detail: str


# ── Application Factory ────────────────────────────────────────────────

def create_app() -> FastAPI:
    """FastAPI application factory for ESG-Audit-Gym."""

    app = FastAPI(
        title="ESG-Audit-Gym",
        description=(
            "OpenEnv-compliant RL environment for ESG sustainability auditing. "
            "An AI agent navigates corporate ESG reports, extracts metrics, "
            "and validates them against CSRD/ISSB regulatory frameworks."
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    env = ESGAuditEnvironment()

    # ── Health ──
    @app.get("/health", tags=["system"])
    async def health() -> Dict[str, str]:
        return {"status": "healthy", "environment": "esg_audit_gym"}

    # ── Root / Web (Hugging Face UI Redirects) ──
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get("/web", include_in_schema=False)
    async def web_ui() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    # ── Metadata ──
    @app.get("/metadata", tags=["system"])
    async def metadata() -> Dict[str, str]:
        return {
            "name": "MetricTrace-Gym",
            "description": "OpenEnv-compliant RL environment for ESG sustainability auditing.",
        }

    # ── Schema ──
    @app.get("/schema", tags=["system"])
    async def get_schema() -> Dict[str, Any]:
        return {
            "action": Action.model_json_schema(),
            "observation": Observation.model_json_schema(),
            "state": State.model_json_schema()
        }

    # ── MCP ──
    @app.post("/mcp", tags=["system"])
    async def mcp_endpoint(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:
            body = {}
        return {
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "result": {"status": "ok"}
        }

    # ── List Tasks ──
    @app.get("/tasks", tags=["environment"])
    async def list_tasks() -> list:
        return ESGAuditEnvironment.list_tasks()

    # ── Reset ──
    @app.post("/reset", response_model=Observation, tags=["environment"])
    async def reset(request: Optional[ResetRequest] = None) -> Observation:
        try:
            # Default to easy task if grader sends empty body
            target_task = request.task_id if request and request.task_id else "easy_water_consumption"
            
            obs = env.reset(task_id=target_task)
            logger.info(
                "Episode reset | task=%s difficulty=%s",
                target_task,
                obs.task_difficulty,
            )
            return obs
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Reset error: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Step ──
    @app.post("/step", response_model=StepResponse, tags=["environment"])
    async def step(action: Action) -> StepResponse:
        try:
            obs, reward, done, info = env.step(action)
            logger.info(
                "step=%d action=%s reward=%.3f done=%s score=%.3f",
                info.get("step", 0),
                action.action_type.value,
                reward,
                done,
                info.get("score", 0.0),
            )
            return StepResponse(
                observation=obs,
                reward=reward,
                done=done,
                info=info,
            )
        except ValidationError as exc:
            logger.warning("Malformed action: %s", exc)
            raise HTTPException(
                status_code=422,
                detail=f"Malformed action arguments: {exc.error_count()} validation error(s). {str(exc)}",
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Step error: %s\n%s", exc, traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(exc))

    # ── State ──
    @app.get("/state", response_model=State, tags=["environment"])
    async def get_state() -> State:
        try:
            return env.state()
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return app


# ── Entrypoint for `uvicorn server.app:app` ─────────────────────────────
app = create_app()

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
