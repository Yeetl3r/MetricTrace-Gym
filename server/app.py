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
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import RedirectResponse, HTMLResponse
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

# ── Front-End Landing Page ──────────────────────────────────────────────

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MetricTrace-Gym</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Fira+Code&display=swap');
        
        :root {
            --bg-dark: #0f172a;
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --glass-bg: rgba(30, 41, 59, 0.7);
            --glass-border: rgba(255, 255, 255, 0.1);
        }
        body {
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-dark);
            color: #f8fafc;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            overflow-x: hidden;
            background: radial-gradient(circle at top right, rgba(16, 185, 129, 0.15), transparent 40%),
                        radial-gradient(circle at bottom left, rgba(59, 130, 246, 0.15), transparent 40%);
        }
        .container {
            max-width: 900px;
            padding: 2rem;
            width: 100%;
            box-sizing: border-box;
            z-index: 10;
        }
        .hero {
            text-align: center;
            margin-bottom: 3rem;
            animation: fadeIn 1s ease-out;
        }
        .title {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -1px;
        }
        .subtitle {
            font-size: 1.25rem;
            color: #94a3b8;
            line-height: 1.6;
            max-width: 700px;
            margin: 0 auto;
        }
        .glass-panel {
            background: var(--glass-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            animation: slideUp 1s ease-out 0.2s both;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }
        .feature-card {
            background: rgba(15, 23, 42, 0.5);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: transform 0.3s ease, border-color 0.3s ease;
        }
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: rgba(16, 185, 129, 0.3);
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 1rem;
        }
        .feature-title {
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: #e2e8f0;
        }
        .feature-desc {
            font-size: 0.9rem;
            color: #94a3b8;
            line-height: 1.5;
        }
        .terminal {
            background: #0f172a;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'Fira Code', 'Monaco', monospace;
            font-size: 0.85rem;
            color: #64748b;
            margin-bottom: 2.5rem;
            border: 1px solid #1e293b;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5);
            height: 150px;
            overflow: hidden;
            position: relative;
        }
        .terminal-header {
            display: flex;
            gap: 6px;
            margin-bottom: 1rem;
        }
        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        .dot.red { background: #ef4444; }
        .dot.yellow { background: #f59e0b; }
        .dot.green { background: #10b981; }
        .log-line {
            margin-bottom: 0.5rem;
            opacity: 0;
            transform: translateY(10px);
            animation: terminalLine 0.5s forwards;
            color: #94a3b8;
        }
        .log-tag { color: #3b82f6; font-weight: bold; }
        .log-value { color: #10b981; }
        
        @keyframes terminalLine {
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideUp {
            from { opacity: 0; transform: translateY(40px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .cta-container {
            text-align: center;
        }
        .btn {
            display: inline-block;
            background: linear-gradient(135deg, var(--accent-green), var(--accent-blue));
            color: white;
            padding: 1rem 2.5rem;
            border-radius: 9999px;
            font-weight: 600;
            text-decoration: none;
            font-size: 1.1rem;
            transition: all 0.3s ease;
            box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
            border: none;
            cursor: pointer;
        }
        .btn:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(16, 185, 129, 0.4);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="hero">
            <h1 class="title">MetricTrace-Gym</h1>
            <p class="subtitle">An elite reinforcement learning environment where AI agents act as Sustainability Auditors, extracting and validating corporate ESG metrics against CSRD frameworks.</p>
        </div>
        
        <div class="glass-panel">
            <div class="features">
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Multi-Page Mapping</div>
                    <div class="feature-desc">Agents traverse synthetic 100+ page corporate reports via complex search tools.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Data Extraction</div>
                    <div class="feature-desc">Dynamically process structured and unstructured matrices to aggregate metric footprints.</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🚨</div>
                    <div class="feature-title">Anti-Greenwashing</div>
                    <div class="feature-desc">Cross-validate CEO narrative claims against deep-document quantitative arrays deterministically.</div>
                </div>
            </div>
            
            <div class="terminal">
                <div class="terminal-header">
                    <div class="dot red"></div>
                    <div class="dot yellow"></div>
                    <div class="dot green"></div>
                </div>
                <div id="log-container">
                    <!-- Logs injected via JS -->
                </div>
            </div>
            
            <div class="cta-container">
                <a href="/docs" class="btn">View API Dashboard</a>
            </div>
        </div>
    </div>

    <script>
        const logs = [
            '<span class="log-tag">[START]</span> Initializing easy_water_consumption environment...',
            '<span class="log-tag">[STEP]</span> action=search_page <span class="log-value">reward=0.10</span> done=False',
            '<span class="log-tag">[STEP]</span> action=extract_table <span class="log-value">reward=0.15</span> done=False',
            '<span class="log-tag">[STEP]</span> action=submit_finding <span class="log-value">reward=0.60</span> done=True',
            '<span class="log-tag">[END]</span> task=easy_water_consumption <span class="log-value">success=true</span> score=0.850'
        ];
        
        const container = document.getElementById('log-container');
        let index = 0;
        
        function appendLog() {
            if (index < logs.length) {
                const div = document.createElement('div');
                div.className = 'log-line';
                div.innerHTML = logs[index];
                container.appendChild(div);
                index++;
                setTimeout(appendLog, 800 + Math.random() * 800);
            } else {
                setTimeout(() => {
                    container.innerHTML = '';
                    index = 0;
                    appendLog();
                }, 3000);
            }
        }
        
        setTimeout(appendLog, 800);
    </script>
</body>
</html>
"""

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
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    environments: Dict[str, ESGAuditEnvironment] = {}

    def get_env(request: Request) -> ESGAuditEnvironment:
        session_id = request.headers.get("X-Session-Id", "default")
        if session_id not in environments:
            environments[session_id] = ESGAuditEnvironment()
        return environments[session_id]

    # ── Health ──
    @app.get("/health", tags=["system"])
    async def health() -> Dict[str, str]:
        return {"status": "healthy", "environment": "esg_audit_gym"}

    # ── Root / Web (Hugging Face UI Redirects) ──
    @app.get("/", include_in_schema=False)
    async def root() -> HTMLResponse:
        return HTMLResponse(content=LANDING_PAGE_HTML, status_code=200)

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
    async def reset(payload: Optional[ResetRequest] = None, env: ESGAuditEnvironment = Depends(get_env)) -> Observation:
        try:
            # Default to easy task if grader sends empty body
            target_task = payload.task_id if payload and payload.task_id else "easy_water_consumption"
            
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
            logger.exception("Reset error: %s", exc)
            raise HTTPException(status_code=500, detail="Internal Server Error: Execution Failed")

    # ── Step ──
    @app.post("/step", response_model=StepResponse, tags=["environment"])
    async def step(action: Action, env: ESGAuditEnvironment = Depends(get_env)) -> StepResponse:
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
            logger.exception("Step error: %s", exc)
            raise HTTPException(status_code=500, detail="Internal Server Error: Execution Failed")

    # ── State ──
    @app.get("/state", response_model=State, tags=["environment"])
    async def get_state(env: ESGAuditEnvironment = Depends(get_env)) -> State:
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
