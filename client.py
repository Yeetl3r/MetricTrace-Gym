"""
ESG-Audit-Gym: EnvClient (Client-Side Wrapper)
===============================================
Thin HTTP client that wraps the server API, providing a Pythonic
interface for agents and inference scripts.

Usage:
    from client import ESGAuditClient

    client = ESGAuditClient(base_url="http://localhost:8000")
    obs = client.reset(task_id="easy_water_consumption")
    obs, reward, done, info = client.step(action)
    state = client.state()
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import httpx

from models import Action, Observation, State


class ESGAuditClient:
    """Client-side EnvClient wrapper for ESG-Audit-Gym."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.Client(base_url=self._base_url, timeout=self._timeout)

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._client.close()

    def __enter__(self) -> "ESGAuditClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── API Methods ────────────────────────────────────────────────────────

    def health(self) -> Dict[str, str]:
        """Check server health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> List[Dict[str, str]]:
        """List available tasks."""
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str) -> Observation:
        """Reset the environment for the given task."""
        resp = self._client.post("/reset", json={"task_id": task_id})
        resp.raise_for_status()
        return Observation.model_validate(resp.json())

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute a single step in the environment."""
        resp = self._client.post(
            "/step",
            json=action.model_dump(mode="json"),
        )
        resp.raise_for_status()
        data = resp.json()
        obs = Observation.model_validate(data["observation"])
        return obs, data["reward"], data["done"], data.get("info", {})

    def state(self) -> State:
        """Get the full internal state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        return State.model_validate(resp.json())
