#!/usr/bin/env python3
"""
ESG-Audit-Gym: Baseline Inference Script
=========================================
Demonstrates an LLM-driven agent interacting with the ESG-Audit-Gym
environment via the OpenEnv HTTP API.

Reads configuration from environment variables:
  - API_BASE_URL  : LLM API endpoint (default: Groq OpenAI-compatible endpoint)
  - MODEL_NAME    : Model identifier (default: "llama-3.3-70b-versatile")
  - GROQ_API_KEY  : Groq API key
  - ENV_BASE_URL  : ESG-Audit-Gym server URL (default: http://localhost:8000)

Logging format (stdout):
  task=<task_name> env=esg_audit_gym model=<model_name>
  step=<n> action=<str> reward=<0.00> done=<bool> error=<msg|null>
  success=<bool> steps=<n> score=<0.000> rewards=<r1,r2...>
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import ESGAuditClient
from models import Action, ActionType, Observation


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

TASKS: List[str] = [
    "easy_water_consumption",
    "medium_scope1_aggregation",
    "hard_greenwashing_detection",
]


# ═══════════════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS (for LLM function calling)
# ═══════════════════════════════════════════════════════════════════════════

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_page",
            "description": (
                "Navigate to a page in the ESG report. Provide either a "
                "keyword query to search or a direct page number (1-indexed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword query to search across pages.",
                    },
                    "page_number": {
                        "type": "integer",
                        "description": "Direct page number (1-indexed).",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_table",
            "description": (
                "Extract a structured table from the current page. "
                "Specify table_index (0-indexed) if the page has multiple tables."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_index": {
                        "type": "integer",
                        "description": "Zero-indexed table on the current page.",
                        "default": 0,
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_finding",
            "description": (
                "Submit your audited ESG finding. Provide the metric name, "
                "numeric value, unit, and evidence pages. For greenwashing "
                "tasks, also set discrepancy_detected and narrative_assessment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the ESG metric.",
                    },
                    "value": {
                        "type": "number",
                        "description": "Numeric value of the metric.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit (e.g., 'megalitres', 'metric_tons_co2e').",
                    },
                    "evidence_pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Page numbers used as evidence.",
                    },
                    "discrepancy_detected": {
                        "type": "boolean",
                        "description": "Whether a greenwashing discrepancy was found.",
                    },
                    "narrative_assessment": {
                        "type": "string",
                        "description": "Assessment of the discrepancy (if any).",
                    },
                },
                "required": ["metric_name", "value", "unit"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Convert LLM tool call → Action model
# ═══════════════════════════════════════════════════════════════════════════

def tool_call_to_action(func_name: str, arguments: Dict[str, Any]) -> Action:
    """Convert an LLM function-call into a typed Action model."""
    arguments = arguments or {}
    if func_name == "search_page":
        return Action(
            action_type=ActionType.SEARCH_PAGE,
            search_page={"query": arguments.get("query"), "page_number": arguments.get("page_number")},
        )
    elif func_name == "extract_table":
        return Action(
            action_type=ActionType.EXTRACT_TABLE,
            extract_table={"table_index": arguments.get("table_index", 0)},
        )
    elif func_name == "submit_finding":
        return Action(
            action_type=ActionType.SUBMIT_FINDING,
            submit_finding=arguments,
        )
    else:
        raise ValueError(f"Unknown function: {func_name}")


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Format observation as text for the LLM
# ═══════════════════════════════════════════════════════════════════════════

def format_observation(obs: Observation) -> str:
    """Render an observation as a human-readable text block for the LLM."""
    parts: List[str] = []
    parts.append(f"=== Step {obs.steps_taken}/{obs.max_steps} | Score: {obs.score:.3f} ===")
    parts.append(f"Feedback: {obs.feedback}")

    if obs.error:
        parts.append(f"⚠ Error: {obs.error}")

    if obs.current_page:
        p = obs.current_page
        parts.append(f"\n── Page {p.page_number}: {p.title} ──")
        parts.append(p.body_text)
        if p.has_tables:
            parts.append(f"[Page has {len(p.tables)} table(s). Use extract_table to view.]")

    if obs.extracted_table:
        t = obs.extracted_table
        if t.caption:
            parts.append(f"\n── Table: {t.caption} ──")
        header_line = " | ".join(t.headers)
        parts.append(header_line)
        parts.append("-" * len(header_line))
        for row in t.rows:
            parts.append(" | ".join(row))

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN INFERENCE LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_task(
    llm_client: OpenAI,
    env_client: ESGAuditClient,
    task_id: str,
    model_name: str,
) -> Dict[str, Any]:
    """Run a single task episode and return results."""

    # ── Log header ──
    print(f"task={task_id} env=MetricTrace-Gym model={model_name}")

    obs = env_client.reset(task_id=task_id)

    system_prompt = (
        "You are an expert ESG Sustainability Auditor. You are analyzing a "
        "corporate sustainability report to extract and verify ESG metrics.\n\n"
        f"TASK: {obs.task_description}\n\n"
        f"Difficulty: {obs.task_difficulty}\n"
        f"The report has {obs.available_pages} pages.\n"
        f"You have {obs.max_steps} steps maximum.\n\n"
        "Available tools:\n"
        "- search_page: Navigate to pages by keyword or number\n"
        "- extract_table: Extract table data from the current page\n"
        "- submit_finding: Submit your final finding with metric, value, unit\n\n"
        "Strategy:\n"
        "1. Search for relevant pages\n"
        "2. Extract and analyze tables\n"
        "3. Submit your finding with precise values and correct units\n"
        "4. For unit conversions: 1 metric ton = 1000 kg\n"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_observation(obs)},
    ]

    rewards: List[float] = []
    step_count = 0
    last_error: Optional[str] = None

    while not obs.done:
        try:
            response = llm_client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=1024,
            )
        except Exception as exc:
            last_error = f"LLM API error: {exc}"
            print(f"step={step_count + 1} action=error reward=0.00 done=false error={last_error}")
            break

        choice = response.choices[0]

        # Groq strict API compatibility (strips unsupported OpenAI SDK keys)
        safe_msg = choice.message.model_dump(exclude_none=True)
        for key in ["annotations", "audio", "refusal", "function_call"]:
            safe_msg.pop(key, None)

        # If the model doesn't call a tool, prompt it to act
        if not choice.message.tool_calls:
            messages.append({"role": "assistant", "content": choice.message.content or ""})
            messages.append({
                "role": "user",
                "content": "Please use one of the available tools to continue the audit.",
            })
            continue

        # Process the first tool call
        tool_call = choice.message.tool_calls[0]
        func_name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error in tool arguments: {exc}"
            print(f"step={step_count + 1} action={func_name} reward=-0.05 done=false error={last_error}")
            messages.append(safe_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Error: {last_error}. Please provide valid JSON arguments.",
            })
            continue

        try:
            action = tool_call_to_action(func_name, arguments)
        except Exception as exc:
            last_error = f"Action conversion error: {exc}"
            print(f"step={step_count + 1} action={func_name} reward=-0.05 done=false error={last_error}")
            messages.append(safe_msg)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"Error: {last_error}",
            })
            continue

        # Execute in environment
        try:
            obs, reward, done, info = env_client.step(action)
        except Exception as exc:
            last_error = f"Environment step error: {exc}"
            print(f"step={step_count + 1} action={func_name} reward=0.00 done=false error={last_error}")
            break

        step_count += 1
        rewards.append(reward)
        last_error = obs.error

        # ── Log step ──
        done_str = str(done).lower()
        print(
            f"step={step_count} action={func_name} "
            f"reward={reward:.2f} done={done_str} "
            f"error={last_error or 'null'}"
        )

        # Update conversation
        messages.append(safe_msg)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": format_observation(obs),
        })

    # ── Log summary ──
    success = obs.score >= 0.5
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"success={success_str} steps={step_count} "
        f"score={obs.score:.3f} rewards={rewards_str}\n"
    )

    return {
        "task_id": task_id,
        "success": success,
        "steps": step_count,
        "score": obs.score,
        "rewards": rewards,
    }


def main() -> None:
    """Run inference across all tasks."""
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN or GROQ_API_KEY or os.getenv("OPENAI_API_KEY", "dummy"),
    )

    env_client = ESGAuditClient(base_url=ENV_BASE_URL)

    print("=" * 72)
    print("MetricTrace-Gym Baseline Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"LLM API: {API_BASE_URL}")
    print(f"Env Server: {ENV_BASE_URL}")
    print("=" * 72)

    results: List[Dict[str, Any]] = []

    for task_id in TASKS:
        print(f"\n{'─' * 60}")
        try:
            result = run_task(llm_client, env_client, task_id, MODEL_NAME)
            results.append(result)
        except Exception as exc:
            print(f"task={task_id} FATAL ERROR: {exc}")
            results.append({
                "task_id": task_id,
                "success": False,
                "steps": 0,
                "score": 0.0,
                "rewards": [],
            })

    # ── Final Summary ──
    print(f"\n{'═' * 72}")
    print("FINAL RESULTS")
    print(f"{'═' * 72}")
    total_score = 0.0
    for r in results:
        total_score += r["score"]
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(f"  {status} | {r['task_id']:40s} | score={r['score']:.3f} | steps={r['steps']}")
    avg_score = total_score / len(results) if results else 0.0
    print(f"\n  Average Score: {avg_score:.3f}")

    env_client.close()


if __name__ == "__main__":
    main()
