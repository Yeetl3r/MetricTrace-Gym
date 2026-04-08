"""
ESG-Audit-Gym: Typed Pydantic Models
=====================================
Defines the Action, Observation, and State models for the OpenEnv
3-component design pattern.

Action space:
  - SearchPage: Navigate to a specific page by keyword or page number.
  - ExtractTable: Pull structured table data from the current page.
  - SubmitFinding: Submit an audited ESG finding for grading.

Observation space:
  - Current page content, extracted tables, feedback, and task metadata.

State:
  - Full internal environment state including document store, scores,
    step history, and grading context.
"""

from __future__ import annotations

import enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Action Models
# ---------------------------------------------------------------------------

class ActionType(str, enum.Enum):
    """Enumeration of valid agent actions."""
    SEARCH_PAGE = "search_page"
    EXTRACT_TABLE = "extract_table"
    SUBMIT_FINDING = "submit_finding"


class SearchPageArgs(BaseModel):
    """Arguments for the SearchPage action."""
    query: Optional[str] = Field(
        default=None,
        description="Keyword query to search across pages (e.g., 'water consumption').",
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Direct page number to navigate to (1-indexed).",
    )

    @field_validator("page_number", "query", mode="before")
    @classmethod
    def at_least_one(cls, v: Any, info: Any) -> Any:  # noqa: ANN401
        return v


class ExtractTableArgs(BaseModel):
    """Arguments for the ExtractTable action."""
    table_index: int = Field(
        default=0,
        ge=0,
        description="Zero-indexed table on the current page to extract.",
    )


class SubmitFindingArgs(BaseModel):
    """Arguments for the SubmitFinding action."""
    metric_name: str = Field(
        ...,
        description="Name of the ESG metric (e.g., 'Total Water Consumption').",
    )
    value: float = Field(
        ...,
        description="Numeric value of the metric.",
    )
    unit: str = Field(
        ...,
        description="Unit of the metric (e.g., 'megalitres', 'metric_tons_co2e').",
    )
    evidence_pages: List[int] = Field(
        default_factory=list,
        description="List of page numbers used as evidence.",
    )
    narrative_assessment: Optional[str] = Field(
        default=None,
        description="For hard tasks: assessment of greenwashing discrepancy.",
    )
    discrepancy_detected: Optional[bool] = Field(
        default=None,
        description="For hard tasks: whether a greenwashing discrepancy was found.",
    )


class Action(BaseModel):
    """Top-level action model sent by the agent each step."""
    action_type: ActionType = Field(
        ...,
        description="The type of action to perform.",
    )
    search_page: Optional[SearchPageArgs] = Field(
        default=None,
        description="Arguments when action_type is 'search_page'.",
    )
    extract_table: Optional[ExtractTableArgs] = Field(
        default=None,
        description="Arguments when action_type is 'extract_table'.",
    )
    submit_finding: Optional[SubmitFindingArgs] = Field(
        default=None,
        description="Arguments when action_type is 'submit_finding'.",
    )

    @field_validator("search_page", mode="after")
    @classmethod
    def validate_search_args(cls, v: Optional[SearchPageArgs], info: Any) -> Optional[SearchPageArgs]:
        if info.data.get("action_type") == ActionType.SEARCH_PAGE and v is None:
            raise ValueError("search_page arguments required when action_type is 'search_page'")
        if v is not None and v.query is None and v.page_number is None:
            raise ValueError("SearchPage requires at least one of 'query' or 'page_number'")
        return v

    @field_validator("extract_table", mode="after")
    @classmethod
    def validate_extract_args(cls, v: Optional[ExtractTableArgs], info: Any) -> Optional[ExtractTableArgs]:
        if info.data.get("action_type") == ActionType.EXTRACT_TABLE and v is None:
            raise ValueError("extract_table arguments required when action_type is 'extract_table'")
        return v

    @field_validator("submit_finding", mode="after")
    @classmethod
    def validate_submit_args(cls, v: Optional[SubmitFindingArgs], info: Any) -> Optional[SubmitFindingArgs]:
        if info.data.get("action_type") == ActionType.SUBMIT_FINDING and v is None:
            raise ValueError("submit_finding arguments required when action_type is 'submit_finding'")
        return v


# ---------------------------------------------------------------------------
# Observation Models
# ---------------------------------------------------------------------------

class TableData(BaseModel):
    """Structured table extracted from a document page."""
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    caption: Optional[str] = None


class PageContent(BaseModel):
    """Content of a single page in the document store."""
    page_number: int
    title: str = ""
    body_text: str = ""
    tables: List[TableData] = Field(default_factory=list)
    has_tables: bool = False


class Observation(BaseModel):
    """Observation returned to the agent after each step."""
    current_page: Optional[PageContent] = Field(
        default=None,
        description="Content of the page the agent is currently viewing.",
    )
    extracted_table: Optional[TableData] = Field(
        default=None,
        description="Most recently extracted table data.",
    )
    feedback: str = Field(
        default="",
        description="Natural-language feedback on the last action.",
    )
    task_description: str = Field(
        default="",
        description="Description of the current task objective.",
    )
    task_difficulty: str = Field(
        default="easy",
        description="Difficulty level: 'easy', 'medium', or 'hard'.",
    )
    available_pages: int = Field(
        default=0,
        description="Total number of pages in the document store.",
    )
    steps_taken: int = Field(
        default=0,
        description="Number of steps the agent has taken so far.",
    )
    max_steps: int = Field(
        default=20,
        description="Maximum allowed steps for this episode.",
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Current cumulative normalized score (0.0–1.0).",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the last action was invalid.",
    )


# ---------------------------------------------------------------------------
# State Models (full internal environment state)
# ---------------------------------------------------------------------------

class StepRecord(BaseModel):
    """Record of a single step for audit trail."""
    step_number: int
    action: Action
    reward: float
    cumulative_reward: float
    feedback: str
    error: Optional[str] = None


class GroundTruth(BaseModel):
    """Ground-truth answer for programmatic grading."""
    metric_name: str
    value: float
    unit: str
    tolerance: float = Field(
        default=0.01,
        description="Relative tolerance for numeric comparison.",
    )
    evidence_pages: List[int] = Field(default_factory=list)
    discrepancy_detected: Optional[bool] = None
    narrative_keywords: List[str] = Field(default_factory=list)


class TaskConfig(BaseModel):
    """Configuration for a single task."""
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    ground_truth: GroundTruth
    max_steps: int = 20
    reward_weights: Dict[str, float] = Field(default_factory=dict)


class State(BaseModel):
    """Full internal state of the ESG-Audit-Gym environment."""
    task_config: TaskConfig
    document_pages: List[PageContent] = Field(default_factory=list)
    current_page_index: Optional[int] = None
    steps: List[StepRecord] = Field(default_factory=list)
    cumulative_reward: float = 0.0
    done: bool = False
    submitted: bool = False
    final_score: float = 0.0
    pages_visited: List[int] = Field(default_factory=list)
    tables_extracted: List[int] = Field(default_factory=list)
    correct_pages_found: bool = False
    correct_table_extracted: bool = False
    partial_value_match: bool = False
