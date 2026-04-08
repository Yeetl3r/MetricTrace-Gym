"""
ESG-Audit-Gym: Server-Side Environment Logic
=============================================
Core RL environment implementing reset(), step(), and state() with:
  - Simulated corporate ESG document store (multi-page, multi-table)
  - Three graded tasks (easy / medium / hard)
  - Dense, potential-based scalar rewards
  - Deterministic programmatic grading (0.0–1.0)
"""

from __future__ import annotations

import copy
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..models import (
        Action,
        ActionType,
        GroundTruth,
        Observation,
        PageContent,
        State,
        StepRecord,
        TableData,
        TaskConfig,
    )
except (ImportError, ValueError):
    from models import (
        Action,
        ActionType,
        GroundTruth,
        Observation,
        PageContent,
        State,
        StepRecord,
        TableData,
        TaskConfig,
    )


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENT STORE — Simulated ESG Sustainability Report
# ═══════════════════════════════════════════════════════════════════════════

def _build_document_store() -> List[PageContent]:
    """Build a realistic 12-page ESG sustainability report."""
    pages: List[PageContent] = []

    # Page 1: Cover
    pages.append(PageContent(
        page_number=1,
        title="FY2024 Sustainability Report — GreenCorp Industries",
        body_text=(
            "GreenCorp Industries Annual Sustainability Report for the fiscal "
            "year ending December 31, 2024. Prepared in accordance with the "
            "Corporate Sustainability Reporting Directive (CSRD) and aligned "
            "with ISSB S1/S2 standards. This report covers environmental, "
            "social, and governance metrics across all global operations."
        ),
        tables=[],
        has_tables=False,
    ))

    # Page 2: CEO Letter / Narrative Claims
    pages.append(PageContent(
        page_number=2,
        title="Letter from the CEO",
        body_text=(
            "Dear Stakeholders, I am proud to report that GreenCorp has made "
            "significant strides in sustainability this year. We achieved a "
            "10% reduction in our total greenhouse gas emissions compared to "
            "the prior year. Our Scope 1 emissions decreased markedly thanks "
            "to our transition to renewable energy at three manufacturing "
            "facilities. Water consumption intensity dropped by 15%, and our "
            "social programs now reach over 50,000 community members. "
            "We remain committed to net-zero by 2040."
        ),
        tables=[],
        has_tables=False,
    ))

    # Page 3: Environmental Summary
    pages.append(PageContent(
        page_number=3,
        title="Environmental Performance Summary",
        body_text=(
            "This section provides a high-level overview of GreenCorp's "
            "environmental performance for FY2024. Detailed breakdowns of "
            "emissions, energy usage, water consumption, and waste management "
            "are provided in subsequent sections."
        ),
        tables=[TableData(
            headers=["Metric", "FY2023", "FY2024", "Unit", "Change (%)"],
            rows=[
                ["Total GHG Emissions", "125,000", "128,750", "metric tons CO2e", "+3.0%"],
                ["Scope 1 Emissions", "45,000", "47,250", "metric tons CO2e", "+5.0%"],
                ["Scope 2 Emissions", "60,000", "58,200", "metric tons CO2e", "-3.0%"],
                ["Scope 3 Emissions", "20,000", "23,300", "metric tons CO2e", "+16.5%"],
                ["Total Water Consumption", "8.5", "7.2", "megalitres", "-15.3%"],
                ["Total Energy Use", "1,200,000", "1,150,000", "MWh", "-4.2%"],
                ["Waste Generated", "15,000", "14,200", "metric tons", "-5.3%"],
            ],
            caption="Table 3.1: Environmental KPIs Year-over-Year",
        )],
        has_tables=True,
    ))

    # Page 4: Scope 1 Emissions Detail — Facility Breakdown
    pages.append(PageContent(
        page_number=4,
        title="Scope 1 Emissions — Facility Breakdown",
        body_text=(
            "Scope 1 emissions comprise direct GHG emissions from owned or "
            "controlled sources. Below is the breakdown by facility for FY2024. "
            "All values are reported in metric tons of CO2 equivalent (CO2e) "
            "unless otherwise indicated."
        ),
        tables=[TableData(
            headers=["Facility", "Region", "Emissions (tCO2e)", "Fuel Type"],
            rows=[
                ["Houston Refinery", "North America", "18,500", "Natural Gas"],
                ["Rotterdam Plant", "Europe", "12,300", "Diesel / Gas"],
                ["Shanghai Complex", "Asia-Pacific", "9,200", "Coal / Gas"],
                ["Lagos Terminal", "Africa", "4,750", "Diesel"],
            ],
            caption="Table 4.1: Scope 1 Emissions by Facility (FY2024)",
        )],
        has_tables=True,
    ))

    # Page 5: Scope 1 Emissions Detail — Mobile Combustion
    pages.append(PageContent(
        page_number=5,
        title="Scope 1 Emissions — Mobile & Fugitive Sources",
        body_text=(
            "In addition to stationary combustion, GreenCorp reports mobile "
            "combustion from its owned fleet and fugitive emissions from "
            "refrigerant leaks and process venting. Note: fleet emissions "
            "are reported in kilograms of CO2e and must be converted."
        ),
        tables=[TableData(
            headers=["Source", "Emissions", "Unit"],
            rows=[
                ["Company Fleet — Trucks", "1,850,000", "kg CO2e"],
                ["Company Fleet — Aviation", "650,000", "kg CO2e"],
                ["Fugitive — Refrigerants", "120", "metric tons CO2e"],
                ["Fugitive — Process Venting", "80", "metric tons CO2e"],
            ],
            caption="Table 5.1: Mobile and Fugitive Scope 1 Emissions (FY2024)",
        )],
        has_tables=True,
    ))

    # Page 6: Water Consumption
    pages.append(PageContent(
        page_number=6,
        title="Water Stewardship",
        body_text=(
            "GreenCorp is committed to responsible water management. "
            "Total water withdrawal decreased by 15.3% year-over-year, "
            "driven by closed-loop cooling systems installed at the "
            "Houston and Rotterdam facilities."
        ),
        tables=[TableData(
            headers=["Metric", "Value", "Unit"],
            rows=[
                ["Total Water Consumption", "7.2", "megalitres"],
                ["Water Recycled", "3.1", "megalitres"],
                ["Water Discharged", "4.1", "megalitres"],
                ["Water Intensity", "0.006", "megalitres / $M revenue"],
            ],
            caption="Table 6.1: Water Consumption Metrics (FY2024)",
        )],
        has_tables=True,
    ))

    # Page 7: Energy
    pages.append(PageContent(
        page_number=7,
        title="Energy Management",
        body_text=(
            "Total energy consumption was 1,150,000 MWh. Renewable energy "
            "accounted for 34% of total consumption, up from 28% in FY2023."
        ),
        tables=[TableData(
            headers=["Source", "MWh", "% of Total"],
            rows=[
                ["Natural Gas", "520,000", "45.2%"],
                ["Grid Electricity", "239,000", "20.8%"],
                ["Solar PV", "195,500", "17.0%"],
                ["Wind", "150,000", "13.0%"],
                ["Other Renewables", "45,500", "4.0%"],
            ],
            caption="Table 7.1: Energy Consumption by Source (FY2024)",
        )],
        has_tables=True,
    ))

    # Page 8: Social — Workforce
    pages.append(PageContent(
        page_number=8,
        title="Social — Workforce & Diversity",
        body_text=(
            "GreenCorp employs 24,500 full-time employees globally. "
            "Gender diversity in leadership increased to 38% women, "
            "exceeding our 2024 target of 35%."
        ),
        tables=[TableData(
            headers=["Metric", "FY2023", "FY2024"],
            rows=[
                ["Total Employees", "23,100", "24,500"],
                ["Women in Leadership (%)", "32%", "38%"],
                ["Employee Turnover (%)", "14.2%", "11.8%"],
                ["Training Hours per Employee", "42", "48"],
                ["LTIR (Lost Time Injury Rate)", "0.85", "0.72"],
            ],
            caption="Table 8.1: Workforce KPIs",
        )],
        has_tables=True,
    ))

    # Page 9: Social — Community
    pages.append(PageContent(
        page_number=9,
        title="Social — Community Investment",
        body_text=(
            "GreenCorp invested $12.4M in community programs in FY2024, "
            "a 22% increase from the prior year. Programs span education, "
            "health, and environmental restoration."
        ),
        tables=[],
        has_tables=False,
    ))

    # Page 10: Governance
    pages.append(PageContent(
        page_number=10,
        title="Governance — Board & Ethics",
        body_text=(
            "The Board of Directors comprises 12 members, with 5 independent "
            "directors. The ESG Committee meets quarterly. Zero material "
            "compliance violations were reported in FY2024."
        ),
        tables=[TableData(
            headers=["Metric", "Value"],
            rows=[
                ["Board Size", "12"],
                ["Independent Directors", "5"],
                ["Board Gender Diversity (%)", "33%"],
                ["ESG Committee Meetings", "4"],
                ["Material Compliance Violations", "0"],
                ["Anti-Corruption Training Completion (%)", "98%"],
            ],
            caption="Table 10.1: Governance KPIs",
        )],
        has_tables=True,
    ))

    # Page 11: CSRD Alignment Index
    pages.append(PageContent(
        page_number=11,
        title="CSRD & ISSB Alignment Index",
        body_text=(
            "This section maps GreenCorp's reported metrics to the EU's "
            "Corporate Sustainability Reporting Directive (CSRD) European "
            "Sustainability Reporting Standards (ESRS) and ISSB S1/S2 "
            "disclosure requirements."
        ),
        tables=[TableData(
            headers=["ESRS Standard", "Disclosure", "Reported", "Page Ref"],
            rows=[
                ["E1", "Climate Change — Scope 1", "Yes", "4, 5"],
                ["E1", "Climate Change — Scope 2", "Yes", "3"],
                ["E1", "Climate Change — Scope 3", "Yes", "3"],
                ["E3", "Water & Marine Resources", "Yes", "6"],
                ["S1", "Own Workforce", "Yes", "8"],
                ["G1", "Business Conduct", "Yes", "10"],
            ],
            caption="Table 11.1: CSRD/ESRS Alignment Matrix",
        )],
        has_tables=True,
    ))

    # Page 12: Appendix — Methodology
    pages.append(PageContent(
        page_number=12,
        title="Appendix: Methodology & Assurance",
        body_text=(
            "Emission factors sourced from DEFRA 2024 and EPA eGRID. "
            "GWP values use AR6 100-year potentials. Water data measured "
            "via on-site flow meters with ±2% accuracy. Independent "
            "limited assurance provided by Deloitte LLP."
        ),
        tables=[],
        has_tables=False,
    ))

    return pages


# ═══════════════════════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

TASKS: Dict[str, TaskConfig] = {
    # --- EASY ---------------------------------------------------------------
    "easy_water_consumption": TaskConfig(
        task_id="easy_water_consumption",
        task_name="Extract Total Water Consumption",
        difficulty="easy",
        description=(
            "Extract the 'Total Water Consumption' for FY2024 from the "
            "sustainability report. The metric is explicitly labeled in "
            "a structured table. Report the value and its unit."
        ),
        ground_truth=GroundTruth(
            metric_name="Total Water Consumption",
            value=7.2,
            unit="megalitres",
            tolerance=0.01,
            evidence_pages=[6],
        ),
        max_steps=10,
        reward_weights={
            "correct_page": 0.15,
            "correct_table": 0.15,
            "correct_value": 0.40,
            "correct_unit": 0.15,
            "correct_schema": 0.15,
        },
    ),

    # --- MEDIUM -------------------------------------------------------------
    "medium_scope1_aggregation": TaskConfig(
        task_id="medium_scope1_aggregation",
        task_name="Aggregate Scope 1 Emissions",
        difficulty="medium",
        description=(
            "Calculate the total Scope 1 GHG emissions for FY2024 by "
            "aggregating data from multiple tables across the report. "
            "You must find facility-level stationary combustion data, "
            "mobile combustion data (note: fleet emissions are in kg CO2e "
            "and need conversion to metric tons), and fugitive emissions. "
            "Report the total in metric tons CO2e."
        ),
        ground_truth=GroundTruth(
            metric_name="Scope 1 Emissions",
            # Facilities: 18500 + 12300 + 9200 + 4750 = 44750
            # Fleet: (1850000 + 650000) / 1000 = 2500
            # Fugitive: 120 + 80 = 200
            # Total: 44750 + 2500 + 200 = 47450
            # Note: The summary table on page 3 says 47,250 — a deliberate
            # rounding that the agent should flag or reconcile with detail.
            value=47450.0,
            unit="metric_tons_co2e",
            tolerance=0.02,  # Allow 2% tolerance for rounding approaches
            evidence_pages=[4, 5],
        ),
        max_steps=15,
        reward_weights={
            "found_facility_page": 0.10,
            "found_mobile_page": 0.10,
            "extracted_facility_table": 0.10,
            "extracted_mobile_table": 0.10,
            "unit_conversion": 0.15,
            "correct_value": 0.30,
            "correct_unit": 0.10,
            "correct_schema": 0.05,
        },
    ),

    # --- HARD ---------------------------------------------------------------
    "hard_greenwashing_detection": TaskConfig(
        task_id="hard_greenwashing_detection",
        task_name="Detect Greenwashing Discrepancy",
        difficulty="hard",
        description=(
            "Cross-validate the CEO's narrative claims against the "
            "quantitative data in the report. The CEO claims a '10% "
            "reduction in total greenhouse gas emissions.' Verify this "
            "claim against the actual emissions data. Report the actual "
            "Total GHG Emissions for FY2024, the correct unit, and "
            "whether a greenwashing discrepancy exists. The narrative "
            "claims a reduction, but the data may tell a different story."
        ),
        ground_truth=GroundTruth(
            metric_name="Total GHG Emissions",
            value=128750.0,
            unit="metric_tons_co2e",
            tolerance=0.01,
            evidence_pages=[2, 3],
            discrepancy_detected=True,
            narrative_keywords=["increase", "greenwashing", "discrepancy", "misleading"],
        ),
        max_steps=20,
        reward_weights={
            "found_ceo_page": 0.05,
            "found_data_page": 0.10,
            "extracted_data_table": 0.10,
            "correct_value": 0.20,
            "correct_unit": 0.05,
            "discrepancy_flag": 0.25,
            "narrative_quality": 0.15,
            "correct_schema": 0.10,
        },
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# ENVIRONMENT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class ESGAuditEnvironment:
    """
    Stateful RL environment for ESG auditing.

    Implements the OpenEnv 3-API contract:
        reset(task_id) -> Observation
        step(action)   -> (Observation, reward, done, info)
        state()        -> State
    """

    def __init__(self) -> None:
        self._state: Optional[State] = None
        self._document_store: List[PageContent] = _build_document_store()

    # ── reset ──────────────────────────────────────────────────────────────

    def reset(self, task_id: str) -> Observation:
        """Initialise a new episode for the given task."""
        if task_id not in TASKS:
            valid = ", ".join(TASKS.keys())
            raise ValueError(f"Unknown task_id '{task_id}'. Valid tasks: {valid}")

        task_config = copy.deepcopy(TASKS[task_id])
        self._state = State(
            task_config=task_config,
            document_pages=copy.deepcopy(self._document_store),
            current_page_index=None,
            steps=[],
            cumulative_reward=0.0,
            done=False,
            submitted=False,
            final_score=0.0,
            pages_visited=[],
            tables_extracted=[],
            correct_pages_found=False,
            correct_table_extracted=False,
            partial_value_match=False,
        )

        return Observation(
            current_page=None,
            extracted_table=None,
            feedback="Episode started. Use actions to navigate the document and extract ESG metrics.",
            task_description=task_config.description,
            task_difficulty=task_config.difficulty,
            available_pages=len(self._document_store),
            steps_taken=0,
            max_steps=task_config.max_steps,
            score=0.0,
            done=False,
            error=None,
        )

    # ── step ───────────────────────────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Execute one action and return (observation, reward, done, info)."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        state = self._state
        reward: float = 0.0
        feedback: str = ""
        error: Optional[str] = None
        current_page: Optional[PageContent] = None
        extracted_table: Optional[TableData] = None

        try:
            if action.action_type == ActionType.SEARCH_PAGE:
                current_page, reward, feedback, error = self._handle_search(action)
            elif action.action_type == ActionType.EXTRACT_TABLE:
                extracted_table, reward, feedback, error = self._handle_extract(action)
                current_page = (
                    state.document_pages[state.current_page_index]
                    if state.current_page_index is not None
                    else None
                )
            elif action.action_type == ActionType.SUBMIT_FINDING:
                reward, feedback, error = self._handle_submit(action)
                current_page = (
                    state.document_pages[state.current_page_index]
                    if state.current_page_index is not None
                    else None
                )
            else:
                error = f"Unknown action type: {action.action_type}"
                reward = -0.05
                feedback = error
        except Exception as exc:
            error = f"Malformed action: {str(exc)}"
            reward = -0.05
            feedback = f"Action error: {error}"

        # Penalty for invalid tool calls
        if error is not None and reward >= 0:
            reward = -0.05

        state.cumulative_reward += reward

        # Step budget exhaustion
        step_num = len(state.steps) + 1
        if step_num >= state.task_config.max_steps and not state.done:
            state.done = True
            feedback += " | Step limit reached. Episode terminated."

        # Record step
        state.steps.append(StepRecord(
            step_number=step_num,
            action=action,
            reward=reward,
            cumulative_reward=state.cumulative_reward,
            feedback=feedback,
            error=error,
        ))

        # Compute normalized score
        state.final_score = self._compute_score()

        info: Dict[str, Any] = {
            "step": step_num,
            "reward": reward,
            "cumulative_reward": state.cumulative_reward,
            "score": state.final_score,
        }

        obs = Observation(
            current_page=current_page,
            extracted_table=extracted_table,
            feedback=feedback,
            task_description=state.task_config.description,
            task_difficulty=state.task_config.difficulty,
            available_pages=len(state.document_pages),
            steps_taken=step_num,
            max_steps=state.task_config.max_steps,
            score=state.final_score,
            done=state.done,
            error=error,
        )

        return obs, reward, state.done, info

    # ── state ──────────────────────────────────────────────────────────────

    def state(self) -> State:
        """Return a deep copy of the current internal state."""
        if self._state is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return copy.deepcopy(self._state)

    # ── list_tasks ─────────────────────────────────────────────────────────

    @staticmethod
    def list_tasks() -> List[Dict[str, str]]:
        """Return metadata for all available tasks."""
        return [
            {
                "task_id": t.task_id,
                "task_name": t.task_name,
                "difficulty": t.difficulty,
                "description": t.description,
            }
            for t in TASKS.values()
        ]

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE: Action Handlers
    # ═══════════════════════════════════════════════════════════════════════

    def _handle_search(
        self, action: Action
    ) -> Tuple[Optional[PageContent], float, str, Optional[str]]:
        assert self._state is not None
        args = action.search_page
        if args is None:
            return None, -0.05, "", "search_page args missing"

        state = self._state
        page: Optional[PageContent] = None
        reward = 0.0
        feedback = ""
        error: Optional[str] = None

        if args.page_number is not None:
            idx = args.page_number - 1
            if 0 <= idx < len(state.document_pages):
                page = state.document_pages[idx]
                state.current_page_index = idx
            else:
                error = f"Page {args.page_number} out of range (1–{len(state.document_pages)})"
                return None, -0.05, f"Invalid page number.", error

        elif args.query is not None:
            query_lower = args.query.lower()
            query_words = query_lower.split()
            best_score = 0.0
            best_page: Optional[PageContent] = None
            best_idx = -1
            for i, p in enumerate(state.document_pages):
                # Build searchable text from title, body, and all tables
                title_text = p.title.lower()
                body_text = p.body_text.lower()
                table_text_parts: List[str] = []
                for t in p.tables:
                    if t.caption:
                        table_text_parts.append(t.caption.lower())
                    table_text_parts.extend(h.lower() for h in t.headers)
                    for row in t.rows:
                        table_text_parts.extend(cell.lower() for cell in row)
                table_text = " ".join(table_text_parts)

                # Title matches weighted 3x, table matches 2x, body 1x
                score = 0.0
                for w in query_words:
                    if w in title_text:
                        score += 3.0
                    if w in body_text:
                        score += 1.0
                    if w in table_text:
                        score += 2.0

                if score > best_score:
                    best_score = score
                    best_page = p
                    best_idx = i
            if best_page is not None:
                page = best_page
                state.current_page_index = best_idx
            else:
                feedback = "No pages matched your query."
                return None, -0.02, feedback, None

        if page is not None:
            state.pages_visited.append(page.page_number)
            feedback = f"Navigated to page {page.page_number}: '{page.title}'"

            # Reward for finding evidence pages
            gt = state.task_config.ground_truth
            if page.page_number in gt.evidence_pages:
                if not state.correct_pages_found:
                    reward += 0.10
                    feedback += " [+0.10: relevant evidence page found]"
                    # Check if ALL evidence pages found
                    if all(ep in state.pages_visited for ep in gt.evidence_pages):
                        state.correct_pages_found = True
                        reward += 0.05
                        feedback += " [+0.05: all evidence pages located]"

        return page, reward, feedback, error

    def _handle_extract(
        self, action: Action
    ) -> Tuple[Optional[TableData], float, str, Optional[str]]:
        assert self._state is not None
        args = action.extract_table
        if args is None:
            return None, -0.05, "", "extract_table args missing"

        state = self._state
        if state.current_page_index is None:
            return None, -0.05, "No page selected. Use SearchPage first.", "No current page"

        current = state.document_pages[state.current_page_index]
        if not current.has_tables or len(current.tables) == 0:
            return (
                None, -0.02,
                f"Page {current.page_number} has no tables.",
                None,
            )

        idx = args.table_index
        if idx < 0 or idx >= len(current.tables):
            return (
                None, -0.05,
                f"Table index {idx} out of range. Page has {len(current.tables)} table(s).",
                f"Invalid table_index {idx}",
            )

        table = current.tables[idx]
        state.tables_extracted.append(current.page_number)

        reward = 0.05  # Small reward for any valid extraction
        feedback = f"Extracted table from page {current.page_number}: '{table.caption or 'Untitled'}'"

        # Check if this is a relevant evidence page
        gt = state.task_config.ground_truth
        if current.page_number in gt.evidence_pages:
            if not state.correct_table_extracted:
                reward += 0.15
                state.correct_table_extracted = True
                feedback += " [+0.15: extracted table from evidence page]"

        return table, reward, feedback, None

    def _handle_submit(
        self, action: Action
    ) -> Tuple[float, str, Optional[str]]:
        assert self._state is not None
        args = action.submit_finding
        if args is None:
            return -0.05, "", "submit_finding args missing"

        state = self._state
        gt = state.task_config.ground_truth

        reward = 0.0
        feedback_parts: List[str] = []

        # ── Schema validation reward ──
        reward += 0.10
        feedback_parts.append("+0.10 valid submission schema")

        # ── Metric name match ──
        name_match = _fuzzy_match(args.metric_name, gt.metric_name)
        if name_match:
            reward += 0.05
            feedback_parts.append("+0.05 metric name match")

        # ── Unit match ──
        unit_match = _unit_equivalent(args.unit, gt.unit)
        if unit_match:
            reward += 0.10
            feedback_parts.append("+0.10 correct unit")
        else:
            feedback_parts.append("-0.00 incorrect unit")

        # ── Value match ──
        if gt.value != 0:
            relative_error = abs(args.value - gt.value) / abs(gt.value)
        else:
            relative_error = abs(args.value - gt.value)

        if relative_error <= gt.tolerance:
            reward += 0.35
            feedback_parts.append(f"+0.35 exact value match (error={relative_error:.4f})")
            state.partial_value_match = True
        elif relative_error <= 0.05:
            reward += 0.20
            feedback_parts.append(f"+0.20 close value (error={relative_error:.4f})")
            state.partial_value_match = True
        elif relative_error <= 0.15:
            reward += 0.10
            feedback_parts.append(f"+0.10 approximate value (error={relative_error:.4f})")
        else:
            feedback_parts.append(f"+0.00 value too far (error={relative_error:.4f})")

        # ── Evidence pages ──
        if args.evidence_pages:
            matching = set(args.evidence_pages) & set(gt.evidence_pages)
            if matching:
                ep_reward = 0.05 * (len(matching) / len(gt.evidence_pages))
                reward += ep_reward
                feedback_parts.append(f"+{ep_reward:.2f} evidence pages")

        # ── Greenwashing detection (hard task only) ──
        if gt.discrepancy_detected is not None:
            if args.discrepancy_detected == gt.discrepancy_detected:
                reward += 0.20
                feedback_parts.append("+0.20 correct discrepancy detection")
            else:
                reward -= 0.10
                feedback_parts.append("-0.10 incorrect discrepancy assessment")

            # Narrative quality
            if args.narrative_assessment and gt.narrative_keywords:
                assessment_lower = args.narrative_assessment.lower()
                hits = sum(1 for kw in gt.narrative_keywords if kw in assessment_lower)
                if hits > 0:
                    nq_reward = 0.10 * min(hits / len(gt.narrative_keywords), 1.0)
                    reward += nq_reward
                    feedback_parts.append(f"+{nq_reward:.2f} narrative quality")

        state.submitted = True
        state.done = True
        feedback = "SUBMISSION GRADED | " + " | ".join(feedback_parts)

        return reward, feedback, None

    # ═══════════════════════════════════════════════════════════════════════
    # PRIVATE: Scoring / Grading
    # ═══════════════════════════════════════════════════════════════════════

    def _compute_score(self) -> float:
        """Deterministic 0.0–1.0 score based on cumulative reward."""
        if self._state is None:
            return 0.0
        state = self._state

        # Max possible reward varies by task but we normalise to 1.0
        max_possible = self._max_possible_reward()
        if max_possible <= 0:
            return 0.0

        raw = state.cumulative_reward / max_possible
        return round(max(0.0, min(1.0, raw)), 3)

    def _max_possible_reward(self) -> float:
        """Calculate the theoretical maximum reward for the current task."""
        if self._state is None:
            return 1.0
        difficulty = self._state.task_config.difficulty
        if difficulty == "easy":
            # Page found(0.15) + all pages(0.05) + extract(0.20) + submit(0.65)
            return 1.05
        elif difficulty == "medium":
            # 2 pages found(0.20) + all pages(0.05) + 2 extracts(0.40) + submit(0.65)
            return 1.15
        else:
            # 2 pages(0.20) + all pages(0.05) + extract(0.20) + submit(0.90)
            return 1.35


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _fuzzy_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Simple token-overlap fuzzy match."""
    tokens_a = set(re.sub(r"[^a-z0-9 ]", " ", a.lower()).split())
    tokens_b = set(re.sub(r"[^a-z0-9 ]", " ", b.lower()).split())
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b)
    return (overlap / max(len(tokens_a), len(tokens_b))) >= threshold


_UNIT_ALIASES: Dict[str, str] = {
    "metric_tons_co2e": "tco2e",
    "metric tons co2e": "tco2e",
    "tco2e": "tco2e",
    "tonnes co2e": "tco2e",
    "t co2e": "tco2e",
    "mt co2e": "tco2e",
    "megalitres": "ml",
    "megaliters": "ml",
    "megalitres": "ml",
    "ml": "ml",
    "mwh": "mwh",
    "metric tons": "mt",
    "metric_tons": "mt",
    "tonnes": "mt",
    "kg co2e": "kgco2e",
    "kg_co2e": "kgco2e",
    "kgco2e": "kgco2e",
}


def _unit_equivalent(a: str, b: str) -> bool:
    """Check if two unit strings are semantically equivalent."""
    norm_a = _UNIT_ALIASES.get(a.lower().strip(), a.lower().strip())
    norm_b = _UNIT_ALIASES.get(b.lower().strip(), b.lower().strip())
    return norm_a == norm_b
