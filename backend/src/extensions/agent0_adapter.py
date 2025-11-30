"""
Optional integration helpers for Agent0 (https://github.com/aiming-lab/Agent0).

The real Agent0 project provides an agent runtime that can chain tools and
memories together. This adapter keeps the interface lightweight so that the
SecOps AI backend can:
  - declare the skills/tools it wants an agent to use,
  - build a short action plan from those skills,
  - capture telemetry about how each step performed.

No external dependencies are required; you can plug in a real Agent0 runner
later by swapping out the execution logic. The goal is to keep the data shapes
and terminology aligned with Agent0 so the rest of the codebase can evolve
without large refactors when the runtime is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

TelemetryHook = Callable[["Agent0Step"], None]


@dataclass
class Agent0Skill:
    """Metadata describing a capability the agent can invoke."""

    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    cost_hint: Optional[float] = None


@dataclass
class Agent0Step:
    """Represents a single step taken by an agent plan."""

    skill_name: str
    input: Dict[str, Any]
    output: Dict[str, Any]
    started_at: datetime
    finished_at: datetime
    success: bool
    notes: Optional[str] = None


@dataclass
class Agent0Plan:
    """
    Minimal action plan describing how the agent will achieve a goal.

    This mirrors Agent0's notion of a "plan" but stays framework agnostic so
    that SecOps-specific context (checks, issues, policy suggestions) can be
    embedded directly.
    """

    goal: str
    steps: List[Dict[str, Any]]
    created_at: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "created_at": self.created_at.isoformat(),
            "steps": self.steps,
        }


class Agent0Adapter:
    """
    Lightweight adapter that mirrors the Agent0 concepts.

    - `register_skill` lets you declare available tools.
    - `propose_plan` arranges those tools into a small, deterministic plan.
    - `execute_plan` runs the plan and emits telemetry via a hook.

    This class purposely avoids real LLM calls; those should be plugged in via
    `llm_selector` or replaced entirely once the Agent0 runtime is available.
    """

    def __init__(
        self,
        telemetry_hook: Optional[TelemetryHook] = None,
        llm_selector: Optional[Callable[[str, Dict[str, Any]], str]] = None,
    ) -> None:
        self._skills: Dict[str, Agent0Skill] = {}
        self._telemetry_hook = telemetry_hook
        self._llm_selector = llm_selector

    @property
    def skills(self) -> Dict[str, Agent0Skill]:
        return dict(self._skills)

    def register_skill(self, skill: Agent0Skill) -> None:
        """Register a skill that can be used in a proposed plan."""

        self._skills[skill.name] = skill

    def register_many(self, skills: Iterable[Agent0Skill]) -> None:
        for skill in skills:
            self.register_skill(skill)

    def propose_plan(self, goal: str, context: Optional[Dict[str, Any]] = None) -> Agent0Plan:
        """
        Build a simple, reproducible action plan based on known skills.

        The heuristic ordering favors skills with lower `cost_hint` so the
        resulting plan is economical until a smarter planner is wired in.
        """

        context = context or {}
        sorted_skills = sorted(
            self._skills.values(),
            key=lambda s: s.cost_hint if s.cost_hint is not None else 0.0,
        )

        steps: List[Dict[str, Any]] = []
        for skill in sorted_skills:
            steps.append(
                {
                    "skill": skill.name,
                    "why": skill.description,
                    "inputs": {**skill.parameters, **context},
                }
            )

        return Agent0Plan(goal=goal, steps=steps, created_at=datetime.utcnow())

    def execute_plan(self, plan: Agent0Plan, ambient_context: Optional[Dict[str, Any]] = None) -> List[Agent0Step]:
        """
        Execute a plan locally and emit telemetry for each step.

        This is intentionally deterministic: it invokes `llm_selector` (if
        provided) to fabricate a textual response, then captures timestamps so
        downstream components (cost tracker, metrics) can reason about runtime.
        """

        ambient_context = ambient_context or {}
        results: List[Agent0Step] = []

        for step in plan.steps:
            started = datetime.utcnow()
            skill_name = step["skill"]
            skill = self._skills.get(skill_name)
            merged_inputs = {**ambient_context, **step.get("inputs", {})}

            # Optional LLM-backed response; fall back to deterministic echoing.
            llm_output = None
            if self._llm_selector:
                llm_output = self._llm_selector(skill_name, merged_inputs)

            output = {
                "echo": merged_inputs,
                "llm": llm_output or f"Executed {skill_name} with provided context.",
                "trace": step.get("why", ""),
            }

            finished = datetime.utcnow()
            result = Agent0Step(
                skill_name=skill_name,
                input=merged_inputs,
                output=output,
                started_at=started,
                finished_at=finished,
                success=True,
                notes=skill.description if skill else None,
            )
            results.append(result)

            if self._telemetry_hook:
                self._telemetry_hook(result)

        return results
