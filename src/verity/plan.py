"""Typed DAG. Planner emits Plan; scheduler walks it.

At this stage plans are hand-written (task yaml -> Plan). The LLM planner
arrives later and produces the same shape, which is the point: verification
does not care who wrote the plan.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class Budget(BaseModel):
    tokens: int = 50_000
    seconds: float = 30.0


class Task(BaseModel):
    id: str
    tool: str
    inputs: dict = Field(default_factory=dict)
    deps: list[str] = Field(default_factory=list)
    budget: Budget = Field(default_factory=Budget)


class Plan(BaseModel):
    tasks: list[Task]

    @model_validator(mode="after")
    def _check_graph(self) -> "Plan":
        ids = [t.id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate task ids")
        idset = set(ids)
        for t in self.tasks:
            for d in t.deps:
                if d not in idset:
                    raise ValueError(f"task {t.id} depends on unknown task {d}")
        # Kahn's algorithm; if we don't visit every node, there's a cycle.
        indeg = {t.id: len(t.deps) for t in self.tasks}
        frontier = [i for i, d in indeg.items() if d == 0]
        seen = 0
        while frontier:
            n = frontier.pop()
            seen += 1
            for t in self.tasks:
                if n in t.deps:
                    indeg[t.id] -= 1
                    if indeg[t.id] == 0:
                        frontier.append(t.id)
        if seen != len(self.tasks):
            raise ValueError("cycle in plan")
        return self