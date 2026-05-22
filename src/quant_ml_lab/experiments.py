from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ExperimentRun:
    run_id: str
    name: str
    created_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float | int] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "created_at": self.created_at,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
        }


class JsonlExperimentTracker:
    """Tiny local experiment tracker for public examples.

    This mimics the role of an experiment tracking system without requiring a
    service, credential, private registry, or production artifact.
    """

    def __init__(self, path: str | Path = "reports/experiments.jsonl") -> None:
        self.path = Path(path)

    def log_run(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        metrics: dict[str, float | int] | None = None,
        tags: dict[str, str] | None = None,
    ) -> ExperimentRun:
        run = ExperimentRun(
            run_id=str(uuid4()),
            name=name,
            created_at=datetime.now(UTC).isoformat(),
            params=params or {},
            metrics=metrics or {},
            tags=tags or {},
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(run.as_dict(), sort_keys=True) + "\n")
        return run

    def load_runs(self) -> list[ExperimentRun]:
        if not self.path.exists():
            return []
        runs: list[ExperimentRun] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            runs.append(
                ExperimentRun(
                    run_id=payload["run_id"],
                    name=payload["name"],
                    created_at=payload["created_at"],
                    params=payload.get("params", {}),
                    metrics=payload.get("metrics", {}),
                    tags=payload.get("tags", {}),
                )
            )
        return runs


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    model_version: str
    stage: str
    metrics: dict[str, float | int]
    disclosure: str = "Demo registry entry. No production model artifact is included."

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "stage": self.stage,
            "metrics": self.metrics,
            "disclosure": self.disclosure,
        }


class LocalModelRegistry:
    """Manifest-only model registry for sanitized public demos."""

    def __init__(self, path: str | Path = "reports/model_registry.json") -> None:
        self.path = Path(path)

    def promote(self, entry: ModelRegistryEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entry.as_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def current(self) -> ModelRegistryEntry | None:
        if not self.path.exists():
            return None
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return ModelRegistryEntry(
            model_name=payload["model_name"],
            model_version=payload["model_version"],
            stage=payload["stage"],
            metrics=payload["metrics"],
            disclosure=payload.get(
                "disclosure", "Demo registry entry. No production model artifact is included."
            ),
        )
