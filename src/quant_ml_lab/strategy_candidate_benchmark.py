from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_ml_lab.data import train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import (
    StrategySelectorConfig,
    run_strategy_selector,
)


@dataclass(frozen=True)
class StrategyCandidateBenchmarkCase:
    dataset_id: str
    selected_metrics: dict[str, float | int]
    candidate_metrics: dict[str, dict[str, float | int]]
    best_candidate_by_sharpe: str
    best_candidate_sharpe: float
    selected_minus_best_sharpe: float

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "selected_metrics": self.selected_metrics,
            "candidate_metrics": self.candidate_metrics,
            "best_candidate_by_sharpe": self.best_candidate_by_sharpe,
            "best_candidate_sharpe": self.best_candidate_sharpe,
            "selected_minus_best_sharpe": self.selected_minus_best_sharpe,
        }


@dataclass(frozen=True)
class StrategyCandidateBenchmarkReport:
    cases: list[StrategyCandidateBenchmarkCase]
    summary: dict[str, float | int | bool | dict[str, float] | dict[str, int]]
    disclosure: str = (
        "Candidate-level benchmark matrix. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "cases": [case.as_dict() for case in self.cases],
            "disclosure": self.disclosure,
        }


def run_strategy_candidate_benchmark_matrix(
    price_sets: dict[str, pd.DataFrame],
    selector_config: StrategySelectorConfig | None = None,
    train_fraction: float = 0.65,
) -> StrategyCandidateBenchmarkReport:
    if not price_sets:
        raise ValueError("price_sets must not be empty")

    cfg = selector_config or StrategySelectorConfig()
    cases: list[StrategyCandidateBenchmarkCase] = []
    for dataset_id, prices in price_sets.items():
        train_prices, validation_prices = train_test_split_time(prices, train_fraction)
        dataset = build_hmm_rl_dataset(train_prices, validation_prices)
        _, selector_report = run_strategy_selector(dataset.frame, cfg)
        best_name, best_metrics = _best_candidate(selector_report.candidate_metrics)
        selected_sharpe = float(selector_report.selected_metrics.sharpe)
        best_sharpe = float(best_metrics["sharpe"])
        cases.append(
            StrategyCandidateBenchmarkCase(
                dataset_id=dataset_id,
                selected_metrics=selector_report.selected_metrics.as_dict(),
                candidate_metrics=selector_report.candidate_metrics,
                best_candidate_by_sharpe=best_name,
                best_candidate_sharpe=best_sharpe,
                selected_minus_best_sharpe=selected_sharpe - best_sharpe,
            )
        )

    return StrategyCandidateBenchmarkReport(
        cases=cases,
        summary=_summarize_candidate_benchmarks(cases),
    )


def _best_candidate(
    candidate_metrics: dict[str, dict[str, float | int]],
) -> tuple[str, dict[str, float | int]]:
    return max(
        candidate_metrics.items(),
        key=lambda item: float(item[1]["sharpe"]),
    )


def _summarize_candidate_benchmarks(
    cases: list[StrategyCandidateBenchmarkCase],
) -> dict[str, float | int | bool | dict[str, float] | dict[str, int]]:
    candidate_names = tuple(cases[0].candidate_metrics)
    mean_candidate_sharpe = {
        name: float(
            pd.Series(
                [case.candidate_metrics[name]["sharpe"] for case in cases],
                dtype=float,
            ).mean()
        )
        for name in candidate_names
    }
    mean_candidate_return = {
        name: float(
            pd.Series(
                [case.candidate_metrics[name]["total_return"] for case in cases],
                dtype=float,
            ).mean()
        )
        for name in candidate_names
    }
    best_counts = {
        name: sum(1 for case in cases if case.best_candidate_by_sharpe == name)
        for name in candidate_names
    }
    selected_minus_best = pd.Series(
        [case.selected_minus_best_sharpe for case in cases],
        dtype=float,
    )
    mean_selected_sharpe = float(
        pd.Series([case.selected_metrics["sharpe"] for case in cases], dtype=float).mean()
    )
    strongest_candidate = max(mean_candidate_sharpe, key=mean_candidate_sharpe.get)
    return {
        "cases": len(cases),
        "mean_selected_sharpe": mean_selected_sharpe,
        "mean_selected_minus_best_sharpe": float(selected_minus_best.mean()),
        "worst_selected_minus_best_sharpe": float(selected_minus_best.min()),
        "selected_matches_best_cases": int((selected_minus_best == 0.0).sum()),
        "strongest_candidate_by_mean_sharpe": strongest_candidate,
        "mean_candidate_sharpe": mean_candidate_sharpe,
        "mean_candidate_total_return": mean_candidate_return,
        "best_candidate_counts": {name: int(value) for name, value in best_counts.items()},
        "benchmark_ready": bool(
            float(selected_minus_best.mean()) >= -0.10
            and int((selected_minus_best >= -0.10).sum()) == len(cases)
        ),
    }
