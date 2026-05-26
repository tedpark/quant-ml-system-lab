from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_ml_lab.data import train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import (
    StrategySelectorConfig,
    build_strategy_candidates,
    run_strategy_selector,
)
from quant_ml_lab.validation import compute_metrics

SummaryValue = float | int | bool | str | dict[str, float] | dict[str, int] | list[str]


@dataclass(frozen=True)
class StrategyCandidateBenchmarkCase:
    dataset_id: str
    selected_metrics: dict[str, float | int]
    candidate_metrics: dict[str, dict[str, float | int]]
    regime_selected_metrics: dict[str, dict[str, float | int]]
    regime_candidate_metrics: dict[str, dict[str, dict[str, float | int]]]
    best_candidate_by_sharpe: str
    best_candidate_sharpe: float
    selected_minus_best_sharpe: float
    weakest_regime_by_selected_sharpe: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "selected_metrics": self.selected_metrics,
            "candidate_metrics": self.candidate_metrics,
            "regime_selected_metrics": self.regime_selected_metrics,
            "regime_candidate_metrics": self.regime_candidate_metrics,
            "best_candidate_by_sharpe": self.best_candidate_by_sharpe,
            "best_candidate_sharpe": self.best_candidate_sharpe,
            "selected_minus_best_sharpe": self.selected_minus_best_sharpe,
            "weakest_regime_by_selected_sharpe": self.weakest_regime_by_selected_sharpe,
        }


@dataclass(frozen=True)
class StrategyCandidateBenchmarkReport:
    cases: list[StrategyCandidateBenchmarkCase]
    summary: dict[str, SummaryValue]
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
        frame = dataset.frame.copy()
        if "synthetic_regime" in validation_prices.columns:
            frame["synthetic_regime"] = validation_prices["synthetic_regime"].reindex(frame.index)
        selected_output, selector_report = run_strategy_selector(frame, cfg)
        regime_selected_metrics = _selected_metrics_by_regime(selected_output)
        regime_candidate_metrics = _candidate_metrics_by_regime(frame, cfg)
        weakest_regime = _weakest_regime(regime_selected_metrics)
        best_name, best_metrics = _best_candidate(selector_report.candidate_metrics)
        selected_sharpe = float(selector_report.selected_metrics.sharpe)
        best_sharpe = float(best_metrics["sharpe"])
        cases.append(
            StrategyCandidateBenchmarkCase(
                dataset_id=dataset_id,
                selected_metrics=selector_report.selected_metrics.as_dict(),
                candidate_metrics=selector_report.candidate_metrics,
                regime_selected_metrics=regime_selected_metrics,
                regime_candidate_metrics=regime_candidate_metrics,
                best_candidate_by_sharpe=best_name,
                best_candidate_sharpe=best_sharpe,
                selected_minus_best_sharpe=selected_sharpe - best_sharpe,
                weakest_regime_by_selected_sharpe=weakest_regime,
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


def _selected_metrics_by_regime(
    selected_output: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    if "synthetic_regime" not in selected_output.columns:
        return {}
    metrics: dict[str, dict[str, float | int]] = {}
    for regime, group in selected_output.groupby("synthetic_regime"):
        turnover = group["selected_position"].diff().abs().fillna(group["selected_position"].abs())
        metrics[str(regime)] = compute_metrics(group["selected_net_return"], turnover).as_dict()
    return metrics


def _candidate_metrics_by_regime(
    frame: pd.DataFrame,
    config: StrategySelectorConfig,
) -> dict[str, dict[str, dict[str, float | int]]]:
    if "synthetic_regime" not in frame.columns:
        return {}
    candidates = build_strategy_candidates(frame, config)
    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for regime, group in frame.groupby("synthetic_regime"):
        regime_metrics: dict[str, dict[str, float | int]] = {}
        for name, candidate in candidates.items():
            position = candidate.position.loc[group.index]
            returns = _returns_from_position(group, position, config.transaction_cost_bps)
            turnover = position.diff().abs().fillna(position.abs())
            regime_metrics[name] = compute_metrics(returns, turnover).as_dict()
        metrics[str(regime)] = regime_metrics
    return metrics


def _returns_from_position(
    frame: pd.DataFrame,
    position: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    gross_return = position * -frame["spread_return_next"]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    return gross_return - cost


def _weakest_regime(
    regime_selected_metrics: dict[str, dict[str, float | int]],
) -> str | None:
    if not regime_selected_metrics:
        return None
    return min(
        regime_selected_metrics,
        key=lambda regime: float(regime_selected_metrics[regime]["sharpe"]),
    )


def _summarize_candidate_benchmarks(
    cases: list[StrategyCandidateBenchmarkCase],
) -> dict[str, SummaryValue]:
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
    weakest_regime_counts = {
        str(regime): sum(1 for case in cases if case.weakest_regime_by_selected_sharpe == regime)
        for regime in sorted(
            {
                case.weakest_regime_by_selected_sharpe
                for case in cases
                if case.weakest_regime_by_selected_sharpe is not None
            }
        )
    }
    diagnostics = _offline_rl_readiness_diagnostics(cases, candidate_names)
    benchmark_ready = bool(
        float(selected_minus_best.mean()) >= -0.10
        and int((selected_minus_best >= -0.10).sum()) == len(cases)
    )
    rl_allocation_ready = bool(
        benchmark_ready
        and diagnostics["no_trade_best_rate"] <= 0.25
        and diagnostics["non_no_trade_best_rate"] >= 0.50
        and diagnostics["selected_positive_sharpe_rate"] >= 0.50
        and diagnostics["negative_selected_regime_rate"] <= 0.25
    )
    redesign_reasons = _redesign_reasons(benchmark_ready, rl_allocation_ready, diagnostics)
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
        "weakest_regime_counts": {
            regime: int(value) for regime, value in weakest_regime_counts.items()
        },
        "no_trade_best_rate": diagnostics["no_trade_best_rate"],
        "non_no_trade_best_rate": diagnostics["non_no_trade_best_rate"],
        "selected_positive_sharpe_rate": diagnostics["selected_positive_sharpe_rate"],
        "negative_selected_regime_rate": diagnostics["negative_selected_regime_rate"],
        "mean_candidate_trade_rate": diagnostics["mean_candidate_trade_rate"],
        "benchmark_ready": benchmark_ready,
        "rl_allocation_ready": rl_allocation_ready,
        "research_decision": (
            "candidate_signal_redesign_before_rl"
            if not rl_allocation_ready
            else "rl_allocator_iteration_allowed"
        ),
        "redesign_reasons": redesign_reasons,
    }


def _offline_rl_readiness_diagnostics(
    cases: list[StrategyCandidateBenchmarkCase],
    candidate_names: tuple[str, ...],
) -> dict[str, float]:
    selected_sharpes = pd.Series(
        [case.selected_metrics["sharpe"] for case in cases],
        dtype=float,
    )
    regime_sharpes = pd.Series(
        [
            metrics["sharpe"]
            for case in cases
            for metrics in case.regime_selected_metrics.values()
        ],
        dtype=float,
    )
    candidate_trade_rates = [
        float(
            pd.Series(
                [case.candidate_metrics[name]["trades"] for case in cases],
                dtype=float,
            ).gt(0.0).mean()
        )
        for name in candidate_names
        if name != "no_trade"
    ]
    no_trade_best_count = sum(1 for case in cases if case.best_candidate_by_sharpe == "no_trade")
    return {
        "no_trade_best_rate": float(no_trade_best_count / len(cases)),
        "non_no_trade_best_rate": float(1.0 - no_trade_best_count / len(cases)),
        "selected_positive_sharpe_rate": float(selected_sharpes.gt(0.0).mean()),
        "negative_selected_regime_rate": (
            float(regime_sharpes.lt(0.0).mean()) if not regime_sharpes.empty else 0.0
        ),
        "mean_candidate_trade_rate": (
            float(pd.Series(candidate_trade_rates, dtype=float).mean())
            if candidate_trade_rates
            else 0.0
        ),
    }


def _redesign_reasons(
    benchmark_ready: bool,
    rl_allocation_ready: bool,
    diagnostics: dict[str, float],
) -> list[str]:
    if rl_allocation_ready:
        return []
    reasons: list[str] = []
    if not benchmark_ready:
        reasons.append("selected_policy_does_not_beat_best_static_candidate")
    if diagnostics["no_trade_best_rate"] > 0.25:
        reasons.append("no_trade_is_the_dominant_best_candidate")
    if diagnostics["non_no_trade_best_rate"] < 0.50:
        reasons.append("tradable_candidate_edge_is_not_repeated_across_datasets")
    if diagnostics["selected_positive_sharpe_rate"] < 0.50:
        reasons.append("selected_policy_has_low_positive_sharpe_frequency")
    if diagnostics["negative_selected_regime_rate"] > 0.25:
        reasons.append("selected_policy_loses_across_too_many_regimes")
    return reasons
