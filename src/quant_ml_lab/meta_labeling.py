from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_ml_lab.data import train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import (
    StrategySelectorConfig,
    build_strategy_candidates,
)


@dataclass(frozen=True)
class MetaLabelConfig:
    horizon: int = 5
    min_events: int = 30
    train_fraction: float = 0.60
    min_positive_rate: float = 0.52
    min_lift: float = 0.08
    transaction_cost_bps: float = 2.0
    feature_columns: tuple[str, ...] = (
        "feature_abs_zscore",
        "feature_high_vol_prob",
        "feature_regime_transition",
        "feature_spread_momentum",
        "feature_spread_volatility",
        "feature_recent_pnl",
        "feature_baseline_drawdown",
    )


@dataclass(frozen=True)
class MetaLabelCandidateDiagnostics:
    candidate: str
    events: int
    train_events: int
    validation_events: int
    positive_rate: float
    validation_positive_rate: float
    mean_forward_return: float
    validation_mean_forward_return: float
    best_feature: str | None
    best_feature_bin: str | None
    best_bin_events: int
    best_bin_positive_rate: float
    best_bin_mean_forward_return: float
    best_bin_lift: float
    meta_label_ready: bool

    def as_dict(self) -> dict[str, float | int | bool | str | None]:
        return {
            "candidate": self.candidate,
            "events": self.events,
            "train_events": self.train_events,
            "validation_events": self.validation_events,
            "positive_rate": self.positive_rate,
            "validation_positive_rate": self.validation_positive_rate,
            "mean_forward_return": self.mean_forward_return,
            "validation_mean_forward_return": self.validation_mean_forward_return,
            "best_feature": self.best_feature,
            "best_feature_bin": self.best_feature_bin,
            "best_bin_events": self.best_bin_events,
            "best_bin_positive_rate": self.best_bin_positive_rate,
            "best_bin_mean_forward_return": self.best_bin_mean_forward_return,
            "best_bin_lift": self.best_bin_lift,
            "meta_label_ready": self.meta_label_ready,
        }


@dataclass(frozen=True)
class MetaLabelDatasetCase:
    dataset_id: str
    rows: int
    diagnostics: dict[str, MetaLabelCandidateDiagnostics]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "rows": self.rows,
            "diagnostics": {
                name: diagnostics.as_dict() for name, diagnostics in self.diagnostics.items()
            },
        }


@dataclass(frozen=True)
class MetaLabelReadinessReport:
    cases: list[MetaLabelDatasetCase]
    summary: dict[str, object]
    disclosure: str = (
        "Meta-label readiness diagnostics. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "cases": [case.as_dict() for case in self.cases],
            "disclosure": self.disclosure,
        }


def run_meta_label_readiness_matrix(
    price_sets: dict[str, pd.DataFrame],
    selector_config: StrategySelectorConfig | None = None,
    meta_config: MetaLabelConfig | None = None,
    train_fraction: float = 0.65,
) -> MetaLabelReadinessReport:
    if not price_sets:
        raise ValueError("price_sets must not be empty")

    selector_cfg = selector_config or StrategySelectorConfig()
    meta_cfg = meta_config or MetaLabelConfig(
        transaction_cost_bps=selector_cfg.transaction_cost_bps
    )
    cases: list[MetaLabelDatasetCase] = []
    for dataset_id, prices in price_sets.items():
        train_prices, validation_prices = train_test_split_time(prices, train_fraction)
        dataset = build_hmm_rl_dataset(train_prices, validation_prices)
        frame = dataset.frame.copy()
        label_frame = build_meta_label_frame(frame, selector_cfg, meta_cfg)
        cases.append(
            MetaLabelDatasetCase(
                dataset_id=dataset_id,
                rows=int(len(label_frame)),
                diagnostics=_diagnostics_by_candidate(label_frame, meta_cfg),
            )
        )

    return MetaLabelReadinessReport(
        cases=cases,
        summary=_summarize_meta_label_readiness(cases),
    )


def build_meta_label_frame(
    frame: pd.DataFrame,
    selector_config: StrategySelectorConfig | None = None,
    meta_config: MetaLabelConfig | None = None,
) -> pd.DataFrame:
    selector_cfg = selector_config or StrategySelectorConfig()
    meta_cfg = meta_config or MetaLabelConfig(
        transaction_cost_bps=selector_cfg.transaction_cost_bps
    )
    candidates = build_strategy_candidates(frame, selector_cfg)
    rows: list[pd.DataFrame] = []
    for name, candidate in candidates.items():
        if name == "no_trade":
            continue
        position = candidate.position.astype(float)
        active = position.abs() > 0.0
        if not bool(active.any()):
            continue
        forward_return = _forward_fixed_position_return(
            frame,
            position,
            meta_cfg.horizon,
            meta_cfg.transaction_cost_bps,
        )
        candidate_frame = frame.loc[active, list(meta_cfg.feature_columns)].copy()
        candidate_frame["candidate"] = str(name)
        candidate_frame["position_abs"] = position.loc[active].abs()
        candidate_frame["forward_return"] = forward_return.loc[active]
        candidate_frame["meta_label"] = (candidate_frame["forward_return"] > 0.0).astype(int)
        rows.append(candidate_frame)

    if not rows:
        return pd.DataFrame(
            columns=[
                *meta_cfg.feature_columns,
                "candidate",
                "position_abs",
                "forward_return",
                "meta_label",
            ]
        )
    return pd.concat(rows).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)


def _forward_fixed_position_return(
    frame: pd.DataFrame,
    position: pd.Series,
    horizon: int,
    transaction_cost_bps: float,
) -> pd.Series:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    one_step = position * -frame["spread_return_next"]
    forward = pd.Series(0.0, index=frame.index)
    for offset in range(horizon):
        forward = forward + one_step.shift(-offset).fillna(0.0)
    entry_cost = position.diff().abs().fillna(position.abs()) * (transaction_cost_bps / 10_000.0)
    return forward - entry_cost


def _diagnostics_by_candidate(
    label_frame: pd.DataFrame,
    config: MetaLabelConfig,
) -> dict[str, MetaLabelCandidateDiagnostics]:
    diagnostics: dict[str, MetaLabelCandidateDiagnostics] = {}
    for candidate, group in label_frame.groupby("candidate"):
        diagnostics[str(candidate)] = _candidate_diagnostics(str(candidate), group, config)
    return diagnostics


def _candidate_diagnostics(
    candidate: str,
    group: pd.DataFrame,
    config: MetaLabelConfig,
) -> MetaLabelCandidateDiagnostics:
    events = int(len(group))
    if not 0.1 < config.train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.1 and 0.9")
    split = int(events * config.train_fraction)
    train = group.iloc[:split]
    validation = group.iloc[split:]
    train_events = int(len(train))
    validation_events = int(len(validation))
    positive_rate = float(group["meta_label"].mean()) if events else 0.0
    validation_positive_rate = (
        float(validation["meta_label"].mean()) if validation_events else 0.0
    )
    mean_forward_return = float(group["forward_return"].mean()) if events else 0.0
    validation_mean_forward_return = (
        float(validation["forward_return"].mean()) if validation_events else 0.0
    )
    best_feature: str | None = None
    best_feature_bin: str | None = None
    best_train_lift = 0.0

    for feature in config.feature_columns:
        if feature not in train:
            continue
        low, high = _feature_tail_masks(train[feature])
        for bin_name, mask in (("low", low), ("high", high)):
            bin_events = int(mask.sum())
            if bin_events == 0:
                continue
            bin_positive_rate = float(train.loc[mask, "meta_label"].mean())
            lift = bin_positive_rate - float(train["meta_label"].mean())
            if lift > best_train_lift:
                best_feature = feature
                best_feature_bin = bin_name
                best_train_lift = lift

    best_bin_events = 0
    best_bin_positive_rate = 0.0
    best_bin_mean_forward_return = 0.0
    best_bin_lift = 0.0
    if best_feature is not None and best_feature_bin is not None and validation_events:
        train_feature = train[best_feature]
        threshold = float(
            train_feature.quantile(0.25 if best_feature_bin == "low" else 0.75)
        )
        validation_mask = (
            validation[best_feature] <= threshold
            if best_feature_bin == "low"
            else validation[best_feature] >= threshold
        )
        best_bin_events = int(validation_mask.sum())
        if best_bin_events:
            best_bin_positive_rate = float(
                validation.loc[validation_mask, "meta_label"].mean()
            )
            best_bin_mean_forward_return = float(
                validation.loc[validation_mask, "forward_return"].mean()
            )
            best_bin_lift = best_bin_positive_rate - validation_positive_rate

    meta_label_ready = bool(
        events >= config.min_events
        and train_events >= max(10, int(events * 0.30))
        and validation_events >= max(10, int(events * 0.20))
        and best_bin_events >= max(10, int(validation_events * 0.20))
        and best_bin_positive_rate >= config.min_positive_rate
        and best_bin_mean_forward_return > 0.0
        and best_bin_lift >= config.min_lift
    )
    return MetaLabelCandidateDiagnostics(
        candidate=candidate,
        events=events,
        train_events=train_events,
        validation_events=validation_events,
        positive_rate=positive_rate,
        validation_positive_rate=validation_positive_rate,
        mean_forward_return=mean_forward_return,
        validation_mean_forward_return=validation_mean_forward_return,
        best_feature=best_feature,
        best_feature_bin=best_feature_bin,
        best_bin_events=best_bin_events,
        best_bin_positive_rate=best_bin_positive_rate,
        best_bin_mean_forward_return=best_bin_mean_forward_return,
        best_bin_lift=best_bin_lift,
        meta_label_ready=meta_label_ready,
    )


def _feature_tail_masks(feature: pd.Series) -> tuple[pd.Series, pd.Series]:
    low_threshold = float(feature.quantile(0.25))
    high_threshold = float(feature.quantile(0.75))
    return feature <= low_threshold, feature >= high_threshold


def _summarize_meta_label_readiness(
    cases: list[MetaLabelDatasetCase],
) -> dict[str, object]:
    flat = [
        diagnostics
        for case in cases
        for diagnostics in case.diagnostics.values()
    ]
    ready = [diagnostics for diagnostics in flat if diagnostics.meta_label_ready]
    best = max(flat, key=lambda item: item.best_bin_lift, default=None)
    candidate_names = sorted({diagnostics.candidate for diagnostics in flat})
    ready_counts = {
        candidate: sum(
            1
            for case in cases
            if candidate in case.diagnostics and case.diagnostics[candidate].meta_label_ready
        )
        for candidate in candidate_names
    }
    mean_lift = {
        candidate: float(
            pd.Series(
                [
                    case.diagnostics[candidate].best_bin_lift
                    for case in cases
                    if candidate in case.diagnostics
                ],
                dtype=float,
            ).mean()
        )
        for candidate in candidate_names
    }
    return {
        "cases": len(cases),
        "candidate_diagnostics": len(flat),
        "ready_candidate_diagnostics": len(ready),
        "ready_candidate_rate": float(len(ready) / len(flat)) if flat else 0.0,
        "ready_counts": ready_counts,
        "mean_best_bin_lift": mean_lift,
        "best_candidate_by_lift": best.candidate if best else None,
        "best_feature_by_lift": best.best_feature if best else None,
        "best_bin_lift": best.best_bin_lift if best else 0.0,
        "meta_label_ready": bool(len(ready) >= max(1, len(cases))),
        "research_decision": (
            "meta_label_filter_iteration_allowed"
            if len(ready) >= max(1, len(cases))
            else "candidate_features_or_labels_need_redesign"
        ),
    }
