from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import (
    StrategyCandidate,
    StrategyName,
    StrategySelectorConfig,
    build_strategy_candidates,
    run_strategy_selector,
)
from quant_ml_lab.torch_sac import (
    GaussianActor,
    TorchSACConfig,
    save_sac_actor_checkpoint,
    train_torch_sac,
)
from quant_ml_lab.validation import BacktestMetrics, compute_metrics
from quant_ml_lab.walk_forward import WalkForwardConfig, iter_walk_forward_splits

SAC_STRATEGY_ACTIONS: tuple[StrategyName, ...] = (
    "no_trade",
    "mean_reversion_full",
    "mean_reversion_low_risk",
    "volatility_defensive",
    "cvar_defensive",
)


@dataclass(frozen=True)
class SACStrategyAllocatorEnvConfig:
    transaction_cost_bps: float = 2.0
    turnover_penalty: float = 0.0005
    high_vol_penalty: float = 0.001
    drawdown_penalty: float = 0.002
    concentration_penalty: float = 0.0003
    softmax_temperature: float = 0.75


@dataclass(frozen=True)
class SACStrategyAllocatorReport:
    train_metrics: BacktestMetrics
    validation_metrics: BacktestMetrics
    rule_based_validation_metrics: BacktestMetrics
    candidate_validation_metrics: dict[str, dict[str, float | int]]
    validation_mean_weights: dict[str, float]
    validation_weight_concentration: float
    deterministic_action: float | list[float]
    reward_tail_mean: float
    actor_loss_tail_mean: float
    critic_loss_tail_mean: float
    alpha_final: float
    alpha_loss_tail_mean: float
    diagnostics: dict[str, bool | float]
    checkpoint_path: str | None
    disclosure: str = (
        "Public SAC strategy allocator scaffold. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "train_metrics": self.train_metrics.as_dict(),
            "validation_metrics": self.validation_metrics.as_dict(),
            "rule_based_validation_metrics": self.rule_based_validation_metrics.as_dict(),
            "candidate_validation_metrics": self.candidate_validation_metrics,
            "validation_mean_weights": self.validation_mean_weights,
            "validation_weight_concentration": self.validation_weight_concentration,
            "deterministic_action": self.deterministic_action,
            "reward_tail_mean": self.reward_tail_mean,
            "actor_loss_tail_mean": self.actor_loss_tail_mean,
            "critic_loss_tail_mean": self.critic_loss_tail_mean,
            "alpha_final": self.alpha_final,
            "alpha_loss_tail_mean": self.alpha_loss_tail_mean,
            "diagnostics": self.diagnostics,
            "checkpoint_path": self.checkpoint_path,
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class SACAllocatorWalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    validation_metrics: dict[str, float | int]
    rule_based_validation_metrics: dict[str, float | int]
    validation_mean_weights: dict[str, float]
    validation_weight_concentration: float
    diagnostics: dict[str, bool | float]

    def as_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "validation_metrics": self.validation_metrics,
            "rule_based_validation_metrics": self.rule_based_validation_metrics,
            "validation_mean_weights": self.validation_mean_weights,
            "validation_weight_concentration": self.validation_weight_concentration,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class SACAllocatorWalkForwardReport:
    folds: list[SACAllocatorWalkForwardFold]
    summary: dict[str, float | int | bool]
    disclosure: str = (
        "Walk-forward SAC allocator report. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "folds": [fold.as_dict() for fold in self.folds],
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class SACAllocatorRobustnessCase:
    dataset_id: str
    sac_seed: int
    transaction_cost_bps: float
    walk_forward_summary: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "sac_seed": self.sac_seed,
            "transaction_cost_bps": self.transaction_cost_bps,
            "walk_forward_summary": self.walk_forward_summary,
        }


@dataclass(frozen=True)
class SACAllocatorRobustnessReport:
    cases: list[SACAllocatorRobustnessCase]
    summary: dict[str, float | int | bool]
    disclosure: str = (
        "Robustness matrix for public SAC allocator. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "cases": [case.as_dict() for case in self.cases],
            "disclosure": self.disclosure,
        }


@dataclass(frozen=True)
class SACAllocatorRewardAblationCase:
    ablation: str
    dataset_id: str
    walk_forward_summary: dict[str, float | int | bool]

    def as_dict(self) -> dict[str, object]:
        return {
            "ablation": self.ablation,
            "dataset_id": self.dataset_id,
            "walk_forward_summary": self.walk_forward_summary,
        }


@dataclass(frozen=True)
class SACAllocatorRewardAblationReport:
    cases: list[SACAllocatorRewardAblationCase]
    summary: dict[str, float | int | bool | str]
    disclosure: str = (
        "Reward ablation report for public SAC allocator. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "cases": [case.as_dict() for case in self.cases],
            "disclosure": self.disclosure,
        }


class SACStrategyAllocatorEnv:
    """SAC environment that allocates continuous weights across strategy candidates."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        candidates: dict[StrategyName, StrategyCandidate],
        selector_config: StrategySelectorConfig | None = None,
        env_config: SACStrategyAllocatorEnvConfig | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.candidates = {
            name: StrategyCandidate(
                name=candidate.name,
                description=candidate.description,
                position=candidate.position.reset_index(drop=True),
                multiplier=candidate.multiplier.reset_index(drop=True),
            )
            for name, candidate in candidates.items()
        }
        self.selector_config = selector_config or StrategySelectorConfig()
        self.env_config = env_config or SACStrategyAllocatorEnvConfig()
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.net_returns: list[float] = []
        self.positions: list[float] = []
        self.weights: list[np.ndarray] = []

    @property
    def state_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def action_dim(self) -> int:
        return len(SAC_STRATEGY_ACTIONS)

    def reset(self) -> np.ndarray:
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.net_returns = []
        self.positions = []
        self.weights = []
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        weights = _action_to_weights(action, self.env_config.softmax_temperature)
        row = self.frame.iloc[self.cursor]
        candidate_positions = np.asarray(
            [
                float(self.candidates[name].position.iloc[self.cursor])
                for name in SAC_STRATEGY_ACTIONS
            ],
            dtype=np.float64,
        )
        position = float(weights @ candidate_positions)
        gross_return = position * -float(row["spread_return_next"])
        turnover = abs(position - self.prev_position)
        cost = turnover * (self.selector_config.transaction_cost_bps / 10_000.0)
        net_return = gross_return - cost
        next_equity = self.equity * (1.0 + net_return)
        next_peak = max(self.peak_equity, next_equity)
        drawdown = max(0.0, 1.0 - next_equity / next_peak)
        concentration = float(np.square(weights).sum())
        reward = (
            net_return
            - turnover * self.env_config.turnover_penalty
            - float(row["high_vol_prob"]) * abs(position) * self.env_config.high_vol_penalty
            - drawdown * self.env_config.drawdown_penalty
            - concentration * self.env_config.concentration_penalty
        )

        self.prev_position = position
        self.equity = next_equity
        self.peak_equity = next_peak
        self.net_returns.append(net_return)
        self.positions.append(position)
        self.weights.append(weights)
        self.cursor += 1
        done = self.cursor >= len(self.frame)
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, float(reward), done

    def _state(self) -> np.ndarray:
        return self.frame.loc[self.cursor, list(self.feature_columns)].to_numpy(dtype=np.float32)


def train_validate_strategy_allocator_sac(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[SACStrategyAllocatorReport, pd.DataFrame, pd.DataFrame]:
    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    train_candidates = build_strategy_candidates(train_frame, selector_cfg)
    validation_candidates = build_strategy_candidates(validation_frame, selector_cfg)
    env = SACStrategyAllocatorEnv(train_frame, feature_columns, train_candidates, selector_cfg, env_cfg)
    result = train_torch_sac(
        env,
        sac_config
        or TorchSACConfig(
            steps=700,
            warmup_steps=96,
            batch_size=48,
            gamma=0.95,
            hidden_dim=64,
            target_entropy=-float(len(SAC_STRATEGY_ACTIONS)),
        ),
    )
    train_output, train_metrics, _ = evaluate_strategy_allocator_sac(
        train_frame,
        feature_columns,
        train_candidates,
        result.actor,
        selector_cfg,
        env_cfg,
    )
    validation_output, validation_metrics, validation_mean_weights = evaluate_strategy_allocator_sac(
        validation_frame,
        feature_columns,
        validation_candidates,
        result.actor,
        selector_cfg,
        env_cfg,
    )
    _, rule_report = run_strategy_selector(validation_frame, selector_cfg)

    saved_path: str | None = None
    if checkpoint_path is not None:
        save_sac_actor_checkpoint(result, checkpoint_path)
        saved_path = str(checkpoint_path)

    validation_weight_concentration = float(sum(value**2 for value in validation_mean_weights.values()))
    alpha_final = float(result.alpha_trace[-1]) if result.alpha_trace else float(result.config.alpha)
    actor_loss_tail_mean = (
        float(np.mean(result.actor_loss_trace[-20:])) if result.actor_loss_trace else 0.0
    )
    critic_loss_tail_mean = (
        float(np.mean(result.critic_loss_trace[-20:])) if result.critic_loss_trace else 0.0
    )
    alpha_loss_tail_mean = (
        float(np.mean(result.alpha_loss_trace[-20:])) if result.alpha_loss_trace else 0.0
    )
    diagnostics = {
        "actor_loss_is_finite": bool(np.isfinite(actor_loss_tail_mean)),
        "critic_loss_is_finite": bool(np.isfinite(critic_loss_tail_mean)),
        "alpha_is_positive": alpha_final > 0.0,
        "beats_rule_based_sharpe": validation_metrics.sharpe >= rule_report.selected_metrics.sharpe,
        "weight_concentration": validation_weight_concentration,
    }
    report = SACStrategyAllocatorReport(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        rule_based_validation_metrics=rule_report.selected_metrics,
        candidate_validation_metrics=rule_report.candidate_metrics,
        validation_mean_weights=validation_mean_weights,
        validation_weight_concentration=validation_weight_concentration,
        deterministic_action=result.deterministic_action,
        reward_tail_mean=float(np.mean(result.reward_trace[-50:])) if result.reward_trace else 0.0,
        actor_loss_tail_mean=actor_loss_tail_mean,
        critic_loss_tail_mean=critic_loss_tail_mean,
        alpha_final=alpha_final,
        alpha_loss_tail_mean=alpha_loss_tail_mean,
        diagnostics=diagnostics,
        checkpoint_path=saved_path,
    )
    return report, train_output, validation_output


def run_strategy_allocator_sac_walk_forward(
    prices: pd.DataFrame,
    wf_config: WalkForwardConfig | None = None,
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    rl_train_fraction: float = 0.65,
) -> SACAllocatorWalkForwardReport:
    wf_cfg = wf_config or WalkForwardConfig(train_size=420, test_size=120, step_size=120)
    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    if not 0.2 < rl_train_fraction < 0.9:
        raise ValueError("rl_train_fraction must be between 0.2 and 0.9")

    folds: list[SACAllocatorWalkForwardFold] = []
    for fold_idx, (train_prices, validation_prices) in enumerate(
        iter_walk_forward_splits(prices, wf_cfg),
        start=1,
    ):
        rl_split = int(len(train_prices) * rl_train_fraction)
        regime_fit_prices = train_prices.iloc[:rl_split].copy()
        rl_train_prices = train_prices.iloc[rl_split:].copy()
        train_dataset = build_hmm_rl_dataset(regime_fit_prices, rl_train_prices)
        validation_dataset = build_hmm_rl_dataset(train_prices, validation_prices)
        report, _, _ = train_validate_strategy_allocator_sac(
            train_frame=train_dataset.frame,
            validation_frame=validation_dataset.frame,
            feature_columns=validation_dataset.feature_columns,
            selector_config=selector_cfg,
            env_config=env_cfg,
            sac_config=sac_config,
            checkpoint_path=None,
        )
        folds.append(
            SACAllocatorWalkForwardFold(
                fold=fold_idx,
                train_start=_index_label(train_prices.index[0]),
                train_end=_index_label(train_prices.index[-1]),
                validation_start=_index_label(validation_prices.index[0]),
                validation_end=_index_label(validation_prices.index[-1]),
                validation_metrics=report.validation_metrics.as_dict(),
                rule_based_validation_metrics=report.rule_based_validation_metrics.as_dict(),
                validation_mean_weights=report.validation_mean_weights,
                validation_weight_concentration=report.validation_weight_concentration,
                diagnostics=report.diagnostics,
            )
        )

    if not folds:
        raise ValueError("walk-forward configuration produced no folds")
    summary = _summarize_sac_allocator_walk_forward(folds)
    return SACAllocatorWalkForwardReport(folds=folds, summary=summary)


def run_strategy_allocator_sac_robustness_matrix(
    price_sets: dict[str, pd.DataFrame],
    sac_seeds: tuple[int, ...] = (61, 62),
    transaction_cost_bps_values: tuple[float, ...] = (2.0, 5.0),
    wf_config: WalkForwardConfig | None = None,
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    rl_train_fraction: float = 0.65,
) -> SACAllocatorRobustnessReport:
    if not price_sets:
        raise ValueError("price_sets must not be empty")
    if not sac_seeds:
        raise ValueError("sac_seeds must not be empty")
    if not transaction_cost_bps_values:
        raise ValueError("transaction_cost_bps_values must not be empty")

    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    sac_cfg = sac_config or TorchSACConfig(
        steps=180,
        warmup_steps=32,
        batch_size=32,
        gamma=0.95,
        hidden_dim=32,
        target_entropy=-float(len(SAC_STRATEGY_ACTIONS)),
    )

    cases: list[SACAllocatorRobustnessCase] = []
    for dataset_id, prices in price_sets.items():
        for cost_bps in transaction_cost_bps_values:
            cost_selector_cfg = replace(selector_cfg, transaction_cost_bps=float(cost_bps))
            cost_env_cfg = replace(env_cfg, transaction_cost_bps=float(cost_bps))
            for sac_seed in sac_seeds:
                seeded_sac_cfg = replace(sac_cfg, seed=int(sac_seed))
                wf_report = run_strategy_allocator_sac_walk_forward(
                    prices=prices,
                    wf_config=wf_config,
                    selector_config=cost_selector_cfg,
                    env_config=cost_env_cfg,
                    sac_config=seeded_sac_cfg,
                    rl_train_fraction=rl_train_fraction,
                )
                cases.append(
                    SACAllocatorRobustnessCase(
                        dataset_id=dataset_id,
                        sac_seed=int(sac_seed),
                        transaction_cost_bps=float(cost_bps),
                        walk_forward_summary=wf_report.summary,
                    )
                )

    return SACAllocatorRobustnessReport(
        cases=cases,
        summary=_summarize_sac_allocator_robustness(cases),
    )


def run_strategy_allocator_sac_reward_ablation(
    price_sets: dict[str, pd.DataFrame],
    wf_config: WalkForwardConfig | None = None,
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    rl_train_fraction: float = 0.65,
) -> SACAllocatorRewardAblationReport:
    if not price_sets:
        raise ValueError("price_sets must not be empty")

    base_env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    ablations = {
        "full_reward": base_env_cfg,
        "no_turnover_penalty": replace(base_env_cfg, turnover_penalty=0.0),
        "no_high_vol_penalty": replace(base_env_cfg, high_vol_penalty=0.0),
        "no_drawdown_penalty": replace(base_env_cfg, drawdown_penalty=0.0),
        "no_concentration_penalty": replace(base_env_cfg, concentration_penalty=0.0),
    }
    cases: list[SACAllocatorRewardAblationCase] = []
    for dataset_id, prices in price_sets.items():
        for ablation, ablated_env_cfg in ablations.items():
            wf_report = run_strategy_allocator_sac_walk_forward(
                prices=prices,
                wf_config=wf_config,
                selector_config=selector_config,
                env_config=ablated_env_cfg,
                sac_config=sac_config,
                rl_train_fraction=rl_train_fraction,
            )
            cases.append(
                SACAllocatorRewardAblationCase(
                    ablation=ablation,
                    dataset_id=dataset_id,
                    walk_forward_summary=wf_report.summary,
                )
            )

    return SACAllocatorRewardAblationReport(
        cases=cases,
        summary=_summarize_sac_allocator_reward_ablation(cases),
    )


def evaluate_strategy_allocator_sac(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    candidates: dict[StrategyName, StrategyCandidate],
    actor: GaussianActor,
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
) -> tuple[pd.DataFrame, BacktestMetrics, dict[str, float]]:
    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    actor.eval()
    positions: list[float] = []
    weights_by_row: list[np.ndarray] = []
    device = next(actor.parameters()).device
    for _, row in frame.iterrows():
        state = row.loc[list(feature_columns)].to_numpy(dtype=np.float32)
        with torch.no_grad():
            action = (
                actor.deterministic(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )
        weights = _action_to_weights(action, env_cfg.softmax_temperature)
        candidate_positions = np.asarray(
            [float(candidates[name].position.loc[row.name]) for name in SAC_STRATEGY_ACTIONS],
            dtype=np.float64,
        )
        positions.append(float(weights @ candidate_positions))
        weights_by_row.append(weights)

    output = frame.copy()
    output["sac_allocator_position"] = positions
    output["sac_allocator_net_return"] = _returns_from_position(
        output,
        pd.Series(positions, index=output.index),
        selector_cfg.transaction_cost_bps,
    )
    output["sac_allocator_equity"] = (1.0 + output["sac_allocator_net_return"]).cumprod()
    weight_matrix = np.vstack(weights_by_row) if weights_by_row else np.zeros((0, len(SAC_STRATEGY_ACTIONS)))
    for index, name in enumerate(SAC_STRATEGY_ACTIONS):
        output[f"weight_{name}"] = weight_matrix[:, index] if len(weight_matrix) else []
    turnover = output["sac_allocator_position"].diff().abs().fillna(
        output["sac_allocator_position"].abs()
    )
    mean_weights = {
        name: float(weight_matrix[:, index].mean()) if len(weight_matrix) else 0.0
        for index, name in enumerate(SAC_STRATEGY_ACTIONS)
    }
    return output, compute_metrics(output["sac_allocator_net_return"], turnover), mean_weights


def _action_to_weights(action: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    logits = np.asarray(action, dtype=np.float64) / safe_temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _returns_from_position(
    frame: pd.DataFrame,
    position: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    gross_return = position * -frame["spread_return_next"]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    return gross_return - cost


def _summarize_sac_allocator_walk_forward(
    folds: list[SACAllocatorWalkForwardFold],
) -> dict[str, float | int | bool]:
    sharpes = pd.Series([fold.validation_metrics["sharpe"] for fold in folds], dtype=float)
    rule_sharpes = pd.Series(
        [fold.rule_based_validation_metrics["sharpe"] for fold in folds],
        dtype=float,
    )
    returns = pd.Series([fold.validation_metrics["total_return"] for fold in folds], dtype=float)
    rule_returns = pd.Series(
        [fold.rule_based_validation_metrics["total_return"] for fold in folds],
        dtype=float,
    )
    sharpe_delta = sharpes - rule_sharpes
    return_delta = returns - rule_returns
    positive_sharpe_folds = int((sharpe_delta > 0.0).sum())
    positive_return_folds = int((return_delta > 0.0).sum())
    return {
        "folds": len(folds),
        "mean_validation_sharpe": float(sharpes.mean()),
        "mean_rule_based_sharpe": float(rule_sharpes.mean()),
        "mean_sharpe_delta": float(sharpe_delta.mean()),
        "mean_total_return": float(returns.mean()),
        "mean_rule_based_total_return": float(rule_returns.mean()),
        "mean_total_return_delta": float(return_delta.mean()),
        "positive_sharpe_delta_folds": positive_sharpe_folds,
        "positive_return_delta_folds": positive_return_folds,
        "mean_weight_concentration": float(
            pd.Series([fold.validation_weight_concentration for fold in folds]).mean()
        ),
        "robust_ready": bool(
            positive_sharpe_folds > len(folds) / 2
            and positive_return_folds > len(folds) / 2
            and float(sharpe_delta.mean()) > 0.0
        ),
    }


def _summarize_sac_allocator_robustness(
    cases: list[SACAllocatorRobustnessCase],
) -> dict[str, float | int | bool]:
    sharpe_delta = pd.Series(
        [case.walk_forward_summary["mean_sharpe_delta"] for case in cases],
        dtype=float,
    )
    return_delta = pd.Series(
        [case.walk_forward_summary["mean_total_return_delta"] for case in cases],
        dtype=float,
    )
    robust_flags = pd.Series(
        [bool(case.walk_forward_summary["robust_ready"]) for case in cases],
        dtype=bool,
    )
    positive_sharpe_cases = int((sharpe_delta > 0.0).sum())
    positive_return_cases = int((return_delta > 0.0).sum())
    robust_cases = int(robust_flags.sum())
    case_count = len(cases)
    positive_sharpe_rate = positive_sharpe_cases / case_count
    positive_return_rate = positive_return_cases / case_count
    robust_case_rate = robust_cases / case_count
    return {
        "cases": case_count,
        "mean_sharpe_delta": float(sharpe_delta.mean()),
        "median_sharpe_delta": float(sharpe_delta.median()),
        "worst_sharpe_delta": float(sharpe_delta.min()),
        "mean_total_return_delta": float(return_delta.mean()),
        "median_total_return_delta": float(return_delta.median()),
        "worst_total_return_delta": float(return_delta.min()),
        "positive_sharpe_cases": positive_sharpe_cases,
        "positive_sharpe_rate": float(positive_sharpe_rate),
        "positive_return_cases": positive_return_cases,
        "positive_return_rate": float(positive_return_rate),
        "robust_cases": robust_cases,
        "robust_case_rate": float(robust_case_rate),
        "robust_ready": bool(
            robust_case_rate >= 0.75
            and positive_sharpe_rate >= 0.75
            and positive_return_rate >= 0.75
            and float(sharpe_delta.min()) > 0.0
        ),
    }


def _summarize_sac_allocator_reward_ablation(
    cases: list[SACAllocatorRewardAblationCase],
) -> dict[str, float | int | bool | str]:
    sharpe_delta = pd.Series(
        [case.walk_forward_summary["mean_sharpe_delta"] for case in cases],
        dtype=float,
        index=[case.ablation for case in cases],
    )
    return_delta = pd.Series(
        [case.walk_forward_summary["mean_total_return_delta"] for case in cases],
        dtype=float,
        index=[case.ablation for case in cases],
    )
    robust_flags = pd.Series(
        [bool(case.walk_forward_summary["robust_ready"]) for case in cases],
        dtype=bool,
        index=[case.ablation for case in cases],
    )
    by_ablation = (
        pd.DataFrame(
            {
                "sharpe_delta": sharpe_delta.to_numpy(),
                "return_delta": return_delta.to_numpy(),
                "robust_ready": robust_flags.to_numpy(dtype=int),
            },
            index=sharpe_delta.index,
        )
        .groupby(level=0)
        .mean()
    )
    best_ablation = str(by_ablation["sharpe_delta"].idxmax())
    worst_ablation = str(by_ablation["sharpe_delta"].idxmin())
    full_sharpe_delta = float(by_ablation.loc["full_reward", "sharpe_delta"])
    best_sharpe_delta = float(by_ablation.loc[best_ablation, "sharpe_delta"])
    return {
        "cases": len(cases),
        "ablations": int(by_ablation.shape[0]),
        "best_ablation_by_sharpe": best_ablation,
        "worst_ablation_by_sharpe": worst_ablation,
        "best_mean_sharpe_delta": best_sharpe_delta,
        "full_reward_mean_sharpe_delta": full_sharpe_delta,
        "best_minus_full_sharpe_delta": float(best_sharpe_delta - full_sharpe_delta),
        "mean_sharpe_delta": float(sharpe_delta.mean()),
        "worst_sharpe_delta": float(sharpe_delta.min()),
        "mean_total_return_delta": float(return_delta.mean()),
        "worst_total_return_delta": float(return_delta.min()),
        "robust_cases": int(robust_flags.sum()),
        "robust_case_rate": float(robust_flags.mean()),
        "robust_ready": bool(robust_flags.mean() >= 0.75 and float(sharpe_delta.min()) > 0.0),
    }


def _index_label(value: object) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)
