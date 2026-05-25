from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quant_ml_lab.data import train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.torch_sac_sizing import HMMSizingEnvConfig, train_validate_hmm_sac_sizer
from quant_ml_lab.validation import BacktestConfig, BacktestMetrics, compute_metrics


@dataclass(frozen=True)
class PairRLStrategyConfig:
    train_fraction: float = 0.68
    rl_train_fraction: float = 0.65
    seeds: tuple[int, ...] = (3, 7, 11)
    min_validation_rows: int = 50
    max_validation_drawdown: float = -0.20
    min_trades: int = 3
    require_baseline_outperformance: bool = True
    checkpoint_dir: str = "artifacts/strategy_checkpoints"


@dataclass(frozen=True)
class PairRLSignal:
    date: str
    baseline_position: float
    sized_position: float
    leg_a_target: float
    leg_b_target: float
    high_vol_prob: float
    approved_for_paper_trading: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "date": self.date,
            "baseline_position": self.baseline_position,
            "sized_position": self.sized_position,
            "leg_a_target": self.leg_a_target,
            "leg_b_target": self.leg_b_target,
            "high_vol_prob": self.high_vol_prob,
            "approved_for_paper_trading": self.approved_for_paper_trading,
        }


@dataclass(frozen=True)
class RegimeBehaviorSummary:
    regime: str
    rows: int
    active_rows: int
    mean_high_vol_prob: float
    mean_baseline_abs_position: float
    mean_sac_abs_position: float
    mean_multiplier: float
    mean_active_multiplier: float
    exposure_ratio_to_baseline: float
    mean_net_return: float
    total_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    trades: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "regime": self.regime,
            "rows": self.rows,
            "active_rows": self.active_rows,
            "mean_high_vol_prob": self.mean_high_vol_prob,
            "mean_baseline_abs_position": self.mean_baseline_abs_position,
            "mean_sac_abs_position": self.mean_sac_abs_position,
            "mean_multiplier": self.mean_multiplier,
            "mean_active_multiplier": self.mean_active_multiplier,
            "exposure_ratio_to_baseline": self.exposure_ratio_to_baseline,
            "mean_net_return": self.mean_net_return,
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "turnover": self.turnover,
            "trades": self.trades,
        }


@dataclass(frozen=True)
class RegimeBehaviorReport:
    threshold: float
    normal: RegimeBehaviorSummary
    high_vol: RegimeBehaviorSummary
    action_shift_high_minus_normal: float
    multiplier_shift_high_minus_normal: float
    active_multiplier_shift_high_minus_normal: float
    learned_regime_response: str
    interpretation_warning: str

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "normal": self.normal.as_dict(),
            "high_vol": self.high_vol.as_dict(),
            "action_shift_high_minus_normal": self.action_shift_high_minus_normal,
            "multiplier_shift_high_minus_normal": self.multiplier_shift_high_minus_normal,
            "active_multiplier_shift_high_minus_normal": self.active_multiplier_shift_high_minus_normal,
            "learned_regime_response": self.learned_regime_response,
            "interpretation_warning": self.interpretation_warning,
        }


@dataclass(frozen=True)
class PairRLStrategyReport:
    strategy_name: str
    disclosure: str
    baseline_validation: BacktestMetrics
    best_seed: int
    best_checkpoint_path: str
    best_validation: BacktestMetrics
    benchmark_comparison: dict[str, float]
    infrastructure_gates: dict[str, bool]
    risk_gates: dict[str, bool]
    regime_behavior: RegimeBehaviorReport
    trade_ready: bool
    latest_signal: PairRLSignal

    def as_dict(self) -> dict[str, object]:
        return {
            "strategy_name": self.strategy_name,
            "disclosure": self.disclosure,
            "baseline_validation": self.baseline_validation.as_dict(),
            "best_seed": self.best_seed,
            "best_checkpoint_path": self.best_checkpoint_path,
            "best_validation": self.best_validation.as_dict(),
            "benchmark_comparison": self.benchmark_comparison,
            "infrastructure_gates": self.infrastructure_gates,
            "risk_gates": self.risk_gates,
            "regime_behavior": self.regime_behavior.as_dict(),
            "trade_ready": self.trade_ready,
            "latest_signal": self.latest_signal.as_dict(),
        }


def build_pair_rl_strategy(
    prices: pd.DataFrame,
    config: PairRLStrategyConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    env_config: HMMSizingEnvConfig | None = None,
) -> tuple[PairRLStrategyReport, pd.DataFrame]:
    cfg = config or PairRLStrategyConfig()
    _validate_strategy_input(prices, cfg)
    bt_cfg = backtest_config or BacktestConfig()
    sac_cfg = sac_config or TorchSACConfig(
        steps=520,
        warmup_steps=48,
        batch_size=32,
        hidden_dim=64,
        gamma=0.95,
        tau=0.02,
        replay_capacity=20_000,
    )
    env_cfg = env_config or HMMSizingEnvConfig(
        transaction_cost_bps=bt_cfg.transaction_cost_bps,
        turnover_penalty=0.0007,
        high_vol_penalty=0.001,
    )

    base_train, base_test = train_test_split_time(prices, cfg.train_fraction)
    dataset = build_hmm_rl_dataset(base_train, base_test, bt_config=bt_cfg)
    rl_train, rl_validation = _split_rl_frame(dataset.frame, cfg.rl_train_fraction)
    validation_baseline = _baseline_metrics_from_rl_frame(
        rl_validation,
        transaction_cost_bps=bt_cfg.transaction_cost_bps,
    )

    runs: list[dict[str, object]] = []
    for seed in cfg.seeds:
        checkpoint_path = Path(cfg.checkpoint_dir) / f"pair_rl_seed_{seed}.pt"
        report, _, validation_output = train_validate_hmm_sac_sizer(
            train_frame=rl_train,
            validation_frame=rl_validation,
            feature_columns=dataset.feature_columns,
            sac_config=_with_seed(sac_cfg, seed),
            env_config=env_cfg,
            checkpoint_path=checkpoint_path,
        )
        runs.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "report": report,
                "validation_output": validation_output,
            }
        )

    best_run = max(
        runs,
        key=lambda run: run["report"].validation.metrics.sharpe,  # type: ignore[union-attr]
    )
    best_report = best_run["report"]
    best_validation_output = best_run["validation_output"]
    assert hasattr(best_report, "validation")
    assert isinstance(best_validation_output, pd.DataFrame)

    best_metrics = best_report.validation.metrics
    benchmark_comparison = {
        "total_return_minus_baseline": best_metrics.total_return
        - validation_baseline.total_return,
        "sharpe_minus_baseline": best_metrics.sharpe - validation_baseline.sharpe,
        "max_drawdown_minus_baseline": best_metrics.max_drawdown
        - validation_baseline.max_drawdown,
    }
    infrastructure_gates = {
        "validation_rows_ok": len(rl_validation) >= cfg.min_validation_rows,
        "all_seed_metrics_finite": all(
            np.isfinite(run["report"].validation.metrics.sharpe)  # type: ignore[union-attr]
            and np.isfinite(run["report"].validation.metrics.total_return)  # type: ignore[union-attr]
            for run in runs
        ),
        "best_checkpoint_exists": Path(str(best_run["checkpoint_path"])).exists(),
        "position_bounds_ok": bool(best_validation_output["sac_sized_position"].abs().max() <= 1.0),
    }
    risk_gates = {
        "drawdown_within_limit": best_metrics.max_drawdown >= cfg.max_validation_drawdown,
        "has_enough_trades": best_metrics.trades >= cfg.min_trades,
        "beats_baseline_if_required": (
            benchmark_comparison["sharpe_minus_baseline"] > 0.0
            if cfg.require_baseline_outperformance
            else True
        ),
    }
    trade_ready = all(infrastructure_gates.values()) and all(risk_gates.values())
    regime_behavior = analyze_regime_behavior(best_validation_output)
    latest_signal = _latest_pair_signal(
        best_validation_output,
        approved_for_paper_trading=trade_ready,
    )
    strategy_report = PairRLStrategyReport(
        strategy_name="pair_mean_reversion_hmm_sac_sizer",
        disclosure=(
            "Research strategy candidate. Synthetic/public pipeline only. "
            "No live execution, broker integration, production universe, or private alpha."
        ),
        baseline_validation=validation_baseline,
        best_seed=int(best_run["seed"]),
        best_checkpoint_path=str(best_run["checkpoint_path"]),
        best_validation=best_metrics,
        benchmark_comparison=benchmark_comparison,
        infrastructure_gates=infrastructure_gates,
        risk_gates=risk_gates,
        regime_behavior=regime_behavior,
        trade_ready=trade_ready,
        latest_signal=latest_signal,
    )
    return strategy_report, best_validation_output


def analyze_regime_behavior(
    validation_output: pd.DataFrame,
    high_vol_threshold: float = 0.5,
) -> RegimeBehaviorReport:
    required = {
        "baseline_position",
        "sac_sized_position",
        "sac_net_return",
        "high_vol_prob",
    }
    missing = required.difference(validation_output.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    frame = validation_output.copy()
    frame["sac_multiplier"] = (
        frame["sac_sized_position"].abs()
        / frame["baseline_position"].abs().replace(0.0, np.nan)
    ).fillna(0.0)
    frame["sac_turnover"] = (
        frame["sac_sized_position"].diff().abs().fillna(frame["sac_sized_position"].abs())
    )
    high_mask = frame["high_vol_prob"] >= high_vol_threshold
    normal = _summarize_regime("normal", frame.loc[~high_mask])
    high_vol = _summarize_regime("high_vol", frame.loc[high_mask])
    action_shift = high_vol.mean_sac_abs_position - normal.mean_sac_abs_position
    multiplier_shift = high_vol.mean_multiplier - normal.mean_multiplier
    active_multiplier_shift = high_vol.mean_active_multiplier - normal.mean_active_multiplier
    response = _classify_regime_response(active_multiplier_shift)
    return RegimeBehaviorReport(
        threshold=high_vol_threshold,
        normal=normal,
        high_vol=high_vol,
        action_shift_high_minus_normal=action_shift,
        multiplier_shift_high_minus_normal=multiplier_shift,
        active_multiplier_shift_high_minus_normal=active_multiplier_shift,
        learned_regime_response=response,
        interpretation_warning=(
            "Absolute exposure can rise only because the baseline signal is active more often. "
            "Use active_multiplier_shift_high_minus_normal to judge whether SAC itself changed sizing."
        ),
    )


def _validate_strategy_input(prices: pd.DataFrame, cfg: PairRLStrategyConfig) -> None:
    missing = {"asset_a", "asset_b"}.difference(prices.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if len(prices) < 260:
        raise ValueError("at least 260 rows are required")
    if not 0.2 < cfg.train_fraction < 0.9:
        raise ValueError("train_fraction must be between 0.2 and 0.9")
    if not 0.2 < cfg.rl_train_fraction < 0.9:
        raise ValueError("rl_train_fraction must be between 0.2 and 0.9")
    if not cfg.seeds:
        raise ValueError("at least one seed is required")


def _summarize_regime(regime: str, frame: pd.DataFrame) -> RegimeBehaviorSummary:
    if frame.empty:
        return RegimeBehaviorSummary(
            regime=regime,
            rows=0,
            active_rows=0,
            mean_high_vol_prob=0.0,
            mean_baseline_abs_position=0.0,
            mean_sac_abs_position=0.0,
            mean_multiplier=0.0,
            mean_active_multiplier=0.0,
            exposure_ratio_to_baseline=0.0,
            mean_net_return=0.0,
            total_return=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            turnover=0.0,
            trades=0,
        )
    metrics = compute_metrics(frame["sac_net_return"], frame["sac_turnover"])
    active = frame["baseline_position"].abs() > 0.0
    active_multiplier = frame.loc[active, "sac_multiplier"]
    mean_baseline_abs_position = float(frame["baseline_position"].abs().mean())
    mean_sac_abs_position = float(frame["sac_sized_position"].abs().mean())
    exposure_ratio = (
        mean_sac_abs_position / mean_baseline_abs_position
        if mean_baseline_abs_position > 0.0
        else 0.0
    )
    return RegimeBehaviorSummary(
        regime=regime,
        rows=len(frame),
        active_rows=int(active.sum()),
        mean_high_vol_prob=float(frame["high_vol_prob"].mean()),
        mean_baseline_abs_position=mean_baseline_abs_position,
        mean_sac_abs_position=mean_sac_abs_position,
        mean_multiplier=float(frame["sac_multiplier"].mean()),
        mean_active_multiplier=float(active_multiplier.mean()) if len(active_multiplier) else 0.0,
        exposure_ratio_to_baseline=exposure_ratio,
        mean_net_return=float(frame["sac_net_return"].mean()),
        total_return=metrics.total_return,
        sharpe=metrics.sharpe,
        max_drawdown=metrics.max_drawdown,
        turnover=metrics.turnover,
        trades=metrics.trades,
    )


def _classify_regime_response(active_multiplier_shift: float) -> str:
    tolerance = 0.02
    if active_multiplier_shift < -tolerance:
        return "defensive_sizing_in_high_vol"
    if active_multiplier_shift > tolerance:
        return "aggressive_sizing_in_high_vol"
    return "neutral_or_mixed_sizing"


def _split_rl_frame(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(frame) * train_fraction)
    if split < 40 or len(frame) - split < 30:
        raise ValueError("not enough rows for RL train/validation split")
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def _baseline_metrics_from_rl_frame(
    frame: pd.DataFrame,
    transaction_cost_bps: float,
) -> BacktestMetrics:
    baseline_position = frame["baseline_position"]
    gross_return = baseline_position * -frame["spread_return_next"]
    turnover = baseline_position.diff().abs().fillna(baseline_position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    return compute_metrics(gross_return - cost, turnover)


def _latest_pair_signal(
    validation_output: pd.DataFrame,
    approved_for_paper_trading: bool,
) -> PairRLSignal:
    last = validation_output.iloc[-1]
    sized_position = float(last["sac_sized_position"])
    return PairRLSignal(
        date=str(validation_output.index[-1].date())
        if hasattr(validation_output.index[-1], "date")
        else str(validation_output.index[-1]),
        baseline_position=float(last["baseline_position"]),
        sized_position=sized_position,
        leg_a_target=-sized_position,
        leg_b_target=sized_position,
        high_vol_prob=float(last["high_vol_prob"]),
        approved_for_paper_trading=approved_for_paper_trading,
    )


def _with_seed(config: TorchSACConfig, seed: int) -> TorchSACConfig:
    return TorchSACConfig(
        steps=config.steps,
        warmup_steps=config.warmup_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        tau=config.tau,
        alpha=config.alpha,
        alpha_init=config.alpha_init,
        alpha_lr=config.alpha_lr,
        alpha_floor=config.alpha_floor,
        target_entropy=config.target_entropy,
        automatic_entropy_tuning=config.automatic_entropy_tuning,
        actor_lr=config.actor_lr,
        critic_lr=config.critic_lr,
        hidden_dim=config.hidden_dim,
        seed=seed,
        replay_capacity=config.replay_capacity,
        device=config.device,
    )
