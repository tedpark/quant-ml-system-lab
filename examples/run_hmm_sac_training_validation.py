from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.torch_sac import TorchSACConfig, load_sac_actor_checkpoint
from quant_ml_lab.torch_sac_sizing import (
    HMMSizingEnvConfig,
    evaluate_hmm_sac_sizer,
    train_validate_hmm_sac_sizer,
)
from quant_ml_lab.validation import compute_metrics


def _split_rl_frame(frame, train_fraction: float = 0.65):
    split = int(len(frame) * train_fraction)
    if split < 40 or len(frame) - split < 30:
        raise ValueError("not enough rows for train/validation split")
    return frame.iloc[:split].copy(), frame.iloc[split:].copy()


def _baseline_metrics_from_rl_frame(frame):
    baseline_return = frame["baseline_position"] * -frame["spread_return_next"]
    turnover = frame["baseline_position"].diff().abs().fillna(frame["baseline_position"].abs())
    return compute_metrics(baseline_return, turnover)


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=720, seed=99))
    base_train, base_test = train_test_split_time(df)
    dataset = build_hmm_rl_dataset(base_train, base_test)
    rl_train, rl_validation = _split_rl_frame(dataset.frame)
    validation_baseline_metrics = _baseline_metrics_from_rl_frame(rl_validation)

    checkpoint_dir = Path("artifacts/rl_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    seeds = [3, 7, 11]
    runs = []
    for seed in seeds:
        checkpoint_path = checkpoint_dir / f"hmm_sac_seed_{seed}.pt"
        report, _, validation_output = train_validate_hmm_sac_sizer(
            train_frame=rl_train,
            validation_frame=rl_validation,
            feature_columns=dataset.feature_columns,
            sac_config=TorchSACConfig(
                steps=520,
                warmup_steps=48,
                batch_size=32,
                hidden_dim=64,
                gamma=0.95,
                tau=0.02,
                seed=seed,
                replay_capacity=20_000,
            ),
            env_config=HMMSizingEnvConfig(
                transaction_cost_bps=2.0,
                turnover_penalty=0.0007,
                high_vol_penalty=0.001,
            ),
            checkpoint_path=checkpoint_path,
        )
        loaded_actor = load_sac_actor_checkpoint(checkpoint_path)
        reload_report, _ = evaluate_hmm_sac_sizer(
            frame=rl_validation,
            feature_columns=dataset.feature_columns,
            actor=loaded_actor,
            deterministic_action=report.validation.deterministic_action,
            reward_tail_mean=report.validation.reward_tail_mean,
            alpha_final=report.validation.alpha_final,
            actor_loss_tail_mean=report.validation.actor_loss_tail_mean,
            critic_loss_tail_mean=report.validation.critic_loss_tail_mean,
        )
        runs.append(
            {
                "seed": seed,
                "checkpoint_path": str(checkpoint_path),
                "train": report.train.as_dict(),
                "validation": report.validation.as_dict(),
                "reload_validation": reload_report.as_dict(),
                "validation_rows": len(validation_output),
            }
        )

    best_run = max(runs, key=lambda run: run["validation"]["metrics"]["sharpe"])
    best_validation_metrics = best_run["validation"]["metrics"]
    baseline_metrics = validation_baseline_metrics.as_dict()
    validation_returns = [run["validation"]["metrics"]["total_return"] for run in runs]
    validation_sharpes = [run["validation"]["metrics"]["sharpe"] for run in runs]
    benchmark_comparison = {
        "best_total_return_minus_baseline": best_validation_metrics["total_return"]
        - baseline_metrics["total_return"],
        "best_sharpe_minus_baseline": best_validation_metrics["sharpe"]
        - baseline_metrics["sharpe"],
        "best_max_drawdown_minus_baseline": best_validation_metrics["max_drawdown"]
        - baseline_metrics["max_drawdown"],
        "note": "Informational only. This public synthetic lab validates the RL pipeline, not trading alpha.",
    }
    gates = {
        "all_runs_finite": bool(
            np.isfinite(validation_returns).all() and np.isfinite(validation_sharpes).all()
        ),
        "all_checkpoints_written": all(Path(run["checkpoint_path"]).exists() for run in runs),
        "validation_rows_positive": all(run["validation_rows"] > 0 for run in runs),
        "best_checkpoint_reload_matches": abs(
            best_run["validation"]["metrics"]["total_return"]
            - best_run["reload_validation"]["metrics"]["total_return"]
        )
        < 1e-12,
    }
    payload = {
        "experiment": "hmm_sac_training_validation",
        "purpose": "Production-style sanitized RL loop: train/validation split, multi-seed runs, checkpoints, reload validation.",
        "disclosure": dataset.disclosure,
        "features": list(dataset.feature_columns),
        "rows": {
            "rl_train": len(rl_train),
            "rl_validation": len(rl_validation),
        },
        "validation_baseline": baseline_metrics,
        "runs": runs,
        "best_run": best_run,
        "benchmark_comparison": benchmark_comparison,
        "acceptance_gates": gates,
        "accepted": all(gates.values()),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/hmm_sac_training_validation.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
