from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.sac import (
    ContinuousBandit,
    ContinuousBanditConfig,
    SoftActorCriticBanditConfig,
    train_soft_actor_critic_bandit,
)


def main() -> None:
    env = ContinuousBandit(ContinuousBanditConfig(target_action=0.65, noise_scale=0.0))
    result = train_soft_actor_critic_bandit(env, SoftActorCriticBanditConfig())
    report = {
        "experiment": "sac_continuous_bandit",
        "purpose": "SAC concept lab. Not a trading strategy.",
        "target_action": env.config.target_action,
        "expected_action": result.expected_action,
        "greedy_action": result.greedy_action,
        "entropy": result.entropy,
        "first_10_mean_rewards": result.mean_reward_trace[:10],
        "last_10_mean_rewards": result.mean_reward_trace[-10:],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sac_bandit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
