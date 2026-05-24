from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.torch_sac import QuadraticActionEnv, QuadraticEnvConfig, TorchSACConfig, train_torch_sac


def main() -> None:
    env = QuadraticActionEnv(QuadraticEnvConfig(target_action=0.5))
    result = train_torch_sac(env, TorchSACConfig())
    report = {
        "experiment": "torch_sac_quadratic",
        "purpose": "PyTorch SAC learning lab. Not a trading strategy.",
        "target_action": env.config.target_action,
        "deterministic_action": result.deterministic_action,
        "first_10_rewards": result.reward_trace[:10],
        "last_10_rewards": result.reward_trace[-10:],
        "last_5_actor_losses": result.actor_loss_trace[-5:],
        "last_5_critic_losses": result.critic_loss_trace[-5:],
        "last_5_alpha_values": result.alpha_trace[-5:],
        "last_5_alpha_losses": result.alpha_loss_trace[-5:],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/torch_sac_quadratic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
