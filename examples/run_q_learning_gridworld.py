from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.rl import GridWorld, GridWorldConfig, QLearningConfig, evaluate_policy, train_q_learning


def main() -> None:
    env = GridWorld(GridWorldConfig())
    result = train_q_learning(env, QLearningConfig())
    evaluation_return, path = evaluate_policy(env, result.policy)
    report = {
        "experiment": "q_learning_gridworld",
        "purpose": "RL fundamentals lab. Not a trading strategy.",
        "episodes": len(result.episode_returns),
        "first_10_returns": result.episode_returns[:10],
        "last_10_returns": result.episode_returns[-10:],
        "evaluation_return": evaluation_return,
        "path": path,
        "policy": result.policy,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/q_learning_gridworld.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
