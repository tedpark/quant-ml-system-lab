from __future__ import annotations

import numpy as np


def empirical_cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Return empirical lower-tail CVaR for a return sample."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if returns.size == 0:
        raise ValueError("returns must not be empty")

    sorted_returns = np.sort(np.asarray(returns, dtype=float))
    cutoff = max(1, int(np.ceil(alpha * sorted_returns.size)))
    return float(sorted_returns[:cutoff].mean())


def cvar_position_multiplier(cvar: float, target_loss: float = -0.02, floor: float = 0.0) -> float:
    """Convert CVaR into a simple public-demo position multiplier.

    This is intentionally generic and not a production sizing rule.
    """
    if target_loss >= 0:
        raise ValueError("target_loss must be negative")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be between 0 and 1")
    if cvar >= 0:
        return 1.0
    return float(np.clip(abs(target_loss / cvar), floor, 1.0))
