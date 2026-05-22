import numpy as np
import pytest

from quant_ml_lab.risk import cvar_position_multiplier, empirical_cvar


def test_empirical_cvar_uses_lower_tail():
    returns = np.array([-0.10, -0.05, 0.01, 0.02, 0.03])

    assert empirical_cvar(returns, alpha=0.2) == -0.10
    assert empirical_cvar(returns, alpha=0.4) == pytest.approx(-0.075)


def test_cvar_position_multiplier_scales_down_tail_risk():
    assert cvar_position_multiplier(-0.04, target_loss=-0.02) == 0.5
    assert cvar_position_multiplier(-0.01, target_loss=-0.02) == 1.0
    assert cvar_position_multiplier(0.01, target_loss=-0.02) == 1.0
