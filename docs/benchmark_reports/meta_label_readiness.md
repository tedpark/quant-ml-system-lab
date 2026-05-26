# Meta-Label Readiness

This report checks whether the public candidate signals have enough label quality
for a supervised trade/skip filter before SAC allocation.

## Summary

- cases: `3`
- candidate diagnostics: `12`
- ready candidate diagnostics: `1`
- ready candidate rate: `0.08333333333333333`
- ready counts: `{'cvar_defensive': 1, 'mean_reversion_full': 0, 'mean_reversion_low_risk': 0, 'volatility_defensive': 0}`
- mean best-bin lift: `{'cvar_defensive': -0.012085399131403959, 'mean_reversion_full': -0.013238402705715044, 'mean_reversion_low_risk': -0.013238402705715044, 'volatility_defensive': -0.013238402705715044}`
- best candidate by lift: `cvar_defensive`
- best feature by lift: `feature_baseline_drawdown`
- best bin lift: `0.08993427879626431`
- meta-label ready: `False`
- research decision: `candidate_features_or_labels_need_redesign`

## Candidate Diagnostics

| dataset | candidate | events | validation_events | positive_rate | validation_positive_rate | mean_forward_return | validation_mean_forward_return | best_feature | best_feature_bin | best_bin_mean_forward_return | best_bin_lift | ready |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| regime_seed_501 | cvar_defensive | 147 | 59 | 0.442177 | 0.440678 | -0.001257 | -0.002195 | feature_baseline_drawdown | high | 0.000498 | 0.089934 | True |
| regime_seed_501 | mean_reversion_full | 147 | 59 | 0.428571 | 0.423729 | -0.004611 | -0.005196 | feature_baseline_drawdown | high | -0.001034 | 0.086475 | False |
| regime_seed_501 | mean_reversion_low_risk | 147 | 59 | 0.428571 | 0.423729 | -0.002305 | -0.002598 | feature_baseline_drawdown | high | -0.000517 | 0.086475 | False |
| regime_seed_501 | volatility_defensive | 147 | 59 | 0.428571 | 0.423729 | -0.004371 | -0.005196 | feature_baseline_drawdown | high | -0.001034 | 0.086475 | False |
| regime_seed_502 | cvar_defensive | 156 | 63 | 0.448718 | 0.476190 | -0.002632 | -0.002313 | feature_baseline_drawdown | high | -0.004450 | -0.126190 | False |
| regime_seed_502 | mean_reversion_full | 156 | 63 | 0.448718 | 0.476190 | -0.005164 | -0.003012 | feature_baseline_drawdown | high | -0.006403 | -0.126190 | False |
| regime_seed_502 | mean_reversion_low_risk | 156 | 63 | 0.448718 | 0.476190 | -0.002582 | -0.001506 | feature_baseline_drawdown | high | -0.003201 | -0.126190 | False |
| regime_seed_502 | volatility_defensive | 156 | 63 | 0.442308 | 0.476190 | -0.005739 | -0.003012 | feature_baseline_drawdown | high | -0.006403 | -0.126190 | False |
| regime_seed_503 | cvar_defensive | 143 | 58 | 0.419580 | 0.379310 | -0.002218 | -0.002210 | feature_regime_transition | low | -0.002210 | 0.000000 | False |
| regime_seed_503 | mean_reversion_full | 143 | 58 | 0.391608 | 0.327586 | -0.004503 | -0.003503 | feature_regime_transition | low | -0.003503 | 0.000000 | False |
| regime_seed_503 | mean_reversion_low_risk | 143 | 58 | 0.391608 | 0.327586 | -0.002251 | -0.001752 | feature_regime_transition | low | -0.001752 | 0.000000 | False |
| regime_seed_503 | volatility_defensive | 143 | 58 | 0.377622 | 0.327586 | -0.003194 | -0.003503 | feature_regime_transition | low | -0.003503 | 0.000000 | False |

## Interpretation

The best feature bucket is selected on the earlier slice of each candidate's
label history and evaluated on the later slice. Meta-labeling should be used as a
filter only when the validation bucket keeps positive lift and positive mean
forward return. If this gate fails, the next work is better labels, features, or
candidate signals, not more SAC tuning.
