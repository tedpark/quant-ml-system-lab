# HMM + SAC Training Validation

This report documents the public, sanitized RL training loop.

## Scope

The goal is not to publish a profitable strategy. The goal is to prove that the RL system has the minimum production controls:

- deterministic train/validation split
- multi-seed training
- held-out validation
- checkpoint save and reload
- reload-based validation repeatability
- finite metric gates
- explicit baseline comparison

## Command

```bash
make hmm-sac-training-validation
```

## Latest Local Result

Generated report:

```text
reports/hmm_sac_training_validation.json
```

Latest run summary:

- RL train rows: 141
- RL validation rows: 76
- Seeds: 3, 7, 11
- Best seed by validation Sharpe: 7
- Best validation total return: 0.03546836586925117
- Best validation Sharpe: 1.3655551862696664
- Validation baseline total return: 0.07220544994045475
- Validation baseline Sharpe: 1.405264173193039
- Checkpoint reload matched validation return: true
- Acceptance gates passed: true

## Interpretation

The system controls passed, but the best SAC validation result did not beat the simple public baseline on this synthetic split.

That is an acceptable outcome for a public lab. A production RL workflow should surface this clearly instead of presenting every trained policy as useful. The next work is model selection, reward design, robustness testing, and walk-forward validation, not claiming alpha from one synthetic run.
