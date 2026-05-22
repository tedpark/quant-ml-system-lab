# Monitoring

Monitoring examples are implemented with synthetic/sample data.

The public version focuses on engineering patterns:

- input drift via PSI
- distribution shift via KS-style distance
- rolling return
- rolling Sharpe
- rolling max drawdown

Production alert rules and live trading operations remain private.

## Run

```bash
make monitoring-report
```

This writes:

```text
reports/monitoring_report.json
```

## Included Metrics

- `population_stability_index`
- `ks_distance`
- `feature_drift_report`
- `rolling_performance`

## Boundary

This repository does not include production alert thresholds, live order routing, broker state, or private model telemetry.
