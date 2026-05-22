# Architecture

This public lab demonstrates the system shape without disclosing production strategy details.

```text
synthetic/sample data
  -> feature and spread construction
  -> baseline signal generation
  -> leakage-aware time split
  -> backtest with transaction costs
  -> risk metrics
  -> local experiment tracking
  -> manifest-only model registry
  -> serving and monitoring skeletons
```

The production boundary is intentional: live universes, feature recipes, thresholds, model checkpoints, broker code, and raw performance records are excluded.
