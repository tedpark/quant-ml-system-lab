# Experiments And Registry

This repository includes a small local experiment tracker and manifest-only model registry for public demos.

The goal is to show the engineering pattern without exposing private model artifacts, production registry state, credentials, or strategy parameters.

## Run

```bash
make experiment-demo
```

This writes:

```text
reports/experiments.jsonl
reports/model_registry.json
reports/experiment_demo.json
```

Generated reports are local execution artifacts and are not committed.

## Included

- JSONL experiment run logging
- parameter and metric capture
- manifest-only model promotion
- disclosure fields that state no production artifact is included

## Excluded

- production checkpoints
- private registry URI
- credentials
- live trading configuration
- strategy search spaces
