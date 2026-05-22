# Serving

Serving examples are implemented as schema-first skeletons.

The public version will demonstrate:

- request/response schema validation
- model metadata endpoint
- deterministic demo prediction contract
- explicit disclosure text in responses

The public version will not include:

- production checkpoints
- live broker integration
- production routing rules
- private model registry configuration

Current module:

```text
src/quant_ml_lab/serving.py
```

This is intentionally not a live trading API. It is a safe public contract that can later be wrapped by FastAPI without exposing production models.
