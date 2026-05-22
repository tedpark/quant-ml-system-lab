# Serving

Serving examples are implemented as schema-first skeletons.

The public version demonstrates:

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
src/quant_ml_lab/api.py
```

This is intentionally not a live trading API. It is a safe public contract wrapped by FastAPI without exposing production models.

## Run

```bash
make serve
```

Endpoints:

```text
GET  /health
GET  /model/info
POST /predict
GET  /metrics
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-1","features":[0.4,0.3,0.2]}'
```

Latency benchmark:

```bash
make latency-benchmark
```

Monitoring snapshot:

```bash
curl http://127.0.0.1:8000/metrics
```
