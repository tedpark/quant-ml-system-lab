# Deployment

The public deployment path serves only the sanitized demo API.

## Local

```bash
make serve
```

## Docker

```bash
docker build -t quant-ml-system-lab .
docker run --rm -p 8000:8000 quant-ml-system-lab
```

Endpoints:

```text
GET  /health
GET  /model/info
POST /predict
```

## Boundary

The Docker image does not include production checkpoints, broker integrations, private market data, or live trading configuration.
