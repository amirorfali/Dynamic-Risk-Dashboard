# API Contract

Base path: `/api`

## Conventions
- Loss sign: `L = -w^T R * Notional` (positive = loss).
- Return model (MVP): multivariate normal with optional crash mixture.
- Tail threshold: user-provided `tail_threshold` if supplied; otherwise default to `VaR_99`.

## POST `/risk`

Purpose: Compute portfolio risk metrics for the given horizon and model choice.

Request body JSON schema:
```json
{
  "type": "object",
  "required": ["portfolio", "horizon_days"],
  "properties": {
    "portfolio": {
      "type": "object",
      "description": "Portfolio definition. Structure is intentionally flexible until the data model is finalized."
    },
    "horizon_days": {
      "type": "integer",
      "minimum": 1,
      "description": "Risk horizon in trading days."
    },
    "return_model": {
      "type": "string",
      "enum": ["normal", "normal_crash_mixture"],
      "description": "Return model selection. Default is normal."
    },
    "tail_threshold": {
      "type": "number",
      "description": "Loss threshold ℓ. If omitted, defaults to VaR at 99%."
    },
    "quantum_num_qubits": {
      "type": "integer",
      "minimum": 1,
      "maximum": 12,
      "description": "When backend is quantum, discretize losses into 2**quantum_num_qubits bins."
    }
  },
  "additionalProperties": false
}
```

Response body JSON schema:
```json
{
  "type": "object",
  "required": ["var", "cvar", "mean", "vol", "histogram", "backend"],
  "properties": {
    "var": {
      "type": "number",
      "description": "Value at Risk for the requested horizon."
    },
    "cvar": {
      "type": "number",
      "description": "Conditional Value at Risk for the requested horizon."
    },
    "mean": {
      "type": "number",
      "description": "Mean loss over the horizon."
    },
    "vol": {
      "type": "number",
      "description": "Loss volatility over the horizon."
    },
    "histogram": {
      "type": "object",
      "required": ["bin_edges", "counts"],
      "properties": {
        "bin_edges": {
          "type": "array",
          "items": { "type": "number" }
        },
        "counts": {
          "type": "array",
          "items": { "type": "integer" }
        }
      }
    },
    "backend": {
      "type": "object",
      "required": [
        "runtime_ms",
        "n_paths",
        "model",
        "cache_hit",
        "data_source",
        "window_days"
      ],
      "properties": {
        "runtime_ms": { "type": "number" },
        "n_paths": { "type": "integer" },
        "model": { "type": "string" },
        "cache_hit": { "type": "boolean" },
        "data_source": { "type": "string" },
        "window_days": { "type": ["integer", "null"] },
        "crash_params": {
          "type": ["object", "null"],
          "properties": {
            "pc": { "type": "number" },
            "mean_shift": { "type": "number" },
            "vol_jump": { "type": "number" }
          }
        }
      }
    }
  },
  "additionalProperties": false
}
```

Example request:
```json
{
  "portfolio": {
    "AAPL": 0.5,
    "MSFT": 0.5
  },
  "horizon_days": 10,
  "return_model": "normal_crash_mixture",
  "tail_threshold": 0.02,
  "quantum_num_qubits": 5
}
```

Example response:
```json
{
  "var": 0.0,
  "cvar": 0.0,
  "mean": 0.0,
  "vol": 0.0,
  "histogram": {
    "bin_edges": [0.0, 0.1, 0.2],
    "counts": [10, 5]
  },
  "backend": {
    "runtime_ms": 12.3,
    "n_paths": 5000,
    "model": "normal",
    "cache_hit": false,
    "data_source": "yfinance",
    "window_days": 252,
    "crash_params": null
  }
}
```

Errors:
- `422` validation error when request does not match schema.
