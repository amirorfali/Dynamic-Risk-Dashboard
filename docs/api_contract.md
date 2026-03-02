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
    }
  },
  "additionalProperties": false
}
```

Response body JSON schema:
```json
{
  "type": "object",
  "required": ["var", "cvar"],
  "properties": {
    "var": {
      "type": "number",
      "description": "Value at Risk for the requested horizon."
    },
    "cvar": {
      "type": "number",
      "description": "Conditional Value at Risk for the requested horizon."
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
  "tail_threshold": 0.02
}
```

Example response:
```json
{
  "var": 0.0,
  "cvar": 0.0
}
```

Errors:
- `422` validation error when request does not match schema.
