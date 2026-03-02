#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PY="${ROOT_DIR}/venv/bin/python"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "venv not found. Run: python -m venv venv && pip install -r requirements.txt"
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

echo "Starting FastAPI on http://127.0.0.1:8000"
"${VENV_PY}" -m uvicorn app.main:app --reload &
API_PID=$!

echo "Starting Streamlit on http://localhost:8501"
"${VENV_PY}" -m streamlit run "${ROOT_DIR}/dashboard/streamlit_app.py"
