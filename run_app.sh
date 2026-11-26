#!/usr/bin/env bash
# Lance l'interface ChurnGuard (Anaconda) — local ou serveur
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PY="${PYTHON:-/home/wasfade/anaconda3/bin/python}"
PORT="${PORT:-8501}"
ADDR="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
if [[ ! -f artifacts/churn_model.joblib ]]; then
  echo "→ Modèle absent : entraînement initial…"
  "$PY" -c "from app.churn_core import train_and_export; train_and_export()"
fi
exec "$PY" -m streamlit run app/streamlit_app.py \
  --server.port="$PORT" \
  --server.address="$ADDR" \
  "$@"
