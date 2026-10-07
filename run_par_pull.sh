#!/usr/bin/env bash
# Start the ERA5 pull in the background on a VM; survives SSH logout.
#
# Usage:
#   ./run_pull.sh                 # uv run pull_era5.py --c3dir-only
#   ./run_pull.sh --step-hours 3  # extra args are passed through to pull_era5.py
#
# Monitor:  tail -f output.log
# Stop:     kill "$(cat pull_era5.pid)"
# Reruns resume: chunks already in data/era5/chunks are skipped.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

command -v uv >/dev/null 2>&1 || { echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; exit 1; }
[ -f era5_secrets.txt ] || [ -f "$HOME/.cdsapirc" ] || { echo "Missing era5_secrets.txt (or ~/.cdsapirc) with your CDS key." >&2; exit 1; }

if [ -f pull_era5.pid ] && kill -0 "$(cat pull_era5.pid)" 2>/dev/null; then
  echo "Already running (PID $(cat pull_era5.pid)). Stop it first or tail output.log." >&2
  exit 1
fi

uv sync
nohup uv run par_pull_era5.py --c3dir-only "$@" > output.log 2>&1 &
echo $! > pull_era5.pid
echo "Started in background, PID $(cat pull_era5.pid). Follow with: tail -f output.log"
