#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${LAB3_PYTHON:-}" ]]; then
    lab_python="$LAB3_PYTHON"
elif python3 -c 'import mujoco' >/dev/null 2>&1; then
    lab_python=python3
elif [[ -x "$HOME/miniconda3/envs/hw2_autonomy/bin/python" ]]; then
    lab_python="$HOME/miniconda3/envs/hw2_autonomy/bin/python"
else
    printf '%s\n' 'MuJoCo was not found. Install requirements.txt and set LAB3_PYTHON to that environment’s Python.' >&2
    exit 1
fi
exec "$lab_python" run.py --mode sim "$@"
