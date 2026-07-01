#!/usr/bin/env bash
#
# gen.sh -- generate LayoutVLM layouts for every benchmark task.
#
# Thin driver: loops over benchmark_tasks/<category>/<task>.json and calls
# main.py to produce a save dir containing layout.json (joined back to the
# task JSON downstream by the vlmunr_* tools).
#
# Requirements (NOT handled here): processed Objaverse assets under ASSET_DIR
# and an OpenAI API key.  This script does NOT download any models/assets.
#
# Usage:
#   OPENAI_API_KEY=sk-... ASSET_DIR=./objaverse_processed ./gen.sh [RESULTS_DIR]
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"
RESULTS_DIR="${1:-${HERE}/results}"
ASSET_DIR="${ASSET_DIR:-${HERE}/objaverse_processed}"
MODEL="${MODEL:-gpt-4}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: set OPENAI_API_KEY" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

for task_json in "${HERE}"/benchmark_tasks/*/*.json; do
    rel="${task_json#${HERE}/benchmark_tasks/}"   # category/task.json
    name="${rel%.json}"                            # category/task
    save_dir="${RESULTS_DIR}/${name//\//_}"
    echo "==> generating ${name} -> ${save_dir}"
    "${PY}" "${HERE}/main.py" \
        --scene_json_file "${task_json}" \
        --save_dir "${save_dir}" \
        --model "${MODEL}" \
        --asset_dir "${ASSET_DIR}" \
        --openai_api_key "${OPENAI_API_KEY}"
    # Keep a copy of the input task JSON beside layout.json for the audit tools.
    cp "${task_json}" "${save_dir}/task.json"
done

echo "Done. Layouts under ${RESULTS_DIR}"
