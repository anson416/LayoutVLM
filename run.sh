#!/usr/bin/env bash
#
# run.sh -- generate content variants and render the audit sweep for every
# generated scene.
#
# Thin driver: for each save dir (must contain a task JSON + layout.json) it
#   1. calls vlmunr_variants.py to write the variant sibling dirs, then
#   2. calls vlmunr_render.py on the base scene AND each variant dir.
#
# Usage:
#   ./run.sh [RESULTS_DIR] [PHASE]
#     RESULTS_DIR : dir holding the per-scene save dirs (default ./results)
#     PHASE       : 1a|1b|1b_chroma|1c|1d|2|2_pitch|2_yaw|all (default 1a)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-/Users/anson/miniforge3/envs/vlmunr/bin/python}"
RESULTS_DIR="${1:-${HERE}/results}"
PHASE="${2:-1a}"
SEED="${SEED:-42}"

render_dir() {
    local d="$1"
    [[ -f "${d}/layout.json" ]] || return 0
    echo "    rendering ${d} (phase ${PHASE})"
    "${PY}" "${HERE}/vlmunr_render.py" --scene-dir "${d}" --phase "${PHASE}"
}

for scene_dir in "${RESULTS_DIR}"/*/; do
    scene_dir="${scene_dir%/}"
    [[ -f "${scene_dir}/layout.json" ]] || continue
    # Skip variant dirs themselves when iterating top-level scenes.
    case "$(basename "${scene_dir}")" in
        variant_*) continue ;;
    esac

    echo "==> ${scene_dir}"
    "${PY}" "${HERE}/vlmunr_variants.py" --scene-dir "${scene_dir}" --seed "${SEED}"

    # Render base scene.
    render_dir "${scene_dir}"

    # Render each variant sibling.
    parent="$(dirname "${scene_dir}")"
    for v in variant_half variant_quarter variant_eighth \
             variant_alt_0 variant_alt_2 variant_alt_4 \
             variant_subst_within variant_subst_cross variant_scramble; do
        render_dir "${parent}/${v}"
    done
done

echo "Done."
