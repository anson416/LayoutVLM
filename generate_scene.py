#!/usr/bin/env python
"""generate_scene.py -- generate a LayoutVLM scene from a textual prompt (Q6).

USAGE
-----
    python generate_scene.py \
        --prompt "a cozy beach-inspired bedroom, 4m x 5m, with a queen bed and a rattan chair" \
        --base_url https://api.openai.com/v1 \
        --api_key sk-... \
        --model gpt-4o \
        --temperature 0.0 \
        --asset_dir ./objaverse_processed \
        [--variants] [--asset_library ./asset_library.json] [--mock] [--seed 42]

WHAT IT DOES
------------
1. Turn the textual ``--prompt`` into a structured scene spec (boundary +
   asset shopping list) via one LLM call (Q1).  ``--mock`` skips the LLM and
   builds a minimal spec offline so the rest of the pipeline is exercisable
   without network/GPU.
2. Resolve the spec's asset requests against ``--asset_dir`` (processed
   Objaverse assets), then normalize via ``prepare_task_assets``.
3. Generate the *base* scene:
   * real mode: ``LayoutVLM.solve`` (LLM constraint programs + gradient solver);
   * ``--mock`` mode: random placements (no LLM/solver), so the CLI + variants
     are demonstrable without cost.
4. Save everything under ``outputs/<YYYYMMDD-HHMMSS>/``:
   ``config.json`` (prompt + LLM config + flags) and ``layout.json`` (base
   scene).  The prepared task is saved as ``prepared_task.json``.
5. If ``--variants`` is set, also write the four named content variants as
   sub-directories (Q2-Q5):
       variant_01_half / variant_02_biggest-only /
       variant_03_scrambled / variant_04_worst-object
   None of these re-run the LLM or solver -- they fork the base layout.

The run folder name uses ``datetime.datetime.now(datetime.UTC)`` and is
formatted ``YYYYMMDD-HHMMSS`` (e.g. 20260708-023434).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import sys
from typing import Dict, List

# Make repo root importable when run as a script.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vlmunr_scene_spec as scene_spec  # noqa: E402
import vlmunr_variants as variants  # noqa: E402
from main import prepare_task_assets  # noqa: E402


def _json_default(obj):
    """``json.dump`` default hook: coerce numpy/torch values to plain Python.

    The gradient solver may leave positions/rotations as ``torch.Tensor`` or
    ``numpy`` scalars; without this hook ``json.dump`` raises ``TypeError:
    Object of type Tensor is not JSON serializable``.  Unknown types fall back
    to ``str(obj)`` rather than crashing the whole generation.
    """

    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            return obj.item()
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass
    return str(obj)


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a LayoutVLM scene from a textual prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--prompt", required=True,
                   help="Textual scene description.")
    p.add_argument("--base_url", default="https://api.openai.com/v1",
                   help="OpenAI-compatible LLM base URL.")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"),
                   help="LLM API key (or set OPENAI_API_KEY).")
    p.add_argument("--model", default="gpt-4o",
                   help="LLM model name used for generation.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature, applied to all LLM calls.")
    p.add_argument("--asset_dir", default="./objaverse_processed",
                   help="Directory of processed Objaverse assets.")
    p.add_argument("--asset_library", default=None,
                   help="JSON asset library for the worst-object variant.")
    p.add_argument("--variants", action="store_true",
                   help="Also write the 4 named content variants.")
    p.add_argument("--mock", action="store_true",
                   help="Skip the LLM + gradient solver; produce a random "
                        "placement layout so the pipeline + variants run "
                        "offline without cost.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for variant generation.")
    p.add_argument("--outputs_dir", default="./outputs",
                   help="Root outputs directory.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Mock layout (no LLM / no solver)
# ---------------------------------------------------------------------------

def _mock_layout(task: Dict) -> Dict:
    """Place every asset at a random in-floor point with z = bbox.z/2.

    Mirrors the sandbox's initialize_variables convention so floor objects sit
    on the floor.  No LLM and no gradient solver are invoked.
    """

    from utils.placement_utils import get_random_placement
    floor = task["boundary"]["floor_vertices"]
    layout: Dict[str, Dict] = {}
    for inst_id, asset in task.get("assets", {}).items():
        bbox = asset.get("assetMetadata", {}).get("boundingBox", {}) or {}
        pos = get_random_placement(floor, add_z=True)
        if asset.get("onCeiling"):
            pos[-1] = 3.0
        else:
            pos[-1] = float(bbox.get("z", 0.0) or 0.0) / 2.0
        layout[inst_id] = {"position": pos, "rotation": [0.0, 0.0, 0.0]}
    return layout


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("ERROR: --api_key (or OPENAI_API_KEY) is required.", file=sys.stderr)
        return 2

    # --- run folder: outputs/<YYYYMMDD-HHMMSS> (UTC) ---
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(args.outputs_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)

    # api_key is stored redacted (prefix + suffix only) so the config records
    # *which* key was used for auditability without leaking the secret.
    _key = args.api_key or ""
    if len(_key) > 8:
        _redacted = f"{_key[:4]}...{_key[-4:]}"
    elif _key:
        _redacted = "***"
    else:
        _redacted = ""
    config = {
        "prompt": args.prompt,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "api_key_redacted": _redacted,
        "asset_dir": args.asset_dir,
        "asset_library": args.asset_library,
        "variants": args.variants,
        "mock": args.mock,
        "seed": args.seed,
        "timestamp_utc": stamp,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- 1. text -> scene spec (Q1) ---
    if args.mock:
        print("[1/4] mock mode: building scene spec offline (no LLM).")
        spec = scene_spec.mock_scene_spec(args.prompt)
    else:
        print("[1/4] generating scene spec from prompt via LLM ...")
        spec = scene_spec.generate_scene_spec(
            args.prompt,
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            temperature=args.temperature,
        )
    with open(os.path.join(run_dir, "scene_spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    print(f"      spec: {len(spec.get('asset_spec', []))} asset request(s), "
          f"{len(spec['boundary']['floor_vertices'])} floor verts.")

    # --- 2. resolve spec -> raw task, then normalize ---
    print("[2/4] resolving asset requests against asset_dir ...")
    task, warnings = scene_spec.resolve_asset_spec(spec, args.asset_dir)
    for w in warnings:
        print(f"      WARN: {w}")
    if not task["assets"]:
        print("ERROR: no assets resolved from asset_dir; cannot generate a "
              "scene. Pass --mock to exercise the pipeline with placeholders, "
              "or point --asset_dir at a populated processed-asset directory.",
              file=sys.stderr)
        return 3
    # prepare_task_assets reloads metadata from <asset_dir>/<uid>/data.json
    # and produces the prepared-task shape the solver/render expect.
    task = prepare_task_assets(task, args.asset_dir)
    with open(os.path.join(run_dir, "prepared_task.json"), "w") as f:
        json.dump(task, f, indent=2)

    # --- 3. generate the base layout ---
    if args.mock:
        print("[3/4] mock mode: placing assets at random floor points (no solver).")
        layout = _mock_layout(task)
    else:
        print("[3/4] generating base scene via LayoutVLM.solve (LLM + solver) ...")
        from src.layoutvlm.layoutvlm import LayoutVLM
        solver = LayoutVLM(
            mode="one_shot",
            save_dir=os.path.join(run_dir, "solve_run"),
            asset_source="objaverse",
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
        )
        layout = solver.solve(task)
    with open(os.path.join(run_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2, default=_json_default)
    print(f"      base scene: {len(layout)} placed instance(s).")

    # --- 4. variants (Q2-Q5), no re-solve ---
    if args.variants:
        print("[4/4] writing named variants (no LLM / no solver) ...")
        written = variants.generate_named_variants(
            run_dir, task, layout,
            seed=args.seed,
            asset_library_path=args.asset_library,
        )
        for w in written:
            print(f"      - {os.path.relpath(w, run_dir)}")
    else:
        print("[4/4] --variants not set; skipping variant generation.")

    print(f"\nDone. Run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
