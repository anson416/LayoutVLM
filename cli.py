#!/usr/bin/env python
"""cli.py -- generate a LayoutVLM scene from a textual description.

==============================================================================
EXTERNAL LOCAL RESOURCES THIS METHOD REQUIRES (and how to prepare them)
==============================================================================

LayoutVLM's generation path is NOT self-contained: it needs a set of processed
3D assets (objects), environment maps for lighting, and (optionally) a 3D-Front
texture set and an asset library for the worst-object variant. They are NOT
vendored in this repo (except the HDRIs). The full list:

1. PROCESSED OBJAVERSE ASSETS  -- REQUIRED for real generation
   -----------------------------------------------------------------
   A directory (default ./objaverse_processed, overridable with --asset_dir)
   of preprocessed Objaverse GLB meshes, laid out one sub-directory per asset:

       <asset_dir>/<uid>/                # <uid> = the Objaverse hash id
           <uid>.glb                      # the 3D mesh (meters, recentered)
           data.json                      # metadata (see schema below)

   data.json schema (the fields prepare_task_assets reads):
       {
         "annotations": {
           "category":   "bed",                 # short noun
           "description":"a queen bed",
           "onFloor":    true,                  # defaults true
           "onCeiling":   false,                # defaults false
           "onWall":      false,                # defaults false
           "onObject":    false,                # defaults false
           "frontView":   0                      # optional, defaults 0
         },
         "assetMetadata": {
           "boundingBox": { "x": w, "y": d, "z": h }   # meters
         }
       }

   How to prepare them if you don't have them:
     a. The raw assets come from the LayoutVLM authors' preprocessed dump:
          https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/
        Download and unzip it. It contains benchmark_tasks/ and a processed
        asset set; point --asset_dir at the latter.
     b. To build your own from raw Objaverse (advanced), follow the Holodeck /
        objathor preprocessing the repo references:
          https://github.com/allenai/Holodeck
          https://github.com/allenai/objathor
        Concretely: download the GLB from Objaverse, recenter to bounds, scale
        to meters, compute the axis-aligned bounding box, and write data.json
        with category/description (use a tagger/VLM) + the on* placement flags.
     c. For offline development WITHOUT real assets, pass --mock: the CLI then
        skips the spec/asset resolution and places placeholder assets at random
        floor points so the pipeline + all variants are exercisable with no
        network/GPU/Blender. (You still need --asset_dir to exist for the
        variant code paths that join layouts to assets; use a small synthetic
        dir -- the tests/ folder builds one as an example.)

2. HDRIs (ENVIRONMENT MAPS)  -- REQUIRED for real rendering; VENDORED
   ----------------------------------------------------------------
   8 .exr environment maps under ./vlmunr_hdri/ (city, courtyard, forest,
   interior, night, studio, sunrise, sunset) -- already in the repo, with
   license.txt. Override the directory with --hdri_dir. These light the scene
   during the in-loop Blender renders (_solve_single_group calls
   render_existing_scene -> load_hdri). No action needed unless you want to
   swap them.

3. 3D-FRONT OBJECT TEXTURES  -- OPTIONAL, only for .obj assets
   ----------------------------------------------------------------
   If your assets are 3D-Front .obj files (not Objaverse .glb), the renderer
   looks for a sibling texture.png next to each mesh (apply_3dfront_texture).
   This is set automatically by the existing code paths; no CLI flag is
   needed. Objaverse .glb assets carry their materials inside the GLB, so this
   is unused for the default pipeline. No preparation required for .glb.

4. ASSET LIBRARY (for the worst-object variant)  -- OPTIONAL
   ----------------------------------------------------------------
   variant_04_worst-object swaps each placed instance's asset identity to the
   WORST-matching candidate from a JSON asset library. Pass it with
   --asset_library. Schema (a JSON list):

       [
         {"category": "rock", "description": "a heavy boulder",
          "path": "/abs/path/to/boulder.glb",
          "assetMetadata": {"boundingBox": {"x": 2,"y": 2,"z": 2}}},
         ...
       ]

   If you omit --asset_library, that one variant is skipped (with a recorded
   reason); the other three variants still run.

5. BLENDER (bpy)  -- REQUIRED for real rendering; a Python dependency
   ----------------------------------------------------------------
   The in-loop renderer (utils/blender_render.py) imports bpy (the Blender
   Python API). The existing conda env "vlmunr" has bpy. On a fresh env,
   `pip install bpy` works on x86_64 linux and (older) macos arm64 wheels; on
   current macos arm64 there may be no wheel -- install Blender and use its
   bundled python, or reuse the "vlmunr" env. Without bpy, only --mock mode
   runs end to end.

6. ROTATED-IOU CUDA OP  -- OPTIONAL (degrades gracefully)
   ----------------------------------------------------------------
   third_party/Rotated_IoU provides oriented_iou_loss for the overlap term in
   the gradient solver. If it isn't compiled, constraints.py falls back to a
   simpler overlap loss (with a warning) -- generation still works, slightly
   less accurate. To build it (CUDA GPU required):
       cd third_party/Rotated_IoU/cuda_op && python setup.py install

7. OPENAI-COMPATIBLE LLM ENDPOINT  -- REQUIRED for real (non-mock) generation
   ----------------------------------------------------------------
   The constraint-program LLM calls go through an OpenAI-compatible chat API.
   Pass --base_url, --api_key, --model, --temperature. Any compatible
   provider works (OpenAI, an OpenAI-compatible proxy, a local vLLM/LiteLLM,
   etc.). --mock skips all LLM calls.

------------------------------------------------------------------------------
SUMMARY OF CLI ARGS MAPPING TO THE ABOVE
------------------------------------------------------------------------------
  --asset_dir      -> resource #1 (REQUIRED for real mode)
  --hdri_dir       -> resource #2 (default ./vlmunr_hdri, vendored)
  --asset_library  -> resource #4 (optional; for variant_04)
  --base_url/--api_key/--model/--temperature -> resource #7 (LLM)
  --mock           -> skip #1/#5/#6/#7 entirely (offline demo)
  --variants       -> also write the 4 named content variants

==============================================================================
USAGE
==============================================================================

Real generation (needs processed Objaverse assets + an LLM endpoint + bpy):

    python cli.py \
        --prompt "a cozy beach-inspired bedroom, 4m x 5m, with a queen bed and a rattan chair" \
        --base_url https://api.openai.com/v1 \
        --api_key sk-... \
        --model gpt-4o \
        --temperature 0.0 \
        --asset_dir ./objaverse_processed \
        --hdri_dir ./vlmunr_hdri \
        --variants

Offline demonstration (no LLM / no solver / no bpy; uses placeholder assets):

    python cli.py --prompt "a cozy bedroom" --api_key sk-dummy \
        --mock --variants --asset_dir ./tests/_synthetic_assets

OUTPUT
------
Everything is written under outputs/<YYYYMMDD-HHMMSS UTC>/:
  config.json          prompt + LLM config (api key stored REDACTED only)
  scene_spec.json      the LLM-produced scene spec (boundary + asset list)
  prepared_task.json   resolved + normalized task the solver consumes
  layout.json          the BASE scene (placed instances)
  solve_run/           (real mode only) per-group solver artifacts
  variant_01_half/           keep round(n/2) instances (seeded)
  variant_02_biggest-only/   keep the single largest instance (bbox volume)
  variant_03_scrambled/      relocate every instance within the floor polygon
  variant_04_worst-object/    swap asset identity to worst-match library cand
                             (only when --asset_library is given)
None of the variants re-run the LLM or the gradient solver; they fork layout.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import List

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


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a LayoutVLM scene from a textual prompt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- the textual scene description ---
    p.add_argument("--prompt", required=True,
                   help="Textual scene description.")

    # --- LLM endpoint (resource #7) ---
    p.add_argument("--base_url", default="https://api.openai.com/v1",
                   help="OpenAI-compatible LLM base URL.")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"),
                   help="LLM API key (or set OPENAI_API_KEY).")
    p.add_argument("--model", default="gpt-4o",
                   help="LLM model name used for generation.")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="LLM sampling temperature, applied to all LLM calls.")

    # --- external local resources (resources #1, #2, #4) ---
    p.add_argument("--asset_dir", default="./objaverse_processed",
                   help="Processed Objaverse assets dir: <uid>/<uid>.glb + "
                        "data.json (REQUIRED for real mode).")
    p.add_argument("--hdri_dir", default=os.path.join(ROOT, "vlmunr_hdri"),
                   help="Directory of .exr environment maps (vendored by "
                        "default). Used only in real (non-mock) rendering.")
    p.add_argument("--asset_library", default=None,
                   help="JSON asset library for the worst-object variant "
                        "(list of {category,description,path[,assetMetadata]}).")

    # --- behavior flags ---
    p.add_argument("--variants", action="store_true",
                   help="Also write the 4 named content variants.")
    p.add_argument("--mock", action="store_true",
                   help="Skip the LLM + gradient solver; produce a random "
                        "placement layout so the pipeline + variants run "
                        "offline without cost (no bpy/GPU/LLM needed).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for variant generation.")
    p.add_argument("--outputs_dir", default="./outputs",
                   help="Root outputs directory.")
    return p.parse_args(argv)


def _mock_layout(task) -> dict:
    """Place every asset at a random in-floor point with z = bbox.z/2.

    Mirrors the sandbox's initialize_variables convention so floor objects sit
    on the floor.  No LLM and no gradient solver are invoked.
    """

    from utils.placement_utils import get_random_placement
    floor = task["boundary"]["floor_vertices"]
    layout = {}
    for inst_id, asset in task.get("assets", {}).items():
        bbox = asset.get("assetMetadata", {}).get("boundingBox", {}) or {}
        pos = get_random_placement(floor, add_z=True)
        if asset.get("onCeiling"):
            pos[-1] = 3.0
        else:
            pos[-1] = float(bbox.get("z", 0.0) or 0.0) / 2.0
        layout[inst_id] = {"position": pos, "rotation": [0.0, 0.0, 0.0]}
    return layout


def _check_external_resources(args, *, mock: bool) -> List[str]:
    """Warn (not fail) about missing external resources for the chosen mode.

    Returns the list of warning strings. In real mode, missing assets/HDRIs
    cause a later failure with a clearer message; in mock mode we stay silent
    about assets the mock path doesn't actually use.
    """

    warnings = []
    if mock:
        return warnings
    if not os.path.isdir(args.asset_dir):
        warnings.append(
            f"--asset_dir {args.asset_dir!r} does not exist; real generation "
            "needs processed Objaverse assets (see the module docstring)."
        )
    else:
        # Spot-check that at least one data.json is present so the failure is
        # surfaced early instead of after the LLM spec call.
        sample = next(
            (os.path.join(args.asset_dir, n, "data.json")
             for n in os.listdir(args.asset_dir)
             if os.path.exists(os.path.join(args.asset_dir, n, "data.json"))),
            None,
        )
        if sample is None:
            warnings.append(
                f"--asset_dir {args.asset_dir!r} has no <uid>/data.json entries; "
                "it does not look like a processed Objaverse directory."
            )
    if not os.path.isdir(args.hdri_dir):
        warnings.append(
            f"--hdri_dir {args.hdri_dir!r} does not exist; in-loop rendering "
            "will fail to light the scene."
        )
    if args.variants and not args.asset_library:
        warnings.append(
            "--variants set without --asset_library: variant_04_worst-object "
            "will be skipped (the other three variants still run)."
        )
    return warnings


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)

    if not args.api_key:
        print("ERROR: --api_key (or OPENAI_API_KEY) is required.",
              file=sys.stderr)
        return 2

    # Surface missing-external-resource problems early.
    for w in _check_external_resources(args, mock=args.mock):
        print(f"  WARN: {w}")

    # --- run folder: outputs/<YYYYMMDD-HHMMSS> (UTC) ---
    # datetime.UTC exists on 3.11+; datetime.timezone.utc is the 3.10-compatible
    # spelling of the same object, so the call matches the requested
    # datetime.datetime.now(datetime.UTC) form across versions.
    _utc = getattr(datetime, "UTC", None) or datetime.timezone.utc
    stamp = datetime.datetime.now(_utc).strftime("%Y%m%d-%H%M%S")
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
        "hdri_dir": args.hdri_dir,
        "asset_library": args.asset_library,
        "variants": args.variants,
        "mock": args.mock,
        "seed": args.seed,
        "timestamp_utc": stamp,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- 1. text -> scene spec ---
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
        from src.layoutvlm.layoutvlm import LayoutVLM  # lazy: pulls in bpy + torch
        # Expose hdri_dir to the renderer via env (load_hdri resolves a path;
        # the existing vendored dir is the default the code expects).
        os.environ.setdefault("VLMUNR_HDRI_DIR", args.hdri_dir)
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
