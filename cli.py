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
  --variants       -> also write the 4 named content variants (--prompt only)

==============================================================================
USAGE
==============================================================================

INPUT MODES (mutually exclusive; exactly one required):
  --prompt TEXT   generate a fresh scene (+ variants with --variants) into a
                  new outputs/<timestamp>/ folder
  --path DIR      render the scene folders (base/ + variant_*) ALREADY present
                  under DIR; no generation occurs. Every scene folder is
                  self-contained (meshes copied in, paths rewritten to
                  ./meshes/...), so it renders without --asset_dir.

RENDER MODES (mutually exclusive; optional; combine with either input mode):
  --render        single baseline image: 512px / 50mm / pitch 0 (top-down) /
                  yaw 0 / city env / white (255,255,255) bg. Saves BOTH the
                  transparent master and the white composite.
  --render-all    six single-axis sweeps: resolution / focal / pitch /
                  yaw(at pitch 45) / env / background. See vlmunr_config.

Real generation (needs processed Objaverse assets + an LLM endpoint + bpy):

    python cli.py \
        --prompt "a cozy beach-inspired bedroom, 4m x 5m, with a queen bed and a rattan chair" \
        --base_url https://api.openai.com/v1 \
        --api_key sk-... \
        --model gpt-4o \
        --temperature 0.0 \
        --asset_dir ./objaverse_processed \
        --hdri_dir ./vlmunr_hdri \
        --variants --render-all

Offline demonstration (no LLM / no solver; uses placeholder assets):

    python cli.py --prompt "a cozy bedroom" --api_key sk-dummy \
        --mock --variants --asset_dir ./tests/_synthetic_assets

Re-render an already-generated run at the baseline point (no generation):

    python cli.py --path outputs/20260708-131722 --render

OUTPUT
------
Everything is written under outputs/<YYYYMMDD-HHMMSS UTC>/:

  config.json                 prompt + LLM config (api key stored REDACTED only)
  base/                       the BASE scene (the generated layout)
      task.json               resolved + normalized task the solver consumes
      layout.json             placed instances (the scene)
      scene_spec.json         the LLM-produced scene spec (boundary + asset list)
      meshes/                 the GLB meshes referenced by the scene, copied in
                              so the folder is self-contained (no external
                              asset_dir needed to view/render it)
      renderings/             (only with --render / --render-all) PNG renders
      solve_run/              (real mode only) per-group solver artifacts
  variant_01_half/            keep round(n/2) instances (seeded)
  variant_02_biggest-only/    keep the single largest instance (bbox volume)
  variant_03_scrambled/       relocate every instance within the floor polygon
  variant_04_worst-object/    swap asset identity to worst-match library cand
                              (only when --asset_library is given)

Each variant directory has the SAME internal structure as base/ (task.json,
layout.json, meshes/, and renderings/ when a render flag is set), so every
scene -- base or variant -- is independently renderable and portable. None of
the variants re-run the LLM or the gradient solver; they fork the base layout.

The renderings sub-folder is ALWAYS named "renderings" (the renderer writes
there); it is only created when there are actually renders to write (i.e. when
--render / --render-all is passed and bpy is available), so an absent
renderings/ folder means "not rendered", never an error.

Render filenames (per the bpa convention: transparent master rendered once per
(res,focal,pitch,yaw,env), then each bg composited onto it; fit_ratio=1):

  render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png
  render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png

pitch is the literal bpa pitch (0 = top-down). The architectural shell (floor +
walls) is retained and rendered dollhouse-style: camera-facing walls are made
transparent (back-face culling) so the camera always sees into the room while
far walls stay visible. LayoutVLM specs carry only a floor footprint + wall
height, so there is no door/window opening geometry; the shell is neutral
floor+walls.
"""

from __future__ import annotations

import argparse
import copy as _copy
import datetime
import json
import os
import sys
from typing import Dict, List, Optional

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
        description="Generate a LayoutVLM scene from a textual prompt, "
                    "or render scenes already written under a run folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- input mode: generate-from-prompt OR render-from-path (never both) ---
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompt",
                     help="Textual scene description. Generates a fresh scene "
                          "(+ variants with --variants) under a new "
                          "outputs/<timestamp>/ folder.")
    src.add_argument("--path",
                     help="Path to an EXISTING run folder "
                          "(outputs/<timestamp>/). No generation occurs; only "
                          "the already-present scene folders (base/ and every "
                          "variant_*) are rendered.")

    # --- LLM endpoint (resource #7) ---
    p.add_argument("--base_url", default="https://api.openai.com/v1",
                   help="OpenAI-compatible LLM base URL.")
    p.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"),
                   help="LLM API key (or set OPENAI_API_KEY). Required for "
                        "--prompt (real mode); ignored for --path.")
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
                        "default). Used for --render / --render-all lighting.")
    p.add_argument("--asset_library", default=None,
                   help="JSON asset library for the worst-object variant "
                        "(list of {category,description,path[,assetMetadata]}).")

    # --- behavior flags ---
    p.add_argument("--variants", action="store_true",
                   help="Also write the 4 named content variants (--prompt "
                        "mode only).")
    # --- render mode: single image OR full sweep (never both) ---
    render = p.add_mutually_exclusive_group()
    render.add_argument("--render", action="store_true",
                        help="Render each scene at the baseline point only: "
                             "512px, 50mm, pitch 0 (top-down), yaw 0, 'city' "
                             "env map, white (255,255,255) bg. Saves BOTH the "
                             "transparent master and the white composite.")
    render.add_argument("--render-all", action="store_true",
                        help="Render each scene across six single-axis sweeps "
                             "(resolution / focal / pitch / yaw / env / bg). "
                             "See vlmunr_config.all_sweep_specs for the exact "
                             "levels. Renders are best-effort: a failure for "
                             "one scene is logged and the rest still render.")
    p.add_argument("--mock", action="store_true",
                   help="Skip the LLM + gradient solver; produce a random "
                        "placement layout so the pipeline + variants run "
                        "offline without cost (no bpy/GPU/LLM needed).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for variant generation.")
    p.add_argument("--outputs_dir", default="./outputs",
                   help="Root outputs directory (--prompt mode only).")
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


def _make_self_contained(
    task: dict,
    layout: dict,
    scene_dir: str,
    *,
    scene_spec_dict: Optional[dict] = None,
    solve_run_dir: Optional[str] = None,
) -> dict:
    """Write one scene (base or variant) as a self-contained sub-folder.

    Layout::

        <scene_dir>/
            task.json          the task the solver/renderer consume
            layout.json        placed instances
            scene_spec.json    (only when scene_spec_dict is given; base only)
            meshes/            GLB meshes referenced by the scene, copied in
            solve_run/         (only when solve_run_dir is given; base only)

    Each referenced GLB (``asset.path``) is copied into ``meshes/`` with a
    unique flat name, and the saved task's paths are rewritten to
    ``./meshes/<name>`` so the folder is portable -- it carries every mesh it
    needs and no longer depends on the external ``--asset_dir``.  Meshes that
    do not exist on disk are skipped (with a warning); the asset entry keeps
    its original path so the renderer can decide what to do.  The returned
    task is the rewritten copy (also what gets written to task.json).
    """

    import shutil

    os.makedirs(scene_dir, exist_ok=True)
    meshes_dir = os.path.join(scene_dir, "meshes")

    # Deduplicate copies by source path so a mesh shared across instances is
    # written once.
    seen: Dict[str, str] = {}
    out_task = _copy.deepcopy(task)
    for inst_id, asset in out_task.get("assets", {}).items():
        src = asset.get("path")
        if not src or not os.path.exists(src):
            if src:
                print(f"      WARN: mesh not found, not copied: {src} "
                      f"({inst_id})")
            continue
        if src in seen:
            asset["path"] = seen[src]
            continue
        ext = os.path.splitext(src)[1] or ".glb"
        flat = f"{inst_id}{ext}"
        # Disambiguate against unlikely collisions across instances.
        dst_name = flat
        i = 1
        while os.path.exists(os.path.join(meshes_dir, dst_name)) and \
                seen.get(src) != f"./meshes/{dst_name}":
            # Only rename if the existing file came from a *different* source.
            i += 1
            dst_name = f"{inst_id}_{i}{ext}"
        os.makedirs(meshes_dir, exist_ok=True)
        try:
            shutil.copy2(src, os.path.join(meshes_dir, dst_name))
        except Exception as exc:  # pragma: no cover - filesystem dependent
            print(f"      WARN: could not copy mesh {src} ({exc})")
            continue
        rel = f"./meshes/{dst_name}"
        seen[src] = rel
        asset["path"] = rel

    with open(os.path.join(scene_dir, "task.json"), "w") as f:
        json.dump(out_task, f, indent=2, default=_json_default)
    with open(os.path.join(scene_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2, default=_json_default)
    if scene_spec_dict is not None:
        with open(os.path.join(scene_dir, "scene_spec.json"), "w") as f:
            json.dump(scene_spec_dict, f, indent=2, default=_json_default)
    if solve_run_dir and os.path.isdir(solve_run_dir):
        # Move the solver's scratch dir into the base scene folder.
        dst = os.path.join(scene_dir, "solve_run")
        if os.path.abspath(solve_run_dir) != os.path.abspath(dst):
            try:
                shutil.move(solve_run_dir, dst)
            except Exception:
                pass
    return out_task


def _render_specs_for(args) -> List[dict]:
    """The render spec list selected by --render / --render-all.

    Returns an empty list when neither flag is set (no rendering).
    """

    import vlmunr_config as _cfg
    if args.render_all:
        return _cfg.all_sweep_specs()
    if args.render:
        return [_cfg.single_render_spec()]
    return []


def _render_scene(
    task: dict, layout: dict, scene_dir: str, args, specs: List[dict],
    reason: str,
) -> List[str]:
    """Render one scene's specs into ``<scene_dir>/renderings/``.

    The renderings folder is ALWAYS named ``renderings`` (the vlmunr_render
    renderer writes there); it is only created when there are actually renders
    to write.  Rendering is best-effort: any failure (no bpy, a missing mesh,
    an HDRI problem) is logged and skipped so the rest of the pipeline still
    completes.  Returns the list of PNG paths written (empty on failure).
    """

    if not specs:
        return []
    try:
        import vlmunr_render as _render  # noqa: WPS433 (lazy: pulls in bpy)
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"      WARN: could not import vlmunr_render ({exc}); "
              f"skipping rendering for {reason}.")
        return []
    try:
        # Self-contained scene folders store meshes at "./meshes/..."; the
        # renderer resolves those against CWD, so chdir into the scene folder
        # for the duration of the render.
        prev_cwd = os.getcwd()
        os.chdir(scene_dir)
        try:
            written = _render.render_specs(
                task, layout, scene_dir, specs,
                env_strength=1.0, asset_dir=None, hdri_dir=args.hdri_dir,
            )
        finally:
            os.chdir(prev_cwd)
        return written
    except Exception as exc:  # pragma: no cover - rendering is best-effort
        print(f"      WARN: rendering failed for {reason} ({exc}); "
              "scene + meshes still saved.")
        return []


def _iter_scene_dirs(run_dir: str) -> List[str]:
    """Return the scene folders to render under a run dir (in fixed order).

    Always includes ``base`` first (if present), then every ``variant_*``
    sub-folder sorted lexically.  Non-scene entries (config.json, solve_run/,
    loose files) are ignored.
    """

    if not os.path.isdir(run_dir):
        return []
    scenes: List[str] = []
    base = os.path.join(run_dir, "base")
    if os.path.isdir(base):
        scenes.append(base)
    try:
        names = sorted(n for n in os.listdir(run_dir) if n.startswith("variant_"))
    except OSError:
        names = []
    for n in names:
        d = os.path.join(run_dir, n)
        if os.path.isdir(d):
            scenes.append(d)
    return scenes


def _render_path_mode(args) -> int:
    """``--path`` mode: render every already-present scene folder, no generation.

    Each scene folder (base/ and every variant_*) carries its own task.json +
    layout.json + meshes/, so it is rendered independently.  Rendering is the
    whole point of this mode, so a missing bpy is a hard error (not graceful).
    """

    run_dir = os.path.abspath(args.path)
    if not os.path.isdir(run_dir):
        print(f"ERROR: --path {run_dir!r} is not a directory.", file=sys.stderr)
        return 2
    specs = _render_specs_for(args)
    if not specs:
        print("ERROR: --path requires --render or --render-all.",
              file=sys.stderr)
        return 2
    scenes = _iter_scene_dirs(run_dir)
    if not scenes:
        print(f"ERROR: no scene folders (base/ or variant_*) found under "
              f"{run_dir!r}.", file=sys.stderr)
        return 3
    # Hard-fail on missing bpy here: in --path mode there is nothing else to do.
    try:
        import vlmunr_render as _render  # noqa: F401
    except Exception as exc:
        print(f"ERROR: bpy/vlmunr_render unavailable ({exc}); cannot render.",
              file=sys.stderr)
        return 4

    n_specs = len(specs)
    print(f"rendering {len(scenes)} scene(s) from {run_dir} "
          f"({'--render-all' if args.render_all else '--render'}, "
          f"{n_specs} spec(s)).")
    total = 0
    for scene_dir in scenes:
        name = os.path.basename(scene_dir)
        try:
            with open(os.path.join(scene_dir, "task.json")) as f:
                task = json.load(f)
            with open(os.path.join(scene_dir, "layout.json")) as f:
                layout = json.load(f)
        except FileNotFoundError as exc:
            print(f"  - {name}: SKIP (missing {exc.filename})")
            continue
        written = _render_scene(task, layout, scene_dir, args, specs, name)
        print(f"  - {name}: {len(written)} image(s) -> renderings/")
        total += len(written)
    print(f"\nDone. Rendered {total} image(s) across {len(scenes)} scene(s).")
    return 0


def main(argv: List[str] = None) -> int:
    args = parse_args(argv)

    # --- --path mode: render already-present scenes, no generation ---
    if args.path:
        return _render_path_mode(args)

    # --- --prompt mode: generate a fresh scene ---
    if not args.api_key:
        print("ERROR: --api_key (or OPENAI_API_KEY) is required for --prompt.",
              file=sys.stderr)
        return 2

    # Surface missing-external-resource problems early.
    for w in _check_external_resources(args, mock=args.mock):
        print(f"  WARN: {w}")

    # Resolve the render spec set once (--render / --render-all / none).
    _specs = _render_specs_for(args)

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
        "render": args.render,
        "render_all": args.render_all,
        "timestamp_utc": stamp,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Step count varies with --variants, so the [i/N] labels are computed.
    _n_steps = 4 if args.variants else 3

    # --- 1. text -> scene spec ---
    if args.mock:
        print("[1/%d] mock mode: building scene spec offline (no LLM)." % _n_steps)
        spec = scene_spec.mock_scene_spec(args.prompt)
    else:
        print("[1/%d] generating scene spec from prompt via LLM ..." % _n_steps)
        spec = scene_spec.generate_scene_spec(
            args.prompt,
            model=args.model, base_url=args.base_url, api_key=args.api_key,
            temperature=args.temperature,
        )
    print(f"      spec: {len(spec.get('asset_spec', []))} asset request(s), "
          f"{len(spec['boundary']['floor_vertices'])} floor verts.")

    # --- 2. resolve spec -> raw task, then normalize ---
    print("[2/%d] resolving asset requests against asset_dir ..." % _n_steps)
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

    # --- 3. generate the base layout ---
    if args.mock:
        print("[3/%d] mock mode: placing assets at random floor points "
              "(no solver)." % _n_steps)
        layout = _mock_layout(task)
    else:
        print("[3/%d] generating base scene via LayoutVLM.solve "
              "(LLM + solver) ..." % _n_steps)
        from src.layoutvlm.layoutvlm import LayoutVLM  # lazy: bpy + torch
        # The solver writes scratch into base/solve_run; _make_self_contained
        # relocates it into the base scene folder below.
        solve_run_dir = os.path.join(run_dir, "solve_run")
        os.environ.setdefault("VLMUNR_HDRI_DIR", args.hdri_dir)
        solver = LayoutVLM(
            mode="one_shot",
            save_dir=solve_run_dir,
            asset_source="objaverse",
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
        )
        layout = solver.solve(task)
    print(f"      base scene: {len(layout)} placed instance(s).")

    # --- write the BASE scene as a self-contained sub-folder ---
    print(f"      writing base scene -> base/")
    base_dir = os.path.join(run_dir, "base")
    solve_run_dir = (
        os.path.join(run_dir, "solve_run")
        if not args.mock and os.path.isdir(os.path.join(run_dir, "solve_run"))
        else None
    )
    base_task = _make_self_contained(
        task, layout, base_dir,
        scene_spec_dict=spec, solve_run_dir=solve_run_dir,
    )
    _rc = _render_scene(base_task, layout, base_dir, args, _specs, "base")
    if _rc:
        print(f"      rendered {len(_rc)} image(s) -> base/renderings")

    # --- 4. variants (Q2-Q5), no re-solve ---
    if args.variants:
        print("[4/%d] writing named variants (no LLM / no solver) ..."
              % _n_steps)
        var_layouts = variants.build_named_variants(
            task, layout,
            seed=args.seed,
            asset_library_path=args.asset_library,
        )
        for name, v_task, v_layout, v_intent in var_layouts:
            v_dir = os.path.join(run_dir, name)
            v_task = _make_self_contained(v_task, v_layout, v_dir)
            if v_intent is not None:
                with open(os.path.join(v_dir, "variant_intent.json"), "w") as f:
                    json.dump(v_intent, f, indent=2, default=_json_default)
            rc_imgs = _render_scene(v_task, v_layout, v_dir, args, _specs, name)
            extra = (f"; rendered {len(rc_imgs)} image(s)"
                     if rc_imgs else "")
            print(f"      - {name}{extra}")
    else:
        print("      --variants not set; skipping variant generation.")

    print(f"\nDone. Run folder: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
