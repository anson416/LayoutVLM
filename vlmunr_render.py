"""Render LayoutVLM scenes for the VLM-unreliability audit harness.

This module joins a LayoutVLM ``layout.json`` (instance id -> position/rotation)
back to its input task JSON (asset paths, bounding boxes, boundary), builds the
scene in Blender honoring the LayoutVLM coordinate convention, then renders a
set of *render specs* (each = one transparent master + its bg composites),
writing PNGs into ``<scene-dir>/renderings/``.

Rendering strategy (per the ``bpa.py`` convention the harness requires):

1. For each ``(res, focal, pitch, yaw, env)`` master, render ONCE with a
   **transparent** background (env-map lighting still applied)::

       render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png

2. Composite each requested background color directly onto the transparent
   master to produce::

       render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png

3. ``fit_ratio=1`` (tight-fit) -- the object's projection fills the viewport
   with no unnecessary empty space.

``pitch`` follows the bpa convention where 0 == top-down; the filename always
carries the literal pitch value used (so filenames are consistent across
methods even if a peer method calls top-down "90").

The scene-loading and filename logic is factored into pure functions so it can
be unit-tested without ``bpy`` installed.  ``bpy``/``vlmunr_bpa`` are imported
lazily inside the rendering entry points only.

LayoutVLM coordinate convention (z = world UP, meters, rotation in DEGREES
about z):

  * Each mesh is recentered (origin -> BOUNDS).
  * A -90 degree pre-rotation about Z is applied (preprocessed meshes face -y;
    this makes them face +x).
  * The layout ``rotation`` (degrees about Z) is then applied.
  * The object is located at ``position``.
  * For floor objects the z coordinate of the position is ``boundingBox.z / 2``.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import vlmunr_config as cfg

# -90 degree pre-rotation about Z applied to every preprocessed mesh.
PREROTATION_Z_DEG: float = -90.0

# Tight-fit: the object projection fills the viewport (per the bpa convention
# the harness mandates).  1.0 == tight-fit, 0.0 == bounding-sphere.
DEFAULT_FIT_RATIO: float = 1.0


# ===========================================================================
# Pure helpers (no bpy) - unit tested
# ===========================================================================


def master_filename(
    res: int, focal: int, pitch: int, yaw: int, env: str
) -> str:
    """Filename for a transparent master render."""

    return f"render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{env}.png"


def composite_filename(
    res: int,
    focal: int,
    bg: Tuple[int, int, int],
    pitch: int,
    yaw: int,
    env: str,
) -> str:
    """Filename for a background-composited render."""

    r, g, b = bg
    return (
        f"render_res-{res}_focal-{focal}_pitch-{pitch}_yaw-{yaw}_env-{env}"
        f"_bg-{r}-{g}-{b}.png"
    )


def resolve_scene_paths(
    task_json: Optional[str],
    layout_json: Optional[str],
    scene_dir: Optional[str],
) -> Tuple[str, str, str]:
    """Resolve ``(task_json, layout_json, scene_dir)`` from CLI inputs.

    Either ``scene_dir`` (containing both files) or both explicit paths must be
    provided.  Returns absolute-ish paths exactly as discovered.
    """

    if scene_dir is not None:
        t = task_json
        if t is None:
            t = _find_task_json(scene_dir)
        layout = layout_json or os.path.join(scene_dir, "layout.json")
        return t, layout, scene_dir

    if task_json is None or layout_json is None:
        raise ValueError(
            "Provide either --scene-dir or both --task-json and --layout-json"
        )
    return task_json, layout_json, os.path.dirname(os.path.abspath(layout_json))


def _find_task_json(scene_dir: str) -> str:
    """Find the input task JSON inside a scene dir (anything but layout.json)."""

    candidates = [
        f
        for f in sorted(os.listdir(scene_dir))
        if f.endswith(".json")
        and f != "layout.json"
        and not f.startswith("variant_")
    ]
    if not candidates:
        raise FileNotFoundError(f"No task JSON found in {scene_dir}")
    # Prefer a file literally named task.json if present.
    for c in candidates:
        if c == "task.json":
            return os.path.join(scene_dir, c)
    return os.path.join(scene_dir, candidates[0])


def join_layout_to_assets(
    task: Dict, layout: Dict, asset_dir: str = None
) -> List[Dict]:
    """Join ``layout.json`` placements back to task asset metadata.

    Raw benchmark tasks carry empty asset entries (all metadata is filled at
    generation time by prepare_task_assets and not saved). When ``asset_dir`` is
    given, resolve each instance's glb path and bounding box from
    ``<asset_dir>/<uid>/{<uid>.glb,data.json}`` (uid = inst_id without the
    trailing ``-<idx>``). Instances with no resolvable path are skipped.
    """

    import json as _json

    assets = task.get("assets", {})
    records: List[Dict] = []
    for inst_id, place in layout.items():
        asset = assets.get(inst_id)
        if asset is None:
            asset = {}
        path = asset.get("path")
        bbox = (
            asset.get("assetMetadata", {}).get("boundingBox")
            or asset.get("boundingBox")
            or {}
        )
        onFloor = bool(asset.get("onFloor", False))
        category = asset.get("category", "")
        description = asset.get("description", "")
        # Resolve from the asset_dir when the task entry is empty.
        if (not path or not bbox) and asset_dir:
            uid = inst_id.rsplit("-", 1)[0]
            adir = os.path.join(asset_dir, uid)
            cand = os.path.join(adir, f"{uid}.glb")
            if os.path.exists(cand):
                path = path or cand
            dj = os.path.join(adir, "data.json")
            if os.path.exists(dj):
                try:
                    d = _json.load(open(dj))
                    ann = d.get("annotations", {})
                    bbox = bbox or d.get("assetMetadata", {}).get("boundingBox", {})
                    onFloor = bool(ann.get("onFloor", onFloor))
                    category = category or ann.get("category", "")
                    description = description or ann.get("description", "")
                except Exception:
                    pass
        # This is a *data* join: drop an instance only when no path can be
        # determined at all. We deliberately do NOT require the path to exist
        # on disk here -- file existence is the renderer's concern, not the
        # join's, and enforcing it broke the pure-join unit test (synthetic
        # nonexistent paths were silently dropped). ``asset_dir`` resolution
        # above already validated any path it supplied.
        if not path:
            continue
        bbox = bbox or {}
        position = list(place.get("position", [0, 0, 0]))
        rotation = list(place.get("rotation", [0, 0, 0]))
        records.append(
            {
                "id": inst_id,
                "path": path,
                "position": position,
                "rotation_z_deg": float(rotation[-1]) if rotation else 0.0,
                "bbox": bbox,
                "onFloor": onFloor,
                "category": category,
                "description": description,
            }
        )
    return records


def compute_object_transform(record: Dict) -> Dict:
    """Compute the Blender transform to apply for one placement record.

    Pure function capturing the LayoutVLM convention:

      * ``rotation`` about Z = pre-rotation (-90) + layout rotation (deg).
      * ``position`` = layout position, but with z overridden to
        ``boundingBox.z / 2`` for floor objects (when z is not already set).
      * ``scale`` = identity (assets are authored in meters).

    Returns ``{"position": (x,y,z), "rotation": (0,0,rz_deg), "scale": (1,1,1)}``.
    """

    pos = list(record.get("position", [0, 0, 0]))
    if len(pos) == 2:
        pos = [pos[0], pos[1], 0.0]
    x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

    bbox = record.get("bbox") or {}
    if record.get("onFloor", False) and "z" in bbox:
        z = float(bbox["z"]) / 2.0

    rz = PREROTATION_Z_DEG + float(record.get("rotation_z_deg", 0.0))

    return {
        "position": (x, y, z),
        "rotation": (0.0, 0.0, rz),
        "scale": (1.0, 1.0, 1.0),
    }


def compute_scene_center_radius(
    task: Dict,
) -> Tuple[Tuple[float, float, float], float]:
    """Estimate a camera target/radius from the floor boundary.

    Center is the centroid of the floor vertices lifted to half the wall
    height; radius is the distance from that center to the farthest floor
    corner / ceiling.  Used as a fallback framing target; the renderer also
    relies on bpa's bounding-sphere fit.
    """

    boundary = task.get("boundary", {})
    verts = boundary.get("floor_vertices", [])
    wall_h = float(boundary.get("wall_height", 2.5))
    if not verts:
        return (0.0, 0.0, wall_h / 2.0), max(wall_h, 1.0)

    xs = [float(v[0]) for v in verts]
    ys = [float(v[1]) for v in verts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cz = wall_h / 2.0
    radius = 0.0
    for vx, vy in zip(xs, ys):
        d = ((vx - cx) ** 2 + (vy - cy) ** 2 + cz**2) ** 0.5
        radius = max(radius, d)
    radius = max(radius, wall_h)
    return (cx, cy, cz), radius


def _as_rgb(bg) -> Tuple[int, int, int]:
    """Normalize a background level to an ``(r, g, b)`` tuple.

    Gray phases store a single int per level; chromatic phases store full
    ``(r, g, b)`` tuples.  Accept either shape.
    """

    if isinstance(bg, (tuple, list)):
        r, g, b = bg
        return (int(r), int(g), int(b))
    return (int(bg), int(bg), int(bg))


def merge_specs(specs: Iterable[Dict]) -> List[Dict]:
    """Merge specs that share a master, unioning their backgrounds.

    A master is identified by ``(res, focal, pitch, yaw, env)``.  When several
    specs share a master (e.g. the baseline point 512/50/0/0/city appears in
    the resolution, focal, pitch, env and background sweeps), the renderer must
    render that transparent master ONCE and composite the UNION of all its
    requested backgrounds onto it.  This returns one merged spec per unique
    master, preserving first-seen order and de-duplicating backgrounds.
    """

    merged: Dict[Tuple, Dict] = {}
    order: List[Tuple] = []
    for s in specs:
        env = s.get("env", s.get("hdri"))
        key = (s["res"], s["focal"], s["pitch"], s["yaw"], env)
        if key not in merged:
            merged[key] = {
                "res": s["res"],
                "focal": s["focal"],
                "pitch": s["pitch"],
                "yaw": s["yaw"],
                "env": env,
                "hdri": env,
                "bgs": [],
            }
            order.append(key)
        for bg in s.get("bgs", []):
            rgb = _as_rgb(bg)
            if rgb not in merged[key]["bgs"]:
                merged[key]["bgs"].append(rgb)
    return [merged[k] for k in order]


def enumerate_renders(phase: str) -> List[Dict]:
    """Expand a legacy phase into a list of render specs (pure, no bpy).

    Each spec is a dict with ``res, focal, pitch, yaw, env`` (=hdri) and a list
    of background ``(r, g, b)`` tuples ``bgs``.  One spec == one master render
    plus its composites.  Kept as a building block for the module CLI and for
    single-axis tests; ``cli.py`` uses ``cfg.single_render_spec`` /
    ``cfg.all_sweep_specs`` instead.
    """

    levels = cfg.phase_levels(phase)
    specs: List[Dict] = []
    for hdri in levels["hdri"]:
        for res in levels["res"]:
            for focal in levels["focal"]:
                for pitch in levels["pitch"]:
                    for yaw in levels["yaw"]:
                        bgs = [_as_rgb(g) for g in levels["bg"]]
                        specs.append(
                            {
                                "res": res,
                                "focal": focal,
                                "pitch": pitch,
                                "yaw": yaw,
                                "env": hdri,
                                "hdri": hdri,
                                "bgs": bgs,
                            }
                        )
    return specs


# ===========================================================================
# bpy-dependent scene loading + rendering
# ===========================================================================


def load_scene_into_blender(task: Dict, layout: Dict, asset_dir: str = None):
    """Import every joined placement into the current Blender scene.

    Requires ``bpy`` (imported lazily via ``vlmunr_bpa``).  Clears the scene
    first, then imports each asset glb and applies its LayoutVLM transform.
    Returns the list of joined records that were actually imported.
    """

    import vlmunr_bpa as bpa

    bpa.clear()
    records = join_layout_to_assets(task, layout, asset_dir)
    imported: List[Dict] = []
    for rec in records:
        path = rec.get("path")
        if not path or not os.path.exists(path):
            # Degrade gracefully: skip missing assets so a partial scene
            # still renders.
            continue
        obj = bpa.import_obj(path)
        # VLMUNR_PATCH fit-to-bbox scale
        # Rescale mesh to its target boundingBox BEFORE the placement transform.
        # Assets are not uniformly authored in meters; fit measured dims to the
        # target bbox so the room stays at its intended 4x5 m scale.
        try:
            import bpy as _bpy  # noqa
            _bb = rec.get("bbox") or {}
            _d = obj.dimensions
            # prepare_task_assets swapped x<->y in bbox; obj.dimensions is (x,y,z)
            _tx = float(_bb.get("y", _d.x)); _ty = float(_bb.get("x", _d.y)); _tz = float(_bb.get("z", _d.z))
            _sx = _tx / _d.x if _d.x else 1.0
            _sy = _ty / _d.y if _d.y else 1.0
            _sz = _tz / _d.z if _d.z else 1.0
        except Exception:
            _sx = _sy = _sz = 1.0
        # Apply the fit-to-bbox scale first.
        obj.scale = (_sx * obj.scale.x, _sy * obj.scale.y, _sz * obj.scale.z)
        import bpy as _b2
        _b2.context.view_layer.update()
        # HARD CLAMP: no imported asset may exceed 3.5 m on any axis (guards
        # against unreliable bbox metadata). Uniformly shrink if it does.
        _dd = obj.dimensions
        _mx = max(_dd.x, _dd.y, _dd.z)
        if _mx > 3.5:
            _f = 3.5 / _mx
            obj.scale = (obj.scale.x*_f, obj.scale.y*_f, obj.scale.z*_f)
            _b2.context.view_layer.update()
        tf = compute_object_transform(rec)
        bpa.transform(
            obj,
            position=tf["position"],
            rotation=tf["rotation"],
            scale=None,
        )
        imported.append(rec)
    return imported


def _hdri_path(env: str, hdri_dir: Optional[str] = None) -> str:
    """Resolve an env-map name to its .exr path.

    ``hdri_dir`` overrides the vendored default; the CLI passes its
    ``--hdri_dir`` through.  The vendored HDRIs ship under ``vlmunr_hdri/``
    next to this module.
    """

    base = hdri_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlmunr_hdri")
    return os.path.join(base, f"{env}.exr")


def _build_lvlm_shell(task):
    """Build floor+walls from task boundary.floor_vertices + wall_height.
    Z-up meters, polygon at z=0. Returns wall objects (tagged) for culling."""
    try:
        import bpy  # noqa
        import vlmunr_shell as _vs
    except Exception:
        return []
    b = (task or {}).get("boundary", {}) or {}
    fv = b.get("floor_vertices") or []
    if len(fv) < 3:
        return []
    verts = [(float(v[0]), float(v[1])) for v in fv]
    wh = float(b.get("wall_height", 2.5) or 2.5)
    if wh < 1.2:
        wh = 2.5  # some tasks store an implausibly short wall height
    try:
        return _vs.build_room_shell(bpy, verts, wh, margin=0.0, ceiling=False)
    except Exception as _e:
        print("VLMUNR lvlm shell build failed:", _e)
        return []


def render_specs(
    task: Dict,
    layout: Dict,
    scene_dir: str,
    specs: List[Dict],
    *,
    env_strength: float = 1.0,
    asset_dir: str = None,
    hdri_dir: Optional[str] = None,
    fit_ratio: float = DEFAULT_FIT_RATIO,
) -> List[str]:
    """Render every (merged) spec, returning the list of PNG paths written.

    Each spec is ``{res, focal, pitch, yaw, env, bgs}`` (``bgs`` = list of
    ``(r, g, b)`` backgrounds to composite onto that master).  Specs sharing a
    master are merged so the transparent master is rendered once and the union
    of its backgrounds composited onto it.

    Pipeline per unique env: load geometry, build the room shell, then set the
    env-map world (geometry is loaded FIRST because ``bpa.clear`` wipes the
    world).  For each master: dollhouse-cull the walls for this camera pose,
    render the transparent master at ``fit_ratio``, then composite each bg.
    """

    import vlmunr_bpa as bpa
    from mathutils import Vector

    out_dir = os.path.join(scene_dir, "renderings")
    os.makedirs(out_dir, exist_ok=True)

    merged = merge_specs(specs)
    if not merged:
        return []

    center, radius = compute_scene_center_radius(task)
    center = Vector(center)

    written: List[str] = []
    # Group by env so the scene is reloaded only when the env map changes.
    for env in sorted({s["env"] for s in merged}):
        env_path = _hdri_path(env, hdri_dir)
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"HDRI not found: {env_path}")
        # Load geometry FIRST: load_scene_into_blender calls bpa.clear() which
        # wipes bpy.data.worlds, so the env world must be set AFTER it or the
        # scene renders unlit/black.
        load_scene_into_blender(task, layout, asset_dir)
        walls = _build_lvlm_shell(task)
        bpa.initialize(transparent=True, environment_map=(env_path, env_strength))

        for spec in merged:
            if spec["env"] != env:
                continue
            try:
                import vlmunr_shell as _vs
                _vs.cull_walls(walls, spec["pitch"], spec["yaw"])
            except Exception:
                pass
            master = os.path.join(
                out_dir,
                master_filename(
                    spec["res"],
                    spec["focal"],
                    spec["pitch"],
                    spec["yaw"],
                    spec["env"],
                ),
            )
            renderer = bpa.Renderer()
            renderer.render_perspective(
                master,
                center,
                radius,
                rotation=(spec["pitch"], 0, spec["yaw"]),
                resolution=spec["res"],
                focal_length=spec["focal"],
                fit_ratio=fit_ratio,
                background=None,
            )
            written.append(master)
            for bg in spec["bgs"]:
                comp = os.path.join(
                    out_dir,
                    composite_filename(
                        spec["res"],
                        spec["focal"],
                        bg,
                        spec["pitch"],
                        spec["yaw"],
                        spec["env"],
                    ),
                )
                bpa.Renderer.add_bg_to_rgba(master, comp, color=bg)
                written.append(comp)

    return written


def render_phase(
    task: Dict,
    layout: Dict,
    scene_dir: str,
    phase: str,
    *,
    env_strength: float = 1.0,
    asset_dir: str = None,
    hdri_dir: Optional[str] = None,
    fit_ratio: float = DEFAULT_FIT_RATIO,
) -> List[str]:
    """Render every spec in a legacy *phase* (thin wrapper over render_specs)."""

    return render_specs(
        task, layout, scene_dir, enumerate_renders(phase),
        env_strength=env_strength, asset_dir=asset_dir,
        hdri_dir=hdri_dir, fit_ratio=fit_ratio,
    )


# ===========================================================================
# CLI
# ===========================================================================


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-json", help="Path to the input task JSON")
    p.add_argument("--layout-json", help="Path to layout.json")
    p.add_argument(
        "--scene-dir", help="Directory containing both task JSON and layout.json"
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--render",
        action="store_true",
        help="Render the single baseline image (512/white/50mm/pitch0/yaw0/city): "
             "transparent master + white composite.",
    )
    mode.add_argument(
        "--render-all",
        action="store_true",
        help="Render the six single-axis sweeps (resolution/focal/pitch/yaw/env/bg).",
    )
    p.add_argument(
        "--phase",
        choices=cfg.PHASES + ["all"],
        default=None,
        help="Legacy single-phase render (building block). Mutually exclusive "
             "with --render/--render-all.",
    )
    p.add_argument("--env-strength", type=float, default=1.0)
    p.add_argument("--fit-ratio", type=float, default=DEFAULT_FIT_RATIO)
    p.add_argument(
        "--hdri-dir",
        default=None,
        help="Directory of .exr env maps (default: vendored vlmunr_hdri/).",
    )
    p.add_argument(
        "--asset-dir",
        default="objaverse_processed",
        help="Asset dir to resolve <uid>/<uid>.glb + data.json when the task "
        "JSON has empty asset entries (raw benchmark tasks).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    task_path, layout_path, scene_dir = resolve_scene_paths(
        args.task_json, args.layout_json, args.scene_dir
    )
    with open(task_path) as f:
        task = json.load(f)
    with open(layout_path) as f:
        layout = json.load(f)

    asset_dir = args.asset_dir
    if asset_dir and not os.path.isabs(asset_dir):
        # resolve relative to CWD, then to the repo dir
        if not os.path.isdir(asset_dir):
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.join(here, asset_dir)
            if os.path.isdir(cand):
                asset_dir = cand

    if args.render:
        specs = [cfg.single_render_spec()]
    elif args.render_all:
        specs = cfg.all_sweep_specs()
    elif args.phase:
        if args.phase == "all":
            specs = [s for ph in cfg.PHASES for s in enumerate_renders(ph)]
        else:
            specs = enumerate_renders(args.phase)
    else:
        # Default to the single baseline render if nothing is specified.
        specs = [cfg.single_render_spec()]

    written = render_specs(
        task, layout, scene_dir, specs,
        env_strength=args.env_strength, asset_dir=asset_dir,
        hdri_dir=args.hdri_dir, fit_ratio=args.fit_ratio,
    )
    print(f"Wrote {len(written)} images to {scene_dir}/renderings")


if __name__ == "__main__":
    main()
