"""Render LayoutVLM scenes for the VLM-unreliability audit harness.

This module joins a LayoutVLM ``layout.json`` (instance id -> position/rotation)
back to its input task JSON (asset paths, bounding boxes, boundary), builds the
scene in Blender honoring the LayoutVLM coordinate convention, and sweeps the
requested audit phase, writing PNGs into ``<scene-dir>/renderings/``.

Rendering uses a two-phase strategy:

1. A transparent *master* render is produced once per
   ``(res, focal, pitch, yaw, hdri)`` combination::

       render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png

2. Each requested background gray is composited onto the master to produce::

       render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png

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
from typing import Dict, List, Optional, Tuple

import vlmunr_config as cfg

# -90 degree pre-rotation about Z applied to every preprocessed mesh.
PREROTATION_Z_DEG: float = -90.0


# ===========================================================================
# Pure helpers (no bpy) - unit tested
# ===========================================================================


def master_filename(
    res: int, focal: int, pitch: int, yaw: int, hdri: str
) -> str:
    """Filename for a transparent master render."""

    return f"render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png"


def composite_filename(
    res: int,
    focal: int,
    bg: Tuple[int, int, int],
    pitch: int,
    yaw: int,
    hdri: str,
) -> str:
    """Filename for a background-composited render."""

    r, g, b = bg
    return f"render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png"


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
        if not path or not os.path.exists(path):
            continue
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


def enumerate_renders(phase: str) -> List[Dict]:
    """Expand a phase into a list of render specs (pure, no bpy).

    Each spec is a dict with ``res, focal, pitch, yaw, hdri`` and a list of
    background ``(r, g, b)`` tuples ``bgs`` (gray levels are expanded to equal
    channels; chromatic phases pass their tuples through).  One spec == one
    master render plus its composites.
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
        tf = compute_object_transform(rec)
        bpa.transform(
            obj,
            position=tf["position"],
            rotation=tf["rotation"],
            scale=tf["scale"],
        )
        imported.append(rec)
    return imported


def _hdri_path(hdri: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "vlmunr_hdri", f"{hdri}.exr")


def render_phase(
    task: Dict,
    layout: Dict,
    scene_dir: str,
    phase: str,
    *,
    env_strength: float = 1.0,
    asset_dir: str = None,
) -> List[str]:
    """Render every spec in *phase*, returning the list of PNG paths written."""

    import vlmunr_bpa as bpa
    from mathutils import Vector

    out_dir = os.path.join(scene_dir, "renderings")
    os.makedirs(out_dir, exist_ok=True)

    center, radius = compute_scene_center_radius(task)
    center = Vector(center)
    specs = enumerate_renders(phase)

    written: List[str] = []
    current_hdri: Optional[str] = None
    scene_loaded = False

    # Group by hdri so we only re-initialize (and reload the scene) on change.
    for hdri in sorted({s["hdri"] for s in specs}):
        env = _hdri_path(hdri)
        if not os.path.exists(env):
            raise FileNotFoundError(f"HDRI not found: {env}")
        bpa.initialize(transparent=True, environment_map=(env, env_strength))
        load_scene_into_blender(task, layout, asset_dir)
        current_hdri = hdri
        scene_loaded = True

        for spec in specs:
            if spec["hdri"] != current_hdri:
                continue
            master = os.path.join(
                out_dir,
                master_filename(
                    spec["res"],
                    spec["focal"],
                    spec["pitch"],
                    spec["yaw"],
                    spec["hdri"],
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
                        spec["hdri"],
                    ),
                )
                bpa.Renderer.add_bg_to_rgba(master, comp, color=bg)
                written.append(comp)

    assert scene_loaded
    return written


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
    p.add_argument(
        "--phase",
        choices=cfg.PHASES + ["all"],
        default="1a",
    )
    p.add_argument("--env-strength", type=float, default=1.0)
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

    phases = cfg.PHASES if args.phase == "all" else [args.phase]
    all_written: List[str] = []
    for ph in phases:
        all_written.extend(
            render_phase(
                task, layout, scene_dir, ph,
                env_strength=args.env_strength, asset_dir=asset_dir,
            )
        )
    print(f"Wrote {len(all_written)} images to {scene_dir}/renderings")


if __name__ == "__main__":
    main()
