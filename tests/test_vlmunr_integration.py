"""Integration tests for the VLM-unreliability rendering + variant layer.

Covers:
  (a) Exact-string filename builder tests.
  (b) Removal-fraction + layout/asset join logic on a synthetic in-memory
      task + layout (deterministic, kept counts, >= 1 guarantee).
  (c) A bpy smoke test rendering one config on a primitive cube via vlmunr_bpa
      (asserts a non-empty PNG is written), plus a pure-function test of the
      transform math (location / rotation / scale, including the -90 deg
      pre-rotation and z = bbox.z / 2 for floor objects).
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vlmunr_config as cfg  # noqa: E402
import vlmunr_render as render  # noqa: E402
import vlmunr_variants as variants  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def make_synthetic_task(n: int = 8):
    assets = {}
    for i in range(n):
        assets[f"uid{i:02d}-0"] = {
            "path": f"/nonexistent/uid{i:02d}.glb",
            "category": f"cat{i}",
            "description": f"a synthetic object number {i}",
            "onFloor": True,
            "assetMetadata": {"boundingBox": {"x": 1.0, "y": 1.0, "z": 0.4}},
        }
    task = {
        "task_description": "synthetic",
        "boundary": {
            "floor_vertices": [
                [0, 0, 0],
                [4, 0, 0],
                [4, 5, 0],
                [0, 5, 0],
            ],
            "wall_height": 2.5,
        },
        "assets": assets,
    }
    return task


def make_synthetic_layout(n: int = 8):
    layout = {}
    for i in range(n):
        layout[f"uid{i:02d}-0"] = {
            "position": [float(i), float(i), 0.0],
            "rotation": [0.0, 0.0, float(i * 10)],
        }
    return layout


# ===========================================================================
# (a) Filename builders -- exact strings
# ===========================================================================


def test_master_filename_exact():
    assert (
        render.master_filename(512, 50, 0, 0, "city")
        == "render_512_50_0_0_city.png"
    )
    assert (
        render.master_filename(1024, 200, 60, 330, "sunset")
        == "render_1024_200_60_330_sunset.png"
    )


def test_composite_filename_exact():
    assert (
        render.composite_filename(512, 50, (128, 128, 128), 0, 0, "city")
        == "render_512_50_128_128_128_0_0_city.png"
    )
    assert (
        render.composite_filename(224, 24, (0, 0, 0), 90, 180, "studio")
        == "render_224_24_0_0_0_90_180_studio.png"
    )


def test_phase_levels_shapes():
    assert cfg.phase_levels("1a")["res"] == cfg.RESOLUTIONS
    assert cfg.phase_levels("1b")["bg"] == cfg.BACKGROUND_GRAYS
    assert cfg.phase_levels("1c")["hdri"] == cfg.HDRIS
    assert cfg.phase_levels("1d")["focal"] == cfg.FOCAL_LENGTHS
    p2 = cfg.phase_levels("2")
    assert p2["pitch"] == cfg.PITCHES and p2["yaw"] == cfg.YAWS
    with pytest.raises(ValueError):
        cfg.phase_levels("bogus")


def test_enumerate_renders_counts():
    # phase 1a: one master per resolution, each with one (baseline) bg.
    specs = render.enumerate_renders("1a")
    assert len(specs) == len(cfg.RESOLUTIONS)
    assert all(len(s["bgs"]) == 1 for s in specs)
    # phase 1b: single master, all bg grays composited onto it.
    specs = render.enumerate_renders("1b")
    assert len(specs) == 1
    assert len(specs[0]["bgs"]) == len(cfg.BACKGROUND_GRAYS)
    # phase 2: pitch x yaw masters.
    specs = render.enumerate_renders("2")
    assert len(specs) == len(cfg.PITCHES) * len(cfg.YAWS)


# ===========================================================================
# (b) Removal fraction + join logic
# ===========================================================================


@pytest.mark.parametrize(
    "n,k,expected",
    [
        (8, 2, 4),
        (8, 4, 2),
        (8, 8, 1),
        (10, 2, 5),
        (10, 4, 2),  # round(2.5) -> 2
        (1, 8, 1),  # clamp to >= 1
        (3, 8, 1),  # round(0.375) -> 0 -> clamp 1
        (0, 2, 0),  # empty scene
    ],
)
def test_kept_count(n, k, expected):
    assert variants.kept_count(n, k) == expected


def test_select_kept_ids_counts_and_determinism():
    layout = make_synthetic_layout(8)
    ids = list(layout.keys())
    for k, exp in [(2, 4), (4, 2), (8, 1)]:
        kept = variants.select_kept_ids(ids, k, seed=42)
        assert len(kept) == exp
        # subset of original, original order preserved
        assert set(kept).issubset(set(ids))
        assert kept == [i for i in ids if i in set(kept)]
        # deterministic
        assert kept == variants.select_kept_ids(ids, k, seed=42)
    # different seed -> (very likely) different selection for half
    a = variants.select_kept_ids(ids, 2, seed=1)
    b = variants.select_kept_ids(ids, 2, seed=2)
    assert a != b


def test_select_kept_ids_order_independent():
    ids = make_synthetic_layout(8)
    keys = list(ids.keys())
    fwd = variants.select_kept_ids(keys, 2, seed=7)
    rev = variants.select_kept_ids(list(reversed(keys)), 2, seed=7)
    assert set(fwd) == set(rev)


def test_make_removal_layout():
    layout = make_synthetic_layout(8)
    half = variants.make_removal_layout(layout, 2, seed=42)
    assert len(half) == 4
    assert all(half[i] == layout[i] for i in half)


def test_join_layout_to_assets():
    task = make_synthetic_task(8)
    layout = make_synthetic_layout(8)
    # add a layout entry with no matching asset -> must be skipped
    layout["orphan-0"] = {"position": [0, 0, 0], "rotation": [0, 0, 0]}
    records = render.join_layout_to_assets(task, layout)
    assert len(records) == 8  # orphan dropped
    rec = records[0]
    assert rec["onFloor"] is True
    assert rec["bbox"]["z"] == 0.4
    assert rec["rotation_z_deg"] == 0.0
    assert records[1]["rotation_z_deg"] == 10.0


# ===========================================================================
# (c) Transform math + bpy smoke
# ===========================================================================


def test_transform_floor_object_z_and_prerotation():
    rec = {
        "position": [2.0, 3.0, 0.0],
        "rotation_z_deg": 45.0,
        "bbox": {"x": 1.0, "y": 1.0, "z": 0.4},
        "onFloor": True,
    }
    tf = render.compute_object_transform(rec)
    # z overridden to bbox.z / 2
    assert tf["position"] == (2.0, 3.0, 0.2)
    # rotation = -90 prerotation + 45 layout
    assert tf["rotation"] == (0.0, 0.0, -45.0)
    assert tf["scale"] == (1.0, 1.0, 1.0)


def test_transform_nonfloor_keeps_z():
    rec = {
        "position": [1.0, 1.0, 1.5],
        "rotation_z_deg": 0.0,
        "bbox": {"x": 1.0, "y": 1.0, "z": 0.4},
        "onFloor": False,
    }
    tf = render.compute_object_transform(rec)
    assert tf["position"] == (1.0, 1.0, 1.5)
    assert tf["rotation"] == (0.0, 0.0, -90.0)


def test_scene_center_radius():
    task = make_synthetic_task(2)
    center, radius = render.compute_scene_center_radius(task)
    assert center == pytest.approx((2.0, 2.5, 1.25))
    assert radius > 0


# --- bpy smoke ------------------------------------------------------------

bpy = pytest.importorskip("bpy")


def test_bpy_smoke_render_cube(tmp_path):
    import io

    import vlmunr_bpa as bpa
    from conftest import real_terminal_fds

    hdri = os.path.join(ROOT, "vlmunr_hdri", "city.exr")
    assert os.path.exists(hdri), "city.exr must be vendored"

    # bpa.redirect_stdout() does dup2 + ``sys.stdout.close()`` at the OS fd
    # level, which clobbers both pytest's capture tempfiles (fds 1/2) and the
    # ``sys.stdout`` object pytest's terminal writer caches.  To run under a
    # plain ``pytest tests/`` we (1) point fds 1/2 at the genuine terminal for
    # the duration of the bpy work, and (2) swap in throwaway ``sys.stdout``/
    # ``sys.stderr`` objects for bpa to close.  Everything is restored after.
    cap_out_fd = os.dup(1)
    cap_err_fd = os.dup(2)
    real_out_fd, real_err_fd = real_terminal_fds()
    os.dup2(real_out_fd, 1)
    os.dup2(real_err_fd, 2)
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = io.TextIOWrapper(os.fdopen(os.dup(1), "wb"), write_through=True)
    sys.stderr = io.TextIOWrapper(os.fdopen(os.dup(2), "wb"), write_through=True)
    try:
        bpa.clear()
        bpa.initialize(transparent=True, environment_map=(hdri, 1.0))
        cube = bpa.Builder.new_cube("smoke")
        bpa.transform(
            cube, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)
        )

        out = str(tmp_path / "smoke.png")
        renderer = bpa.Renderer()
        from mathutils import Vector

        ok = renderer.render_perspective(
            out,
            Vector((0.0, 0.0, 0.0)),
            2.0,
            rotation=(0, 0, 0),
            resolution=64,
            focal_length=50,
            background=None,
        )
        assert ok
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0

        # composite a gray background and confirm a non-empty RGB png results
        comp = str(tmp_path / "smoke_bg.png")
        bpa.Renderer.add_bg_to_rgba(out, comp, color=(128, 128, 128))
        assert os.path.getsize(comp) > 0
    finally:
        try:
            sys.stdout.close()
        except Exception:
            pass
        try:
            sys.stderr.close()
        except Exception:
            pass
        # Restore pytest's captured fds and stream objects.
        os.dup2(cap_out_fd, 1)
        os.dup2(cap_err_fd, 2)
        for fd in (cap_out_fd, cap_err_fd, real_out_fd, real_err_fd):
            try:
                os.close(fd)
            except Exception:
                pass
        sys.stdout, sys.stderr = saved_out, saved_err
