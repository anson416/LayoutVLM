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

import json
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


# ===========================================================================
# Factor-level counts -> paper Table 1 parity
# ===========================================================================


def test_factor_level_counts():
    assert len(cfg.RESOLUTIONS) == 9
    assert len(cfg.FOCAL_LENGTHS) == 7
    assert len(cfg.PITCHES) == 7
    assert len(cfg.YAWS) == 8
    assert len(cfg.BACKGROUND_GRAYS) == 6
    assert len(cfg.BACKGROUND_CHROMATIC) == 3


def test_factor_level_values_exact():
    assert cfg.RESOLUTIONS == [196, 224, 256, 336, 384, 448, 512, 768, 1024]
    assert cfg.FOCAL_LENGTHS == [16, 24, 35, 50, 85, 100, 200]
    assert cfg.PITCHES == [0, 15, 30, 45, 60, 75, 90]
    assert cfg.YAWS == [0, 45, 90, 135, 180, 225, 270, 315]
    assert cfg.BACKGROUND_GRAYS == [0, 65, 128, 186, 204, 255]
    assert cfg.BACKGROUND_CHROMATIC == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    assert cfg.FLOOR_TEXTURE_BACKGROUND == "floor_texture"
    assert cfg.BASELINE_YAW_PITCH == 45


def test_phase_1b_chroma():
    p = cfg.phase_levels("1b_chroma")
    assert p["bg"] == cfg.BACKGROUND_CHROMATIC
    assert len(p["bg"]) == 3
    # all other axes at baseline
    assert p["res"] == [cfg.BASELINE_RES]
    specs = render.enumerate_renders("1b_chroma")
    assert len(specs) == 1
    assert specs[0]["bgs"] == [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def test_phase_2_pitch():
    p = cfg.phase_levels("2_pitch")
    assert p["pitch"] == cfg.PITCHES
    assert len(p["pitch"]) == 7
    assert p["yaw"] == [0]  # baseline yaw
    specs = render.enumerate_renders("2_pitch")
    assert len(specs) == 7
    assert all(s["yaw"] == 0 for s in specs)


def test_phase_2_yaw():
    p = cfg.phase_levels("2_yaw")
    assert p["yaw"] == cfg.YAWS
    assert len(p["yaw"]) == 8
    assert p["pitch"] == [45]  # fixed pitch
    specs = render.enumerate_renders("2_yaw")
    assert len(specs) == 8
    assert all(s["pitch"] == 45 for s in specs)


def test_phases_list_extended():
    assert cfg.PHASES == [
        "1a",
        "1b",
        "1b_chroma",
        "1c",
        "1d",
        "2",
        "2_pitch",
        "2_yaw",
    ]
    assert cfg.ALL_PHASES == cfg.PHASES


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
# (b2) Layout scramble + category-aware substitution
# ===========================================================================


def test_floor_bbox():
    verts = [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    assert variants.floor_bbox(verts) == (0.0, 0.0, 4.0, 5.0)
    assert variants.floor_bbox([]) == (0.0, 0.0, 0.0, 0.0)


def test_scramble_determinism_bounds_and_preserved_ids():
    layout = make_synthetic_layout(8)
    verts = [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    out = variants.scramble_layout(layout, verts, seed=123)
    # determinism
    assert out == variants.scramble_layout(layout, verts, seed=123)
    # preserved instance set + count
    assert set(out.keys()) == set(layout.keys())
    assert len(out) == len(layout)
    min_x, min_y, max_x, max_y = variants.floor_bbox(verts)
    for inst_id, place in out.items():
        x, y, z = place["position"]
        # in-bounds (x, y) within floor bbox
        assert min_x <= x <= max_x
        assert min_y <= y <= max_y
        # rotation + z unchanged
        assert place["rotation"] == layout[inst_id]["rotation"]
        assert z == layout[inst_id]["position"][2]


def test_scramble_different_seed_moves_points():
    layout = make_synthetic_layout(8)
    verts = [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    a = variants.scramble_layout(layout, verts, seed=1)
    b = variants.scramble_layout(layout, verts, seed=2)
    assert any(a[i]["position"][:2] != b[i]["position"][:2] for i in a)


def test_scramble_order_independent():
    layout = make_synthetic_layout(6)
    rev = {k: layout[k] for k in reversed(list(layout.keys()))}
    verts = [[0, 0, 0], [4, 0, 0], [4, 5, 0], [0, 5, 0]]
    fwd = variants.scramble_layout(layout, verts, seed=9)
    bwd = variants.scramble_layout(rev, verts, seed=9)
    for i in fwd:
        assert fwd[i]["position"] == bwd[i]["position"]


def test_subst_within_intent_recording_degraded(monkeypatch):
    monkeypatch.delenv("VLMUNR_ASSET_LIBRARY", raising=False)
    task = make_synthetic_task(4)
    layout = make_synthetic_layout(4)
    new_layout, intent, _ = variants.make_category_subst_layout(
        task, layout, "within"
    )
    # scene unchanged
    assert new_layout == layout
    assert intent["kind"] == "category_subst"
    assert intent["mode"] == "within"
    assert intent["degraded"] is True
    # intent recorded as {instance_id: mode} per joined instance
    assert len(intent["substitutions"]) == 4
    for rec in intent["substitutions"]:
        assert list(rec.values())[0] == "within"
        assert list(rec.keys())[0] in layout


def test_subst_cross_intent_recording_degraded(monkeypatch):
    monkeypatch.delenv("VLMUNR_ASSET_LIBRARY", raising=False)
    task = make_synthetic_task(3)
    layout = make_synthetic_layout(3)
    new_layout, intent, _ = variants.make_category_subst_layout(
        task, layout, "cross"
    )
    assert new_layout == layout
    assert intent["mode"] == "cross"
    assert intent["degraded"] is True
    assert len(intent["substitutions"]) == 3
    for rec in intent["substitutions"]:
        assert list(rec.values())[0] == "cross"


def test_subst_invalid_mode_raises():
    task = make_synthetic_task(2)
    layout = make_synthetic_layout(2)
    with pytest.raises(ValueError):
        variants.make_category_subst_layout(task, layout, "bogus")


# ===========================================================================
# (b3) Named variants for the text->scene CLI (biggest-only, worst-object)
# ===========================================================================


def test_biggest_only_keeps_single_largest():
    task = make_synthetic_task(4)
    # synthetic assets all have bbox 1x1x0.4 -> tie; make uid01 the biggest.
    task["assets"]["uid01-0"]["assetMetadata"]["boundingBox"] = {
        "x": 3.0, "y": 3.0, "z": 2.0
    }
    layout = make_synthetic_layout(4)
    out, kept = variants.make_biggest_only_layout(task, layout)
    assert len(out) == 1
    assert kept == "uid01-0"
    assert out["uid01-0"] == layout["uid01-0"]


def test_biggest_only_empty_layout():
    out, kept = variants.make_biggest_only_layout({"assets": {}}, {})
    assert out == {}
    assert kept is None


def test_worst_object_swaps_identity(tmp_path):
    task = make_synthetic_task(2)
    layout = make_synthetic_layout(2)
    # Library: candidates share NO tokens with the query descriptions
    # ("a synthetic object number N") so they are all worst matches; we just
    # assert identity actually changes and placements are preserved.
    lib = [
        {"category": "rock", "description": "a heavy boulder",
         "path": "/lib/rock.glb",
         "assetMetadata": {"boundingBox": {"x": 1, "y": 1, "z": 1}}},
        {"category": "stick", "description": "a thin twig",
         "path": "/lib/stick.glb"},
    ]
    lib_path = tmp_path / "lib.json"
    lib_path.write_text(json.dumps(lib))
    new_layout, new_task, intent = variants.make_worst_object_layout(
        task, layout, str(lib_path), rank_offset=0
    )
    # placements unchanged
    assert new_layout == layout
    # asset identity changed for every instance
    for inst_id in layout:
        orig = task["assets"][inst_id]
        new = new_task["assets"][inst_id]
        assert new["path"] != orig["path"]
        assert new["path"].startswith("/lib/")
    assert intent["kind"] == "worst_object"
    assert not intent["degraded"]
    assert len(intent["substitutions"]) == 2


def test_worst_object_missing_library_raises(tmp_path):
    task = make_synthetic_task(2)
    layout = make_synthetic_layout(2)
    with pytest.raises(FileNotFoundError):
        variants.make_worst_object_layout(task, layout, str(tmp_path / "nope.json"))


def test_generate_named_variants_writes_four(tmp_path):
    task = make_synthetic_task(6)
    layout = make_synthetic_layout(6)
    # build a library so the worst-object variant is produced (not skipped)
    lib = [{"category": "x", "description": "y", "path": "/lib/x.glb"}]
    lib_path = tmp_path / "lib.json"
    lib_path.write_text(json.dumps(lib))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    written = variants.generate_named_variants(
        str(run_dir), task, layout, seed=7, asset_library_path=str(lib_path)
    )
    names = [os.path.basename(w) for w in written]
    assert names == [
        "variant_01_half",
        "variant_02_biggest-only",
        "variant_03_scrambled",
        "variant_04_worst-object",
    ]
    for w in written:
        assert os.path.exists(os.path.join(w, "layout.json"))
        assert os.path.exists(os.path.join(w, "task.json"))
    # half keeps round(6/2)=3
    half = json.load(open(os.path.join(run_dir, names[0], "layout.json")))
    assert len(half) == 3
    # biggest-only keeps exactly 1
    big = json.load(open(os.path.join(run_dir, names[1], "layout.json")))
    assert len(big) == 1


def test_generate_named_variants_skips_worst_without_library(tmp_path):
    task = make_synthetic_task(4)
    layout = make_synthetic_layout(4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    written = variants.generate_named_variants(
        str(run_dir), task, layout, seed=7, asset_library_path=None
    )
    worst = run_dir / "variant_04_worst-object"
    # no layout.json (skipped), but an intent explains why
    assert not (worst / "layout.json").exists()
    intent = json.load(open(worst / "variant_intent.json"))
    assert intent["degraded"] is True



def test_subst_with_library_within_and_cross(tmp_path, monkeypatch):
    # Build a tiny library so the scoring/category-filter path executes.
    lib = [
        {"category": "cat0", "description": "another cat0 thing", "path": "/a.glb"},
        {"category": "cat1", "description": "a cat1 thing", "path": "/b.glb"},
        {"category": "catX", "description": "totally different", "path": "/c.glb"},
    ]
    lib_path = tmp_path / "lib.json"
    lib_path.write_text(json.dumps(lib))
    monkeypatch.setenv("VLMUNR_ASSET_LIBRARY", str(lib_path))

    task = make_synthetic_task(2)  # instances are cat0, cat1
    layout = make_synthetic_layout(2)

    _, within, _ = variants.make_category_subst_layout(task, layout, "within")
    assert within["degraded"] is False
    # uid00-0 is cat0: a within-category candidate (cat0) exists.
    within_for_0 = [
        s for s in within["substitutions"]
        if s.get("instance_id") == "uid00-0"
    ]
    assert within_for_0 and within_for_0[0]["mode"] == "within"

    _, cross, _ = variants.make_category_subst_layout(task, layout, "cross")
    assert cross["degraded"] is False
    cross_for_0 = [
        s for s in cross["substitutions"]
        if s.get("instance_id") == "uid00-0"
    ]
    # a cross-category candidate (not cat0) must have been chosen
    assert cross_for_0 and cross_for_0[0]["mode"] == "cross"


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
# NOTE: bpy is imported per-test (via pytest.importorskip inside the test body)
# rather than at module scope. A module-scope importorskip would skip the
# *entire* file when bpy is absent, hiding all the pure-function tests above
# (and the CLI/serialization tests below) on environments without Blender.

bpy = None  # populated lazily inside test_bpy_smoke_render_cube


def test_bpy_smoke_render_cube(tmp_path):
    import io
    # Skip just this test (not the whole module) when bpy is unavailable.
    pytest.importorskip("bpy")
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


# ===========================================================================
# (d) Text->scene CLI (cli.py) in --mock mode
# ===========================================================================


def _make_synthetic_asset_dir(tmp_path, cats):
    """Write a minimal processed-asset dir: <uid>/data.json + <uid>.glb."""
    import json as _json
    base = tmp_path / "assets"
    for i, (cat, desc) in enumerate(cats):
        d = base / f"uid{i:02d}"
        d.mkdir(parents=True)
        (d / "data.json").write_text(_json.dumps({
            "annotations": {"category": cat, "description": desc,
                            "onFloor": True, "onCeiling": False, "onWall": False},
            "assetMetadata": {"boundingBox": {"x": 1.0, "y": 1.0, "z": 0.4}},
        }))
        (d / f"uid{i:02d}.glb").write_text("fake")
    return base


def test_cli_mock_mode_end_to_end(tmp_path):
    import cli

    asset_dir = _make_synthetic_asset_dir(tmp_path, [
        ("bed", "a queen bed"), ("chair", "a rattan chair"),
        ("table", "a small table"), ("lamp", "a floor lamp"),
    ])
    lib_path = tmp_path / "lib.json"
    lib_path.write_text(json.dumps([
        {"category": "boulder", "description": "a heavy grey rock",
         "path": str(tmp_path / "boulder.glb"),
         "assetMetadata": {"boundingBox": {"x": 2, "y": 2, "z": 2}}},
    ]))
    out_root = tmp_path / "outputs"

    rc = cli.main([
        "--prompt", "a cozy bedroom with a bed and a chair",
        "--api_key", "sk-test1234567890",
        "--mock", "--variants",
        "--asset_dir", str(asset_dir),
        "--asset_library", str(lib_path),
        "--outputs_dir", str(out_root),
    ])
    assert rc == 0

    runs = sorted(os.listdir(out_root))
    assert len(runs) == 1
    run = out_root / runs[0]
    # run folder name is YYYYMMDD-HHMMSS in UTC
    assert len(runs[0]) == 15 and runs[0][8] == "-"

    # config + base layout + prepared task present; api key redacted
    cfg = json.load(open(run / "config.json"))
    assert cfg["prompt"] == "a cozy bedroom with a bed and a chair"
    assert cfg["mock"] is True
    assert cfg["variants"] is True
    assert cfg["api_key_redacted"].startswith("sk-t")
    assert "7890" in cfg["api_key_redacted"]
    assert "sk-test1234567890" not in json.dumps(cfg)
    assert os.path.exists(run / "layout.json")
    assert os.path.exists(run / "prepared_task.json")

    # the four named variants exist with layout.json
    for name in ["variant_01_half", "variant_02_biggest-only",
                 "variant_03_scrambled", "variant_04_worst-object"]:
        v = run / name
        assert v.is_dir(), name
        assert os.path.exists(v / "layout.json"), name

    # worst-object variant swapped the asset identity to the library candidate
    worst_task = json.load(open(run / "variant_04_worst-object" / "task.json"))
    for ent in worst_task["assets"].values():
        assert "boulder" in ent["path"]


def test_cli_missing_api_key_errors(tmp_path):
    import cli
    # Clear any inherited OPENAI_API_KEY for this test only.
    key = os.environ.pop("OPENAI_API_KEY", None)
    try:
        rc = cli.main([
            "--prompt", "x", "--mock",
            "--asset_dir", str(tmp_path),
            "--outputs_dir", str(tmp_path / "outputs"),
        ])
    finally:
        if key is not None:
            os.environ["OPENAI_API_KEY"] = key
    assert rc == 2


# ===========================================================================
# (e) Serialization safety (Q7): layouts must be JSON-serializable
# ===========================================================================


def test_export_layout_serializes_tensors():
    """Real-mode positions are torch tensors; export_layout must emit floats.

    Guards against the TypeError: Object of type Tensor is not JSON
    serializable crash when json.dump'ing a layout produced by the solver.
    Skipped when bpy/torch are unavailable (it exercises the solver stack).
    """
    import json as _json
    import torch as _torch
    pytest.importorskip("bpy")
    from src.layoutvlm.sandbox import SandBoxEnv

    task = make_synthetic_task(2)
    sb = SandBoxEnv(task, mode="one_shot")
    # Inject tensor-valued positions/rotations like the solver leaves behind.
    class _Inst:
        optimize = 2
        position = _torch.tensor([1.0, 2.0, 3.0])
        rotation = _torch.tensor([0.5, 0.5])  # cos/sin

    class _Assets:
        placements = [_Inst(), _Inst()]

    class _Local(dict):
        pass

    sb.local_vars = {"bed": _Assets()}
    # Patch task asset_var_name so var_name resolves to "bed".
    for k in task["assets"]:
        task["assets"][k]["asset_var_name"] = "bed"
    sb.task = task
    out = sb.export_layout(incomplete_scene=True)
    # Must be JSON-serializable and contain plain floats.
    s = _json.dumps(out)
    assert "1.0" in s and "2.0" in s
    for v in out.values():
        assert all(isinstance(x, float) for x in v["position"])
        assert all(isinstance(x, float) for x in v["rotation"])
