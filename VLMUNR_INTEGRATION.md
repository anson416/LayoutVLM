# VLM-Unreliability Audit Integration

A self-contained rendering + content-variant layer that turns LayoutVLM scene
generations into a controlled image sweep for auditing vision-language models.
Everything here is vendored into the LayoutVLM repo root; there is **no**
dependency on an external `vlmunr` package.

## Files added

| File | Purpose |
|------|---------|
| `vlmunr_bpa.py` | Vendored Blender-Python API (Builder / Renderer). Copied **verbatim** from `vlm-unreliability/vlmunr/blender/bpa.py`. Do not edit. |
| `vlmunr_hdri/` | The 8 environment maps (`city, courtyard, forest, interior, night, studio, sunrise, sunset` `.exr`) + `license.txt`. |
| `vlmunr_config.py` | Fixed factor levels, baselines, and the `phase_levels(phase)` helper. No third-party imports. |
| `vlmunr_render.py` | Joins `layout.json` to the input task JSON, builds the Blender scene honoring the LayoutVLM transform, sweeps a phase, writes PNGs. Pure functions for filenames/transform/join are bpy-free. |
| `vlmunr_variants.py` | Generates content-variant dirs: removal (`half/quarter/eighth`), worst-match (`alt_0/2/4`), category-aware substitution (`subst_within`/`subst_cross`), and layout `scramble`. Removal + scramble logic is pure + unit-tested; substitution hooks degrade gracefully. |
| `tests/test_vlmunr_integration.py` | pytest: filename exact-strings, removal/join logic, transform math, and a bpy smoke render. |
| `tests/conftest.py` | Snapshots the real terminal fds so the bpy smoke test survives pytest capture. |
| `gen.sh` | Driver: loop `benchmark_tasks/*/*.json` -> `python main.py ... --save_dir ...`. |
| `run.sh` | Driver: loop save dirs -> `vlmunr_variants.py` then `vlmunr_render.py`. |

## Run commands

```bash
PY=/Users/anson/miniforge3/envs/vlmunr/bin/python

# Unit + smoke tests
cd /Users/anson/Projects/LayoutVLM
$PY -m pytest tests/ -x -q

# 1. Generate layouts (needs OpenAI key + processed Objaverse assets; downloads nothing here)
OPENAI_API_KEY=sk-... ASSET_DIR=./objaverse_processed ./gen.sh ./results

# 2. Variants + render sweep for every scene
./run.sh ./results 1a        # phase 1a (resolution sweep); use all for every phase

# Direct invocation
$PY vlmunr_variants.py --scene-dir results/bedroom_0 --seed 42
$PY vlmunr_render.py   --scene-dir results/bedroom_0 --phase 1a
$PY vlmunr_render.py   --task-json benchmark_tasks/bedroom/bedroom_0.json \
                       --layout-json results/bedroom_0/layout.json --phase 2
```

## Filename scheme

Renders land in `<scene-dir>/renderings/`. Two-phase: a transparent **master**
is rendered once per `(res, focal, pitch, yaw, hdri)`, then each background gray
is composited onto it.

```
master:     render_{res}_{focal}_{pitch}_{yaw}_{hdri}.png
composite:  render_{res}_{focal}_{r}_{g}_{b}_{pitch}_{yaw}_{hdri}.png
```

Examples: `render_512_50_0_0_city.png`,
`render_512_50_128_128_128_0_0_city.png`.

## Factor levels (`vlmunr_config.py`)

| Axis | Levels |
|------|--------|
| `RESOLUTIONS` | 196, 224, 256, 336, 384, 448, 512, 768, 1024 (9) |
| `FOCAL_LENGTHS` | 16, 24, 35, 50, 85, 100, 200 (7) |
| `BACKGROUND_GRAYS` | 0, 65, 128, 186, 204, 255 (6) |
| `BACKGROUND_CHROMATIC` | (255,0,0), (0,255,0), (0,0,255) (3) |
| `FLOOR_TEXTURE_BACKGROUND` | `"floor_texture"` sentinel — documented for paper Table 1 parity, **NOT rendered** (no in-repo floor-texture compositing path) |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset (8) |
| `PITCHES` | 0, 15, 30, 45, 60, 75, 90 (7; 0 == top-down in bpa convention) |
| `YAWS` | 0, 45, 90, 135, 180, 225, 270, 315 (8) |

Baseline: `RES=512, FOCAL=50, BG=(128,128,128), HDRI=city, PITCH=0, YAW=0`.
`BASELINE_YAW_PITCH=45` is the fixed pitch used when sweeping yaw.

`phase_levels(phase)` returns a `Dict[str, List]` and varies exactly one axis
(all others held at baseline):

| Phase | Sweeps |
|-------|--------|
| `1a` | resolution |
| `1b` | background gray |
| `1b_chroma` | chromatic backgrounds (R/G/B) |
| `1c` | HDRI (re-initializes Blender per HDRI) |
| `1d` | focal length |
| `2`  | camera pose = pitch x yaw (kept for backward compat) |
| `2_pitch` | pitch sweep at baseline yaw (0) |
| `2_yaw` | yaw sweep at fixed pitch (45) |

`PHASES` (== `ALL_PHASES`) = `["1a", "1b", "1b_chroma", "1c", "1d", "2", "2_pitch", "2_yaw"]`.

## Content variants (`vlmunr_variants.py`)

Sibling dirs of the scene dir, each with a variant `layout.json` + task copy:

| Dir | Effect |
|-----|--------|
| `variant_half` / `variant_quarter` / `variant_eighth` | Keep `round(n/k)` instances (>= 1), seeded `random.Random(seed).sample` over **sorted** instance ids -> deterministic and order-independent. |
| `variant_alt_0` / `variant_alt_2` / `variant_alt_4` | Worst-match asset substitution at a rank offset. Lazy hook ranks candidates by ascending category/description text similarity (worst-first). The embedding path (transformers/torch) is **opt-in** via `VLMUNR_USE_EMBED_MODEL=1`; the default is a pure-Python token-overlap ranking (no model download). Writes `variant_intent.json`. |
| `variant_subst_within` / `variant_subst_cross` | Category-aware worst-match: `within` swaps toward a different asset of the **same** category; `cross` swaps toward a **different** category. Same lazy scoring hook, constrained by category. With no `VLMUNR_ASSET_LIBRARY` present the hook degrades gracefully, recording per-instance intent `{instance_id: mode}` in `variant_intent.json` and leaving the scene unchanged. |
| `variant_scramble` | Relocate **every** instance to a random `(x, y)` within the floor polygon (rejection-sampled inside the polygon, else within its bbox). Pure deterministic function of `(layout_dict, floor_vertices, seed)`; preserves the instance set/ids/count and each instance's `rotation` and `z` (floor objects keep `z = bbox.z/2`). |

Worst-match / substitution candidates come from a JSON library at
`VLMUNR_ASSET_LIBRARY` (records of `{"category","description","path"}`). When
absent, the substitution hooks no-op-record intent (graceful degradation) so
the pipeline never crashes — LayoutVLM ships no in-repo retrieval index.

## LayoutVLM coordinate convention (honored by the renderer)

- `layout.json` is a dict keyed by `<objaverse_uid>-<idx>`, each value
  `{"position":[x,y,z], "rotation":[rx,ry,rz]}` in **meters / DEGREES**, with
  **z = world UP** and only `rz` meaningful. It contains no asset path/bbox; the
  renderer **joins** it back to the task JSON `assets` map (`path`,
  `assetMetadata.boundingBox`, `category`, `description`, `onFloor`).
- Per-object transform (`compute_object_transform`):
  1. recenter mesh (`origin -> BOUNDS`, done by `bpa.transform`),
  2. pre-rotate **-90 deg about Z** (preprocessed meshes face -y -> +x),
  3. add the layout `rotation` (deg about Z): final `rz = -90 + layout_rz`,
  4. locate at `position`; for floor objects `z = boundingBox.z / 2`,
  5. scale = identity (assets authored in meters).
- Camera target/radius are derived from `boundary.floor_vertices` /
  `boundary.wall_height`, with bpa's bounding-sphere fit as the framing method.

## VERIFICATION STATUS

**Verified (this environment, conda env `vlmunr`, bpy 5.1.2 + torch):**

- `ast.parse` syntax check passes on all new `.py` files.
- `pytest tests/ -q` -> **34 passed** (filename exact-strings, phase-level
  shapes, paper Table 1 factor-level counts + exact values, the `1b_chroma`
  chromatic phase, `2_pitch` at yaw 0 and `2_yaw` at pitch 45, render-count
  enumeration, removal-fraction + clamp + determinism + order-independence,
  layout/asset join incl. orphan drop, layout-scramble determinism + in-bounds
  + preserved ids/rotation/z + order-independence, within-/cross-category
  substitution intent recording + graceful degradation + a library-backed
  scoring path, transform math incl. -90 pre-rotation and floor `z = bbox.z/2`,
  and a headless bpy smoke render of a primitive cube producing a non-empty
  PNG + composite).
- Standalone headless render through `vlmunr_bpa` (initialize with `city.exr`
  -> cube -> `render_perspective` -> `add_bg_to_rgba`) writes non-empty master
  and composite PNGs.
- Variant generator produces the sibling dirs with correct kept-counts,
  graceful worst-match / category-substitution degradation (recorded intent
  `{instance_id: mode}`, scene unchanged) when no asset library is configured,
  and a layout-scramble dir; the worst-match scorer ranks an unrelated
  candidate as worst via the offline lexical fallback (no model downloaded).

**NOT validated (requires assets / services not present here):**

- **Real-asset coordinate correctness.** The transform math is unit-tested and
  the synthetic cube renders, but no real Objaverse `.glb` was imported, so the
  exact visual placement (recenter / -90 / rz / floor-z) against real meshes is
  **not** pixel-verified. Needs processed Objaverse assets under `ASSET_DIR`.
- **Worst-match retrieval quality.** The structured hook and its lexical
  fallback are exercised, but the embedding-model path (transformers) was not
  run against a real Objaverse library (none ships in-repo). Needs an Objaverse
  asset DB and, for the ML path, the embedding model + an API/download step
  (intentionally not performed).
- **Full layout generation** (`main.py` / `gen.sh`) was not run: it needs an
  OpenAI API key, processed assets, and the CUDA Rotated-IoU op.
