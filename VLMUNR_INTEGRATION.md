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
| `vlmunr_variants.py` | Generates 6 content-variant dirs: removal (`half/quarter/eighth`) + worst-match (`alt_0/2/4`). Removal logic is pure + unit-tested; worst-match is a lazy retrieval hook that degrades gracefully. |
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
| `RESOLUTIONS` | 224, 256, 384, 448, 512, 640, 768, 1024 |
| `FOCAL_LENGTHS` | 24, 35, 50, 85, 100, 200 |
| `BACKGROUND_GRAYS` | 0, 18, 65, 117, 128, 186, 204, 255 |
| `HDRIS` | city, courtyard, forest, interior, night, studio, sunrise, sunset |
| `PITCHES` | 0, 30, 60, 90 (0 == top-down in bpa convention) |
| `YAWS` | 0, 30, ..., 330 |

Baseline: `RES=512, FOCAL=50, BG=(128,128,128), HDRI=city, PITCH=0, YAW=0`.

`phase_levels(phase)` varies exactly one axis (all others held at baseline):

| Phase | Sweeps |
|-------|--------|
| `1a` | resolution |
| `1b` | background gray |
| `1c` | HDRI (re-initializes Blender per HDRI) |
| `1d` | focal length |
| `2`  | camera pose = pitch x yaw |

## Content variants (`vlmunr_variants.py`)

Sibling dirs of the scene dir, each with a variant `layout.json` + task copy:

| Dir | Effect |
|-----|--------|
| `variant_half` / `variant_quarter` / `variant_eighth` | Keep `round(n/k)` instances (>= 1), seeded `random.Random(seed).sample` over **sorted** instance ids -> deterministic and order-independent. |
| `variant_alt_0` / `variant_alt_2` / `variant_alt_4` | Worst-match asset substitution. Lazy hook embeds category/description text (transformers/torch, imported inside the function) and picks an ascending-similarity (worst-first) candidate at the given rank offset. Falls back to a pure-Python token-overlap ranking when the ML stack is unavailable. Writes `variant_intent.json`. |

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
- `pytest tests/ -x -q` -> **20 passed** (filename exact-strings, phase-level
  shapes, render-count enumeration, removal-fraction + clamp + determinism +
  order-independence, layout/asset join incl. orphan drop, transform math
  incl. -90 pre-rotation and floor `z = bbox.z/2`, and a headless bpy smoke
  render of a primitive cube producing a non-empty PNG + composite).
- Standalone headless render through `vlmunr_bpa` (initialize with `city.exr`
  -> cube -> `render_perspective` -> `add_bg_to_rgba`) writes non-empty master
  and composite PNGs.
- Variant generator produces the 6 sibling dirs with correct kept-counts and
  graceful worst-match degradation (recorded intent, scene unchanged) when no
  asset library is configured; the worst-match scorer ranks an unrelated
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
