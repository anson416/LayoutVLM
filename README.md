# LayoutVLM

<div align="left">
    <a href="https://ai.stanford.edu/~sunfanyun/layoutvlm"><img src="https://img.shields.io/badge/🌐 Website-Visit-orange"></a>
    <a href=""><img src="https://img.shields.io/badge/arXiv-PDF-blue"></a>
</div>

<br>

## Installation

1. Clone this repository
2. Install dependencies (python 3.10):
```bash
pip install -r requirements.txt
```
3. Install Rotated IOU Loss (https://github.com/lilanxiao/Rotated\_IoU)
```
cd third_party/Rotated_IoU/cuda_op
python setup.py install
````

## Data preprocessing
1. Download the dataset https://drive.google.com/file/d/1WGbj8gWn-f-BRwqPKfoY06budBzgM0pu/view?usp=sharing
2. Unzip it.

Refer to https://github.com/allenai/Holodeck and https://github.com/allenai/objathor for how we preprocess Objaverse assets.

## Usage

1. Prepare a scene configuration JSON file of Objaverse assets with the following structure:
```json
{
    "task_description": ...,
    "layout_criteria": ...,
    "boundary": {
        "floor_vertices": [[x1, y1, z1], [x2, y2, z2], ...],
        "wall_height": height
    },
    "assets": {
        "asset_id": {
            "path": "path/to/asset.glb",
            "assetMetadata": {
                "boundingBox": {
                    "x": width,
                    "y": depth,
                    "z": height
                }
            }
        }
    }
}
```

2. Run LayoutVLM:
```bash
python main.py --scene_json_file path/to/scene.json --openai_api_key your_api_key
```

## Text-to-scene CLI (`cli.py`)

Generate a scene from a textual description instead of a hand-authored scene
JSON. It turns the prompt into a room boundary + asset shopping list (one
LLM call), resolves those against a processed-asset directory, runs the solver,
and writes the result to `outputs/<YYYYMMDD-HHMMSS UTC>/`. The full list of
external local resources this method requires (processed Objaverse assets,
HDRIs, optional asset library, bpy, the Rotated-IoU CUDA op, an LLM endpoint)
and how to prepare each one is documented in the **top docstring of `cli.py`**
(`python cli.py --help`, or read `cli.py`).

```bash
python cli.py \
    --prompt "a cozy beach-inspired bedroom, 4m x 5m, with a queen bed and a rattan chair" \
    --base_url https://api.openai.com/v1 \
    --api_key sk-... \
    --model gpt-4o \
    --temperature 0.0 \
    --asset_dir ./objaverse_processed \
    --hdri_dir ./vlmunr_hdri \
    --variants
```

`cli.py` has two **mutually exclusive** input modes:

* `--prompt "..."` — generate a fresh scene (+ variants with `--variants`)
  under a new `outputs/<YYYYMMDD-HHMMSS UTC>/` folder.
* `--path outputs/<timestamp>/` — skip generation entirely; only render the
  scene folders (`base/` and every `variant_*`) **already present** in that run
  folder. Useful for re-rendering a previously generated scene at different
  settings without paying the LLM/solver cost again.

And two **mutually exclusive** render modes (either can be combined with
`--prompt` or `--path`):

* `--render` — render each scene at the single baseline point only: 512 px,
  50 mm focal, pitch 0 (top-down), yaw 0, `city` env map, white (255,255,255)
  background. Saves **both** the transparent master and the white composite.
* `--render-all` — render each scene across six single-axis sweeps:
  1. resolution {196,224,256,336,384,448,512,768,1024}
  2. focal length {16,24,35,50,85,100,200}
  3. pitch {0,15,30,45,60,75,90}  (0 = top-down in the bpa convention)
  4. yaw {0,45,90,135,180,225,270,315} — **at pitch 45** (so oblique views look
     *into* the room, not straight down)
  5. env map {city,courtyard,forest,interior,night,studio,sunrise,sunset}
  6. background color {(0,0,0),(65,65,65),(118,118,118),(128,128,128),
     (186,186,186),(204,204,204),(255,255,255),(255,0,0),(0,255,0),(0,0,255)}

  Every sweep fixes its non-varied axis at the baseline (512 / 50 mm / pitch 0
  / yaw 0 / city / white) except the yaw sweep, which fixes pitch at 45.

Each run is a folder `outputs/<YYYYMMDD-HHMMSS UTC>/` containing `config.json`
(prompt + LLM config; the API key is stored **redacted** only) and a `base/`
sub-folder holding the generated scene:

```
outputs/<timestamp>/
  config.json
  base/
    task.json         # resolved + normalized task the solver/renderer consume
    layout.json       # placed instances (the scene)
    scene_spec.json   # the LLM-produced scene spec (boundary + asset list)
    meshes/           # the GLB meshes referenced by the scene, copied in
    renderings/       # only with --render / --render-all (and bpy);
                      #   ALWAYS named "renderings"
    solve_run/        # real mode only: per-group solver artifacts
  variant_01_half/
  variant_02_biggest-only/
  variant_03_scrambled/
  variant_04_worst-object/
```

Every scene folder (`base/` and each variant) has the **same** internal
structure (`task.json`, `layout.json`, `meshes/`, and `renderings/` when
rendered), so each is independently renderable and portable — the referenced
meshes are copied into `meshes/` and the task paths rewritten to `./meshes/…`,
so a folder no longer depends on the external `--asset_dir`. This is what makes
`--path` mode work: every scene folder is self-contained.

**Renderings.** Rendering follows the `bpa.py` convention: each
`(res, focal, pitch, yaw, env)` master is rendered once with a **transparent**
background (env-map lighting still applied), then each requested background
color is composited directly onto it. `fit_ratio=1` (tight-fit) is used so the
scene fills the viewport. Filenames:

```
render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>.png
render_res-<res>_focal-<focal>_pitch-<pitch>_yaw-<yaw>_env-<env>_bg-<r>-<g>-<b>.png
```

The `pitch` value in the filename is the literal bpa pitch (0 = top-down), so
filenames stay consistent across methods even if a peer method calls top-down
"90". The architectural shell (floor + walls) is retained and rendered with the
standard **dollhouse** convention: walls whose camera-facing faces would occlude
the interior are made transparent (back-face culling driven by the surface
normal), so the camera always sees into the room while far walls stay visible.
(LayoutVLM scene specs carry only a floor footprint + wall height, so there is
no door/window opening geometry to retain; the shell is a neutral floor+walls.)

* `--variants` also writes four content variants as sub-directories, **without
  re-running the LLM or solver** (they fork the base layout):
  - `variant_01_half` — keep round(n/2) instances (seeded)
  - `variant_02_biggest-only` — keep the single largest instance (bbox volume)
  - `variant_03_scrambled` — relocate every instance within the floor polygon,
    preserving z/rotation/identity (objects stay in their region)
  - `variant_04_worst-object` — swap each instance's asset identity to the
    worst-matching candidate from `--asset_library` (requires the library)
* `--mock` skips the LLM + gradient solver and places assets at random floor
  points, so the whole pipeline + variants are exercisable offline without
  GPU/Blender or API cost.
* Rendering is best-effort in `--prompt` mode: a failure for one scene (no bpy,
  a missing mesh, an HDRI problem) is logged and the rest of the pipeline still
  completes. In `--path` mode rendering is the whole point, so a missing bpy is
  a hard error.
* `--asset_library path.json` — JSON list of
  `{"category","description","path"[,"assetMetadata":{"boundingBox":...}]}`,
  required for the worst-object variant.

## Output
The script will generate a layout.json file in the specified save directory containing the optimized positions and orientations of all assets in the scene.

## BibTeX
```bibtex
@inproceedings{sun2025layoutvlm,
  title={Layoutvlm: Differentiable optimization of 3d layout via vision-language models},
  author={Sun, Fan-Yun and Liu, Weiyu and Gu, Siyi and Lim, Dylan and Bhat, Goutam and Tombari, Federico and Li, Manling and Haber, Nick and Wu, Jiajun},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={29469--29478},
  year={2025}
}
```
